"""Torch dataset and padding-aware collation.

The PNAS corpus is uniformly 90 nt, so padding is a no-op on this data. The
collate function still pads to the longest sequence in each batch and emits a
boolean mask, because the pooling and LSTM paths must stay correct if the
project later moves to longer or ragged genomic context. The variable-length
path is exercised directly by the test suite.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import AccessibilityCache, FoldCache
from .config import FEATURE_CONFIGS, input_dim
from .data import SplicingDataset
from .encoding import build_features


class PSIFeatureDataset(Dataset):
    """Per-example features and PSI targets for one feature configuration.

    Attributes:
        config: Feature configuration name, ``"A"``, ``"B"`` or ``"C"``.
        input_dim: Per-position channel count implied by ``config``.
    """

    def __init__(
        self,
        dataset: SplicingDataset,
        config: str,
        fold_cache: FoldCache | None = None,
        accessibility_cache: AccessibilityCache | None = None,
        use_npz_structure: bool = True,
        weights: np.ndarray | None = None,
    ) -> None:
        """Build a feature dataset.

        Args:
            dataset: The loaded split.
            config: ``"A"``, ``"B"`` or ``"C"``.
            fold_cache: RNAfold cache. Required for config ``"B"`` unless the
                dataset carries a ``structure`` field and ``use_npz_structure``
                is true.
            accessibility_cache: RNAplfold cache. Required for config ``"C"``.
            use_npz_structure: Prefer the dataset's own precomputed
                ``structure`` field for config ``"B"``. It was verified to equal
                ``RNAfold -T 37`` output on this corpus, so this avoids
                refolding 240k sequences without changing the features.
            weights: Optional per-row loss weights, e.g. read depth.

        Raises:
            ValueError: If ``config`` is unknown, a required feature source is
                missing, or ``weights`` has the wrong length.
        """
        if config not in FEATURE_CONFIGS:
            raise ValueError(
                f"unknown feature config {config!r}; expected one of "
                f"{sorted(FEATURE_CONFIGS)}"
            )
        use_db, use_acc = FEATURE_CONFIGS[config]

        self.config = config
        self.input_dim = input_dim(config)
        self._data = dataset
        self._use_db = use_db
        self._use_acc = use_acc
        self._fold_cache = fold_cache
        self._acc_cache = accessibility_cache
        self._npz_structure = (
            dataset.structure if (use_npz_structure and use_db) else None
        )

        if use_db and self._npz_structure is None and fold_cache is None:
            raise ValueError(
                "config 'B' needs dot-bracket structures: either the dataset "
                "must carry a 'structure' field or an RNAfold cache must be "
                "supplied. Build one with scripts/preprocess_structures.py."
            )
        if use_acc and accessibility_cache is None:
            raise ValueError(
                "config 'C' needs an RNAplfold accessibility cache. Build one "
                "with: scripts/preprocess_structures.py --tool rnaplfold"
            )

        if weights is not None:
            weights = np.asarray(weights, dtype=np.float32)
            if len(weights) != len(dataset):
                raise ValueError(
                    f"weights has {len(weights)} entries but the dataset has "
                    f"{len(dataset)} rows"
                )
            if (weights < 0).any():
                raise ValueError("loss weights must be non-negative")
        self._weights = weights

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one example.

        Args:
            index: Row index.

        Returns:
            Dict with ``features`` ``(length, input_dim)``, scalar ``target``,
            scalar ``weight``, scalar ``length`` and scalar ``index``.
        """
        sequence = str(self._data.model_sequence[index])

        structure = None
        if self._use_db:
            if self._npz_structure is not None:
                structure = str(self._npz_structure[index])
            else:
                structure = self._fold_cache.get(sequence)

        accessibility = None
        if self._use_acc:
            accessibility = self._acc_cache.get(sequence)

        features = build_features(sequence, structure, accessibility)
        weight = (
            1.0 if self._weights is None else float(self._weights[index])
        )
        return {
            "features": torch.from_numpy(features),
            "target": torch.tensor(
                float(self._data.psi[index]), dtype=torch.float32
            ),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "length": torch.tensor(len(features), dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


def collate_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a list of examples into a batch and build the padding mask.

    Args:
        items: Examples from :class:`PSIFeatureDataset`.

    Returns:
        Dict with:

        * ``features`` ``(batch, max_len, input_dim)``, zero-padded
        * ``mask`` ``(batch, max_len)`` bool, ``True`` at real positions
        * ``lengths`` ``(batch,)`` long, true sequence lengths
        * ``targets``, ``weights`` ``(batch,)`` float32
        * ``indices`` ``(batch,)`` long, original dataset rows

    Raises:
        ValueError: If ``items`` is empty or channel counts disagree.
    """
    if not items:
        raise ValueError("cannot collate an empty batch")

    channels = {int(item["features"].shape[1]) for item in items}
    if len(channels) != 1:
        raise ValueError(
            f"batch mixes feature dimensions {sorted(channels)}; every example "
            f"must use the same feature configuration"
        )
    n_channels = channels.pop()
    lengths = torch.stack([item["length"] for item in items])
    max_len = int(lengths.max())

    features = torch.zeros(
        (len(items), max_len, n_channels), dtype=torch.float32
    )
    mask = torch.zeros((len(items), max_len), dtype=torch.bool)
    for i, item in enumerate(items):
        length = int(item["length"])
        features[i, :length] = item["features"]
        mask[i, :length] = True

    return {
        "features": features,
        "mask": mask,
        "lengths": lengths,
        "targets": torch.stack([item["target"] for item in items]),
        "weights": torch.stack([item["weight"] for item in items]),
        "indices": torch.stack([item["index"] for item in items]),
    }
