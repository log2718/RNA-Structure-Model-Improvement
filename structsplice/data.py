"""Loading, validating and splitting the PNAS splicing datasets.

The shipped ``train_data.npz`` / ``test_data.npz`` files are read-only inputs.
Nothing in this module writes to them. The PNAS train/test split is taken
exactly as shipped: the two files are never pooled, reshuffled or re-split
against each other, and the test file is only ever read by
``structsplice.evaluate``.

Verified dataset facts (see ``README.md`` for the full audit):

* ``model_sequence`` is ``CATCCAGGTT`` + 70 nt ``exon`` + ``CAGGTCTGAC``,
  exactly 90 nt, in the DNA alphabet, for every row of both splits.
* ``metadata_PSI`` equals
  ``num_exon_inclusion / (num_exon_inclusion + num_exon_skipping)`` to within
  floating-point rounding, with a non-zero denominator everywhere.
* ``structure`` is the RNAfold ``-T 37`` MFE dot-bracket structure of
  ``model_sequence``, reproduced exactly by :func:`structsplice.vienna.run_rnafold`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import (
    LEFT_FLANK,
    MODEL_SEQUENCE_LENGTH,
    RIGHT_FLANK,
)

#: npz fields this project reads. Large pre-encoded arrays (``seq_oh``,
#: ``struct_oh``, ``wobbles``) are deliberately skipped: features are rebuilt
#: by :mod:`structsplice.encoding`, which was verified to reproduce them exactly.
REQUIRED_FIELDS = ("exon", "model_sequence", "metadata_PSI")
OPTIONAL_FIELDS = (
    "structure",
    "mfe",
    "metadata_num_exon_inclusion",
    "metadata_num_exon_skipping",
    "metadata_total",
)


class DatasetValidationError(ValueError):
    """Raised when a dataset file fails a structural or label check."""


@dataclasses.dataclass
class SplicingDataset:
    """A loaded PSI dataset.

    Attributes:
        exon: Unflanked 70 nt exon sequence per row. Doubles as the grouping
            key, since the PNAS library identifies a variant by its exon.
        model_sequence: Full flanked model input per row (90 nt).
        psi: Measured PSI in ``[0, 1]`` per row.
        structure: Precomputed RNAfold MFE dot-bracket string per row, or
            ``None`` when the file does not carry it.
        n_inclusion: Inclusion read count per row, or ``None``.
        n_skipping: Skipping read count per row, or ``None``.
        source: Path the dataset was loaded from.
        name: Short label used in logs and output filenames.
    """

    exon: np.ndarray
    model_sequence: np.ndarray
    psi: np.ndarray
    structure: np.ndarray | None = None
    n_inclusion: np.ndarray | None = None
    n_skipping: np.ndarray | None = None
    source: Path | None = None
    name: str = ""

    def __len__(self) -> int:
        return len(self.psi)

    @property
    def read_depth(self) -> np.ndarray | None:
        """Return ``n_inclusion + n_skipping`` per row, or ``None``.

        This is the denominator of the PSI ratio, i.e. the number of reads that
        actually inform each label, not the total read count including
        non-canonical products.
        """
        if self.n_inclusion is None or self.n_skipping is None:
            return None
        return (self.n_inclusion + self.n_skipping).astype(np.float64)

    def subset(self, indices: Sequence[int] | np.ndarray) -> "SplicingDataset":
        """Return a new dataset holding only ``indices``.

        Args:
            indices: Row indices to keep.

        Returns:
            A dataset with every present field indexed consistently.
        """
        idx = np.asarray(indices, dtype=np.int64)
        return SplicingDataset(
            exon=self.exon[idx],
            model_sequence=self.model_sequence[idx],
            psi=self.psi[idx],
            structure=None if self.structure is None else self.structure[idx],
            n_inclusion=(
                None if self.n_inclusion is None else self.n_inclusion[idx]
            ),
            n_skipping=(
                None if self.n_skipping is None else self.n_skipping[idx]
            ),
            source=self.source,
            name=self.name,
        )

    def summary(self) -> str:
        """Return a one-paragraph human-readable description."""
        lengths = sorted({len(s) for s in self.model_sequence})
        return (
            f"{self.name or 'dataset'}: n={len(self)}  "
            f"model_sequence lengths={lengths}  "
            f"unique exons={len(set(self.exon.tolist()))}  "
            f"PSI mean={self.psi.mean():.4f} "
            f"min={self.psi.min():.4f} max={self.psi.max():.4f}  "
            f"PSI==0: {int((self.psi == 0).sum())} "
            f"PSI==1: {int((self.psi == 1).sum())}"
        )


def load_npz(
    path: str | Path,
    name: str = "",
    validate: bool = True,
    expect_length: int | None = MODEL_SEQUENCE_LENGTH,
) -> SplicingDataset:
    """Load a PNAS-format ``.npz`` dataset.

    Args:
        path: Path to ``train_data.npz`` or ``test_data.npz``.
        name: Short label for logs; defaults to the file stem.
        validate: Run :func:`validate_dataset` after loading.
        expect_length: Expected model-sequence length, or ``None`` to accept
            any single consistent length.

    Returns:
        The loaded dataset.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        DatasetValidationError: If required fields are missing, or validation
            fails.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")

    with np.load(file_path, allow_pickle=False) as z:
        available = set(z.files)
        missing = [f for f in REQUIRED_FIELDS if f not in available]
        if missing:
            raise DatasetValidationError(
                f"{file_path} is missing required field(s) {missing}. "
                f"Available: {sorted(available)}"
            )

        def maybe(field: str) -> np.ndarray | None:
            return z[field] if field in available else None

        dataset = SplicingDataset(
            exon=np.asarray(z["exon"]).astype(str),
            model_sequence=np.asarray(z["model_sequence"]).astype(str),
            psi=np.asarray(z["metadata_PSI"], dtype=np.float64),
            structure=(
                None
                if "structure" not in available
                else np.asarray(z["structure"]).astype(str)
            ),
            n_inclusion=_as_int(maybe("metadata_num_exon_inclusion")),
            n_skipping=_as_int(maybe("metadata_num_exon_skipping")),
            source=file_path,
            name=name or file_path.stem,
        )

    if validate:
        validate_dataset(dataset, expect_length=expect_length)
    return dataset


def _as_int(array: np.ndarray | None) -> np.ndarray | None:
    """Return ``array`` as int64, or ``None`` if it is ``None``."""
    return None if array is None else np.asarray(array, dtype=np.int64)


def validate_dataset(
    dataset: SplicingDataset,
    expect_length: int | None = MODEL_SEQUENCE_LENGTH,
    expect_flanks: bool = True,
) -> None:
    """Check a dataset's shapes, alphabet and PSI labels.

    Args:
        dataset: The dataset to check.
        expect_length: Required model-sequence length, or ``None`` to only
            require that all rows share one length.
        expect_flanks: Require the fixed PNAS vector flanks around each exon.

    Raises:
        DatasetValidationError: On the first failed check, with a message
            naming the offending field and row.
    """
    n = len(dataset.psi)
    if n == 0:
        raise DatasetValidationError("dataset is empty")

    for field in ("exon", "model_sequence"):
        arr = getattr(dataset, field)
        if len(arr) != n:
            raise DatasetValidationError(
                f"field {field!r} has {len(arr)} rows but metadata_PSI has {n}"
            )
    if dataset.structure is not None and len(dataset.structure) != n:
        raise DatasetValidationError(
            f"field 'structure' has {len(dataset.structure)} rows but "
            f"metadata_PSI has {n}"
        )

    # ── sequence geometry ──
    lengths = {len(s) for s in dataset.model_sequence}
    if len(lengths) != 1:
        raise DatasetValidationError(
            f"model_sequence rows have differing lengths {sorted(lengths)}; "
            f"this pipeline supports variable lengths at the batch level but "
            f"a single dataset file is expected to be uniform"
        )
    length = lengths.pop()
    if expect_length is not None and length != expect_length:
        raise DatasetValidationError(
            f"model_sequence length is {length} but {expect_length} was "
            f"expected. Pass expect_length=None to accept it."
        )

    allowed = set("ACGUTN")
    for i, seq in enumerate(dataset.model_sequence):
        bad = set(seq.upper()) - allowed
        if bad:
            raise DatasetValidationError(
                f"model_sequence row {i} holds unexpected characters "
                f"{sorted(bad)}"
            )
        if expect_flanks and not (
            seq.startswith(LEFT_FLANK) and seq.endswith(RIGHT_FLANK)
        ):
            raise DatasetValidationError(
                f"model_sequence row {i} is not flanked by {LEFT_FLANK}/"
                f"{RIGHT_FLANK}: {seq!r}"
            )
        if expect_flanks:
            core = seq[len(LEFT_FLANK) : len(seq) - len(RIGHT_FLANK)]
            if core != dataset.exon[i]:
                raise DatasetValidationError(
                    f"row {i}: model_sequence core {core!r} does not equal "
                    f"exon {dataset.exon[i]!r}"
                )

    # ── structure geometry ──
    if dataset.structure is not None:
        for i, (seq, dot) in enumerate(
            zip(dataset.model_sequence, dataset.structure)
        ):
            if len(dot) != len(seq):
                raise DatasetValidationError(
                    f"structure row {i} has length {len(dot)} but the sequence "
                    f"has length {len(seq)}"
                )
            bad = set(dot) - set("().")
            if bad:
                raise DatasetValidationError(
                    f"structure row {i} holds unexpected symbols {sorted(bad)}"
                )

    # ── PSI labels ──
    psi = dataset.psi
    if not np.isfinite(psi).all():
        n_bad = int((~np.isfinite(psi)).sum())
        raise DatasetValidationError(
            f"metadata_PSI holds {n_bad} non-finite value(s)"
        )
    if psi.min() < 0.0 or psi.max() > 1.0:
        raise DatasetValidationError(
            f"metadata_PSI must lie in [0, 1]; observed range "
            f"[{psi.min()}, {psi.max()}]"
        )

    # ── read counts, when present ──
    if dataset.n_inclusion is not None and dataset.n_skipping is not None:
        if (dataset.n_inclusion < 0).any() or (dataset.n_skipping < 0).any():
            raise DatasetValidationError("read counts must be non-negative")
        depth = dataset.n_inclusion + dataset.n_skipping
        if (depth == 0).any():
            raise DatasetValidationError(
                f"{int((depth == 0).sum())} row(s) have zero "
                f"inclusion+skipping reads, so their PSI is undefined"
            )
        implied = dataset.n_inclusion / depth
        max_dev = float(np.max(np.abs(implied - psi)))
        if max_dev > 1e-6:
            raise DatasetValidationError(
                f"metadata_PSI disagrees with inclusion/(inclusion+skipping) "
                f"by up to {max_dev:.3g}"
            )


# ── Train/validation splitting ──────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SplitIndices:
    """Row indices of a reproducible train/validation split.

    Attributes:
        train: Row indices used for gradient updates.
        val: Row indices used for early stopping and model selection only.
        seed: The seed that produced the split.
        group_key: Name of the field groups were formed on, or ``"row"`` when
            grouping was disabled.
        n_groups: Number of distinct groups across both sides.
        n_train_groups: Number of groups assigned to training.
        n_val_groups: Number of groups assigned to validation.
    """

    train: np.ndarray
    val: np.ndarray
    seed: int
    group_key: str
    n_groups: int
    n_train_groups: int
    n_val_groups: int


def grouped_train_val_split(
    dataset: SplicingDataset,
    val_fraction: float = 0.1,
    seed: int = 42,
    group_by: str = "exon",
) -> SplitIndices:
    """Split a dataset into train and validation rows, grouped to stop leakage.

    Rows sharing a group key are always assigned to the same side. Grouping on
    ``exon`` matters because the PNAS training split contains 23 exons measured
    more than once; an ungrouped random split would put one replicate of the
    same molecule in training and another in validation, inflating validation
    scores and corrupting early stopping.

    The split is a function of ``(seed, group_by, sorted group keys)`` only, so
    it is identical across machines, runs and feature configurations. That last
    property matters for fairness: models A, B and C see byte-identical train
    and validation rows.

    Args:
        dataset: Dataset to split. Only the training file should be passed; the
            PNAS test file must never be split or used for model selection.
        val_fraction: Target fraction of *groups* held out for validation.
        seed: Random seed.
        group_by: ``"exon"`` to group by exon sequence, ``"model_sequence"`` to
            group by the full flanked input, or ``"row"`` to disable grouping.

    Returns:
        The split indices.

    Raises:
        ValueError: On a bad ``val_fraction``, unknown ``group_by``, or when
            the split would leave either side empty.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"val_fraction must lie strictly between 0 and 1; got {val_fraction}"
        )

    n = len(dataset)
    if group_by == "row":
        keys = np.arange(n).astype(str)
    elif group_by in ("exon", "model_sequence"):
        keys = np.asarray(getattr(dataset, group_by)).astype(str)
    else:
        raise ValueError(
            f"unknown group_by {group_by!r}; expected 'exon', "
            f"'model_sequence' or 'row'"
        )

    unique = np.array(sorted(set(keys.tolist())))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    n_val_groups = int(round(len(unique) * val_fraction))
    n_val_groups = max(1, min(len(unique) - 1, n_val_groups))
    val_groups = set(unique[order[:n_val_groups]].tolist())

    is_val = np.fromiter(
        (k in val_groups for k in keys.tolist()), dtype=bool, count=n
    )
    train_idx = np.nonzero(~is_val)[0]
    val_idx = np.nonzero(is_val)[0]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"val_fraction={val_fraction} produced an empty side "
            f"({len(train_idx)} train / {len(val_idx)} val rows)"
        )

    return SplitIndices(
        train=train_idx,
        val=val_idx,
        seed=seed,
        group_key=group_by,
        n_groups=len(unique),
        n_train_groups=len(unique) - n_val_groups,
        n_val_groups=n_val_groups,
    )


def overlapping_groups(
    left: SplicingDataset, right: SplicingDataset, group_by: str = "exon"
) -> list[str]:
    """Return group keys present in both datasets.

    Used to report, not repair, overlap between the shipped PNAS train and test
    splits. The split is preserved exactly as published.

    Args:
        left: First dataset.
        right: Second dataset.
        group_by: Field to compare, ``"exon"`` or ``"model_sequence"``.

    Returns:
        Sorted list of shared keys.
    """
    a = set(np.asarray(getattr(left, group_by)).astype(str).tolist())
    b = set(np.asarray(getattr(right, group_by)).astype(str).tolist())
    return sorted(a & b)
