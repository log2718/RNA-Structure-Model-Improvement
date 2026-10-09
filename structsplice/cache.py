"""On-disk caches for RNAfold and RNAplfold features.

Both caches are keyed by a SHA-256 hash of the **RNA-normalised sequence that
was actually folded**, never by row index or exon identifier. Three
consequences follow:

* Duplicate sequences are folded once and shared, so the 23 replicate
  measurements in the PNAS training split cost nothing extra.
* A cache built from the training split is reusable for the test split without
  any possibility of label leakage: nothing but the nucleotide string enters
  the key, and no PSI value, split assignment or label statistic is read during
  preprocessing.
* Mutated sequences are distinct keys automatically, because each variant's
  full 90 nt model sequence (constant flanks included) hashes differently.

Caches are stored as ``.npz`` files with parallel arrays and may be written in
shards, so an HPC array job can fold a slice of the corpus per task and a
later ``merge`` step combines them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import N_ACCESSIBILITY_CHANNELS
from .vienna import to_rna


def sequence_key(sequence: str) -> str:
    """Return the cache key for a sequence.

    The sequence is normalised to RNA first, so the same molecule supplied as
    DNA or RNA maps to one key.

    Args:
        sequence: Nucleotide sequence.

    Returns:
        Lowercase hex SHA-256 digest.
    """
    return hashlib.sha256(to_rna(sequence).encode("ascii")).hexdigest()


def unique_sequences(sequences: Iterable[str]) -> list[str]:
    """Return the RNA-normalised unique sequences, in first-seen order.

    Args:
        sequences: Any iterable of nucleotide sequences.

    Returns:
        Deduplicated list preserving first appearance.
    """
    seen: dict[str, None] = {}
    for seq in sequences:
        seen.setdefault(to_rna(seq), None)
    return list(seen)


class FoldCache:
    """Cache of RNAfold MFE dot-bracket structures and free energies.

    Holds the minimum-free-energy secondary structure as a dot-bracket string
    per sequence. It does not store base-pairing probabilities; those would
    require ``RNAfold -p`` and are not used by any model here.
    """

    #: Marker written into the file so a mismatched cache type is caught early.
    KIND = "rnafold_mfe"

    def __init__(self, params_tag: str = "") -> None:
        """Create an empty cache.

        Args:
            params_tag: Identifier for the folding parameters, recorded in the
                file and checked on load.
        """
        self.params_tag = params_tag
        self.structures: dict[str, str] = {}
        self.mfe: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self.structures)

    def __contains__(self, sequence: str) -> bool:
        return sequence_key(sequence) in self.structures

    def add(self, sequence: str, structure: str, mfe: float) -> None:
        """Insert one folded sequence.

        Args:
            sequence: The folded nucleotide sequence.
            structure: Its dot-bracket structure.
            mfe: Its minimum free energy in kcal/mol.

        Raises:
            ValueError: If the structure length differs from the sequence.
        """
        if len(structure) != len(to_rna(sequence)):
            raise ValueError(
                f"structure length {len(structure)} does not match sequence "
                f"length {len(sequence)}"
            )
        key = sequence_key(sequence)
        self.structures[key] = structure
        self.mfe[key] = float(mfe)

    def get(self, sequence: str) -> str:
        """Return the cached dot-bracket structure for a sequence.

        Args:
            sequence: Nucleotide sequence.

        Returns:
            The dot-bracket string.

        Raises:
            KeyError: If the sequence is not cached.
        """
        key = sequence_key(sequence)
        try:
            return self.structures[key]
        except KeyError:
            raise KeyError(
                f"no cached RNAfold structure for sequence starting "
                f"{to_rna(sequence)[:24]!r}. Run "
                f"scripts/preprocess_structures.py to build the cache."
            ) from None

    def missing(self, sequences: Sequence[str]) -> list[str]:
        """Return the RNA-normalised unique sequences absent from the cache."""
        return [s for s in unique_sequences(sequences) if s not in self]

    def save(self, path: str | Path) -> Path:
        """Write the cache to a compressed ``.npz`` file.

        Args:
            path: Destination path; parent directories are created.

        Returns:
            The resolved destination path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.structures)
        np.savez_compressed(
            out,
            kind=np.array(self.KIND),
            params_tag=np.array(self.params_tag),
            keys=np.array(keys, dtype=object).astype("U64")
            if keys
            else np.zeros(0, dtype="U64"),
            structures=np.array(
                [self.structures[k] for k in keys], dtype=object
            ).astype("U")
            if keys
            else np.zeros(0, dtype="U1"),
            mfe=np.array([self.mfe[k] for k in keys], dtype=np.float32),
        )
        return out

    @classmethod
    def load(cls, paths: str | Path | Sequence[str | Path]) -> "FoldCache":
        """Load and merge one or more cache files.

        Args:
            paths: A single path, or several paths/shards to merge.

        Returns:
            The merged cache.

        Raises:
            FileNotFoundError: If a path does not exist.
            ValueError: If a file is not an RNAfold cache.
        """
        items = [paths] if isinstance(paths, (str, Path)) else list(paths)
        cache = cls()
        for item in items:
            p = Path(item)
            if not p.is_file():
                raise FileNotFoundError(f"cache file not found: {p}")
            with np.load(p, allow_pickle=False) as z:
                kind = str(z["kind"])
                if kind != cls.KIND:
                    raise ValueError(
                        f"{p} is a {kind!r} cache, not {cls.KIND!r}"
                    )
                cache.params_tag = cache.params_tag or str(z["params_tag"])
                for key, structure, mfe in zip(
                    z["keys"], z["structures"], z["mfe"]
                ):
                    cache.structures[str(key)] = str(structure)
                    cache.mfe[str(key)] = float(mfe)
        return cache


class AccessibilityCache:
    """Cache of RNAplfold unpaired-stretch accessibility profiles.

    Stores one ``(n_channels, length)`` block per sequence plus the boolean
    ``NA`` mask RNAplfold reported, so boundary substitutions stay auditable
    and are never confused with measured zero probabilities.
    """

    KIND = "rnaplfold_accessibility"

    def __init__(
        self,
        params_tag: str = "",
        na_policy: str = "",
        n_channels: int = N_ACCESSIBILITY_CHANNELS,
    ) -> None:
        """Create an empty cache.

        Args:
            params_tag: Identifier for the RNAplfold parameters.
            na_policy: The ``NA`` fill policy already applied to the values.
            n_channels: Number of accessibility channels per position.
        """
        self.params_tag = params_tag
        self.na_policy = na_policy
        self.n_channels = n_channels
        self.values: dict[str, np.ndarray] = {}
        self.na_masks: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.values)

    def __contains__(self, sequence: str) -> bool:
        return sequence_key(sequence) in self.values

    def add(
        self, sequence: str, block: np.ndarray, na_mask: np.ndarray
    ) -> None:
        """Insert one accessibility profile.

        Args:
            sequence: The folded nucleotide sequence.
            block: Float array shaped ``(n_channels, len(sequence))``.
            na_mask: Boolean array of the same shape.

        Raises:
            ValueError: If either array has the wrong shape.
        """
        length = len(to_rna(sequence))
        expected = (self.n_channels, length)
        if block.shape != expected:
            raise ValueError(
                f"accessibility block has shape {block.shape}, expected "
                f"{expected}"
            )
        if na_mask.shape != expected:
            raise ValueError(
                f"NA mask has shape {na_mask.shape}, expected {expected}"
            )
        key = sequence_key(sequence)
        self.values[key] = np.ascontiguousarray(block, dtype=np.float32)
        self.na_masks[key] = np.ascontiguousarray(na_mask, dtype=bool)

    def get(self, sequence: str) -> np.ndarray:
        """Return the cached accessibility block for a sequence.

        Args:
            sequence: Nucleotide sequence.

        Returns:
            Float32 array shaped ``(n_channels, length)``.

        Raises:
            KeyError: If the sequence is not cached.
        """
        key = sequence_key(sequence)
        try:
            return self.values[key]
        except KeyError:
            raise KeyError(
                f"no cached RNAplfold accessibility for sequence starting "
                f"{to_rna(sequence)[:24]!r}. Run "
                f"scripts/preprocess_structures.py --tool rnaplfold first."
            ) from None

    def get_na_mask(self, sequence: str) -> np.ndarray:
        """Return the cached ``NA`` mask for a sequence.

        Args:
            sequence: Nucleotide sequence.

        Returns:
            Boolean array shaped ``(n_channels, length)``.

        Raises:
            KeyError: If the sequence is not cached.
        """
        key = sequence_key(sequence)
        if key not in self.na_masks:
            raise KeyError("no cached NA mask for this sequence")
        return self.na_masks[key]

    def missing(self, sequences: Sequence[str]) -> list[str]:
        """Return the RNA-normalised unique sequences absent from the cache."""
        return [s for s in unique_sequences(sequences) if s not in self]

    def save(self, path: str | Path) -> Path:
        """Write the cache to a compressed ``.npz`` file.

        All blocks are stacked into one array, which requires a single common
        sequence length. Ragged corpora are rejected rather than silently
        padded.

        Args:
            path: Destination path; parent directories are created.

        Returns:
            The resolved destination path.

        Raises:
            ValueError: If cached sequences have differing lengths.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.values)
        meta = json.dumps(
            {
                "params_tag": self.params_tag,
                "na_policy": self.na_policy,
                "n_channels": self.n_channels,
            }
        )
        if not keys:
            np.savez_compressed(
                out,
                kind=np.array(self.KIND),
                meta=np.array(meta),
                keys=np.zeros(0, dtype="U64"),
                values=np.zeros((0, self.n_channels, 0), dtype=np.float32),
                na_masks=np.zeros((0, self.n_channels, 0), dtype=bool),
            )
            return out

        lengths = {self.values[k].shape[1] for k in keys}
        if len(lengths) != 1:
            raise ValueError(
                f"cannot save a cache with mixed sequence lengths {sorted(lengths)}; "
                f"write one cache file per length"
            )
        np.savez_compressed(
            out,
            kind=np.array(self.KIND),
            meta=np.array(meta),
            keys=np.array(keys, dtype="U64"),
            values=np.stack([self.values[k] for k in keys]),
            na_masks=np.stack([self.na_masks[k] for k in keys]),
        )
        return out

    @classmethod
    def load(
        cls, paths: str | Path | Sequence[str | Path]
    ) -> "AccessibilityCache":
        """Load and merge one or more cache files or shards.

        Args:
            paths: A single path, or several paths/shards to merge.

        Returns:
            The merged cache.

        Raises:
            FileNotFoundError: If a path does not exist.
            ValueError: If a file is not an RNAplfold cache, or shards
                disagree on parameters, ``NA`` policy or channel count.
        """
        items = [paths] if isinstance(paths, (str, Path)) else list(paths)
        cache: AccessibilityCache | None = None
        for item in items:
            p = Path(item)
            if not p.is_file():
                raise FileNotFoundError(f"cache file not found: {p}")
            with np.load(p, allow_pickle=False) as z:
                kind = str(z["kind"])
                if kind != cls.KIND:
                    raise ValueError(f"{p} is a {kind!r} cache, not {cls.KIND!r}")
                meta = json.loads(str(z["meta"]))
                if cache is None:
                    cache = cls(
                        params_tag=meta["params_tag"],
                        na_policy=meta["na_policy"],
                        n_channels=int(meta["n_channels"]),
                    )
                else:
                    for field in ("params_tag", "na_policy", "n_channels"):
                        mine = getattr(
                            cache,
                            field,
                        )
                        theirs = (
                            int(meta[field])
                            if field == "n_channels"
                            else meta[field]
                        )
                        if mine != theirs:
                            raise ValueError(
                                f"shard {p} has {field}={theirs!r} but earlier "
                                f"shards have {mine!r}; refusing to merge "
                                f"inconsistent caches"
                            )
                for key, block, mask in zip(
                    z["keys"], z["values"], z["na_masks"]
                ):
                    cache.values[str(key)] = np.ascontiguousarray(
                        block, dtype=np.float32
                    )
                    cache.na_masks[str(key)] = np.ascontiguousarray(
                        mask, dtype=bool
                    )
        return cache if cache is not None else cls()


def shard_paths(base: str | Path) -> list[Path]:
    """Return the shard files belonging to a cache base path, plus the base.

    Args:
        base: Cache path such as ``cache/accessibility.npz``.

    Returns:
        Sorted list of existing files: ``base`` itself if present, followed by
        every ``base``-derived ``*.shardNNNNN.npz`` file.
    """
    base_path = Path(base)
    stem = base_path.with_suffix("")
    found = sorted(stem.parent.glob(f"{stem.name}.shard*.npz"))
    return ([base_path] if base_path.is_file() else []) + found


def shard_path_for(base: str | Path, shard: int) -> Path:
    """Return the path for one shard of a cache.

    Args:
        base: Cache path such as ``cache/accessibility.npz``.
        shard: Zero-based shard index.

    Returns:
        Path of the form ``<base-without-suffix>.shardNNNNN.npz``.
    """
    stem = Path(base).with_suffix("")
    return stem.parent / f"{stem.name}.shard{shard:05d}.npz"
