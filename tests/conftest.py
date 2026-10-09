"""Shared fixtures.

Unit tests run without ViennaRNA by using synthetic fixtures. Tests that
genuinely need the binaries are marked ``vienna`` and skip when they are
absent, so a missing install is reported rather than silently passed over.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from structsplice.cache import AccessibilityCache, FoldCache
from structsplice.config import (
    LEFT_FLANK,
    MODEL_SEQUENCE_LENGTH,
    N_ACCESSIBILITY_CHANNELS,
    RIGHT_FLANK,
)
from structsplice.data import SplicingDataset
from structsplice.vienna import vienna_available

#: Repository root, so tests can find the real datasets when present.
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_NPZ = REPO_ROOT / "data" / "train_data.npz"
TEST_NPZ = REPO_ROOT / "data" / "test_data.npz"


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "vienna: requires the ViennaRNA command-line tools"
    )
    config.addinivalue_line(
        "markers", "real_data: requires the PNAS npz files in data/"
    )


@pytest.fixture(scope="session")
def has_vienna() -> bool:
    """Whether the ViennaRNA binaries are on PATH."""
    return vienna_available()


@pytest.fixture
def requires_vienna(has_vienna: bool) -> None:
    """Skip the test when ViennaRNA is unavailable."""
    if not has_vienna:
        pytest.skip(
            "ViennaRNA (RNAfold/RNAplfold) not on PATH; real folding test "
            "skipped"
        )


@pytest.fixture(scope="session")
def real_train_npz() -> Path:
    """Path to the shipped PNAS training file, skipping if absent."""
    if not TRAIN_NPZ.is_file():
        pytest.skip(f"{TRAIN_NPZ} not present")
    return TRAIN_NPZ


@pytest.fixture(scope="session")
def real_test_npz() -> Path:
    """Path to the shipped PNAS test file, skipping if absent."""
    if not TEST_NPZ.is_file():
        pytest.skip(f"{TEST_NPZ} not present")
    return TEST_NPZ


def make_exon(seed: int, length: int = 70) -> str:
    """Build a deterministic pseudo-random exon of the given length."""
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), size=length))


def make_structure(length: int = MODEL_SEQUENCE_LENGTH) -> str:
    """Build a balanced dot-bracket string of the given length."""
    arm = length // 4
    middle = length - 2 * arm
    return "(" * arm + "." * middle + ")" * arm


@pytest.fixture
def synthetic_dataset() -> SplicingDataset:
    """A 24-row synthetic dataset shaped exactly like the PNAS corpus.

    Includes two rows that share an exon, so grouped-split behaviour is
    exercised, and PSI values that hit both 0.0 and 1.0.
    """
    n = 24
    exons = [make_exon(i) for i in range(n - 1)]
    exons.append(exons[0])  # a deliberate replicate measurement
    model = np.array([LEFT_FLANK + e + RIGHT_FLANK for e in exons])
    # Derive PSI from the counts so the fixture satisfies the same
    # inclusion/(inclusion+skipping) identity the real corpus obeys.
    depth = 100
    inclusion = np.round(np.linspace(0.0, 1.0, n) * depth).astype(np.int64)
    skipping = (depth - inclusion).astype(np.int64)
    psi = inclusion / depth
    return SplicingDataset(
        exon=np.array(exons),
        model_sequence=model,
        psi=psi,
        structure=np.array([make_structure() for _ in range(n)]),
        n_inclusion=inclusion,
        n_skipping=skipping,
        name="synthetic",
    )


@pytest.fixture
def synthetic_fold_cache(synthetic_dataset: SplicingDataset) -> FoldCache:
    """RNAfold cache covering the synthetic dataset, built without ViennaRNA."""
    cache = FoldCache(params_tag="synthetic")
    for sequence, structure in zip(
        synthetic_dataset.model_sequence, synthetic_dataset.structure
    ):
        cache.add(str(sequence), str(structure), -10.0)
    return cache


@pytest.fixture
def synthetic_accessibility_cache(
    synthetic_dataset: SplicingDataset,
) -> AccessibilityCache:
    """RNAplfold cache covering the synthetic dataset, built without ViennaRNA.

    Values are deterministic and in ``[0, 1]``; the ``NA`` mask reproduces
    RNAplfold's real boundary pattern, ``True`` exactly where ``l > i``.
    """
    cache = AccessibilityCache(
        params_tag="synthetic", na_policy="zero",
        n_channels=N_ACCESSIBILITY_CHANNELS,
    )
    rng = np.random.default_rng(0)
    for sequence in synthetic_dataset.model_sequence:
        length = len(str(sequence))
        block = rng.random((N_ACCESSIBILITY_CHANNELS, length)).astype(np.float32)
        na = np.zeros_like(block, dtype=bool)
        for l_idx in range(N_ACCESSIBILITY_CHANNELS):
            na[l_idx, : l_idx] = True  # stretch of length l+1 undefined for i<l+1
        block[na] = 0.0
        cache.add(str(sequence), block, na)
    return cache


def synthetic_lunp_text(
    seq_length: int = 12, ulength: int = 8, scale: float = 0.01
) -> str:
    """Build a synthetic RNAplfold ``_lunp`` file.

    Reproduces the real layout: two header lines, one row per position, and
    ``NA`` wherever a stretch of length ``l`` would run off the 5' end, i.e.
    whenever ``l > i`` for 1-based position ``i``.

    Args:
        seq_length: Number of position rows.
        ulength: Number of length columns.
        scale: Multiplier making each generated value distinct.

    Returns:
        The file contents.
    """
    header = "#unpaired probabilities\n"
    header += " #$i\tl=" + "\t".join(str(l) for l in range(1, ulength + 1)) + "\n"
    lines = [header.rstrip("\n")]
    for i in range(1, seq_length + 1):
        fields = [str(i)]
        for l in range(1, ulength + 1):
            if l > i:
                fields.append("NA")
            else:
                fields.append(f"{min(0.999, scale * (i + l)):.6f}")
        lines.append("\t".join(fields))
    return "\n".join(lines) + "\n"
