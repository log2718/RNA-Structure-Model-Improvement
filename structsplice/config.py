"""Configuration objects and shared constants.

Every path is configurable; nothing here is tied to a particular machine.
Defaults are resolved relative to the repository root so the code works
unchanged on a laptop and on an HPC scratch filesystem.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

# ── Repository layout ────────────────────────────────────────────────────────

#: Repository root (the parent of the ``structsplice`` package directory).
REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(var: str, default: Path) -> Path:
    """Return ``$var`` as a path if set, else ``default``."""
    raw = os.environ.get(var)
    return Path(raw).expanduser() if raw else default


#: Default locations. Each can be overridden by an environment variable or a
#: command-line flag; the CLI flag always wins.
DEFAULT_DATA_DIR = _env_path("STRUCTSPLICE_DATA_DIR", REPO_ROOT / "data")
DEFAULT_CACHE_DIR = _env_path("STRUCTSPLICE_CACHE_DIR", REPO_ROOT / "cache")
DEFAULT_OUTPUT_DIR = _env_path("STRUCTSPLICE_OUTPUT_DIR", REPO_ROOT / "outputs")

DEFAULT_TRAIN_NPZ = DEFAULT_DATA_DIR / "train_data.npz"
DEFAULT_TEST_NPZ = DEFAULT_DATA_DIR / "test_data.npz"

# ── Dataset constants (verified against the shipped PNAS npz files) ──────────

#: Fixed vector flanks the PNAS model prepends/appends to each 70 nt exon.
#: Verified: every ``model_sequence`` in train and test is
#: ``LEFT_FLANK + exon + RIGHT_FLANK`` with these exact constants.
LEFT_FLANK = "CATCCAGGTT"
RIGHT_FLANK = "CAGGTCTGAC"

#: Length of the full model input (10 + 70 + 10).
MODEL_SEQUENCE_LENGTH = 90

#: Nucleotide channel order. The datasets store DNA (``T``); ``T`` and ``U``
#: are treated as the same symbol and both map to the fourth channel.
NUCLEOTIDE_ORDER = ("A", "C", "G", "U")

#: Dot-bracket channel order, matching the ``struct_oh`` convention in the
#: shipped PNAS npz files (verified channel-by-channel).
DOT_BRACKET_ORDER = (".", "(", ")")

#: Number of RNAplfold accessibility channels handed to the model: the
#: probability that an unpaired stretch of length l = 1..8 ends at a position.
N_ACCESSIBILITY_CHANNELS = 8

# ── Feature configurations ──────────────────────────────────────────────────

#: ``name -> (uses_dot_bracket, uses_accessibility)``
FEATURE_CONFIGS = {
    "A": (False, False),
    "B": (True, False),
    "C": (False, True),
}

#: Human-readable description of each configuration.
CONFIG_DESCRIPTIONS = {
    "A": "sequence only (4 channels)",
    "B": "sequence + RNAfold dot-bracket (7 channels)",
    "C": "sequence + RNAplfold accessibility l=1..8 (12 channels)",
}


def input_dim(config: str) -> int:
    """Return the per-position input dimension for a feature configuration.

    Args:
        config: One of ``"A"``, ``"B"``, ``"C"``.

    Returns:
        Number of input channels per nucleotide.

    Raises:
        ValueError: If ``config`` is not a known configuration name.
    """
    if config not in FEATURE_CONFIGS:
        raise ValueError(
            f"unknown feature config {config!r}; expected one of "
            f"{sorted(FEATURE_CONFIGS)}"
        )
    use_db, use_acc = FEATURE_CONFIGS[config]
    dim = len(NUCLEOTIDE_ORDER)
    if use_db:
        dim += len(DOT_BRACKET_ORDER)
    if use_acc:
        dim += N_ACCESSIBILITY_CHANNELS
    return dim


# ── ViennaRNA parameters ────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class RNAfoldParams:
    """Parameters for an ``RNAfold`` minimum-free-energy run.

    The defaults reproduce the ``structure``/``mfe`` fields shipped inside the
    PNAS npz files exactly (verified on 300 randomly sampled sequences).
    """

    temperature: float = 37.0
    binary: str = "RNAfold"

    def cache_tag(self) -> str:
        """Return a short string identifying these parameters in cache files."""
        return f"rnafold_T{self.temperature:g}"


@dataclasses.dataclass(frozen=True)
class RNAplfoldParams:
    """Parameters for an ``RNAplfold`` accessibility run.

    Defaults are the values specified in the MLCB abstract::

        RNAplfold -W 200 -L 150 -u 31 -T 37

    ``ulength`` is deliberately 31 (not 8) so the cached output matches the
    abstract; only the first :data:`N_ACCESSIBILITY_CHANNELS` columns
    (l = 1..8) are handed to the model.
    """

    winsize: int = 200
    span: int = 150
    ulength: int = 31
    temperature: float = 37.0
    binary: str = "RNAplfold"

    def cache_tag(self) -> str:
        """Return a short string identifying these parameters in cache files."""
        return (
            f"rnaplfold_W{self.winsize}_L{self.span}"
            f"_u{self.ulength}_T{self.temperature:g}"
        )


#: How to fill RNAplfold's boundary ``NA`` entries.
#:
#: ``"zero"``
#:     Substitute ``0.0``. For a position *i* (1-based) and stretch length
#:     *l > i* the stretch would run off the 5' end of the molecule, so the
#:     event is *impossible within the modelled sequence* rather than merely
#:     unobserved. A companion boolean mask is always stored next to the
#:     values so the substitution is auditable and never silently
#:     indistinguishable from a measured zero.
#: ``"edge"``
#:     Carry the longest length that is defined at that position, i.e. the
#:     value at ``l = i``. Keeps the channels monotone in *l* and injects no
#:     artificial zeros.
NA_POLICIES = ("zero", "edge")
DEFAULT_NA_POLICY = "zero"
