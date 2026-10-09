"""RNAplfold output parsing, NA handling, and real ViennaRNA integration.

The parsing tests use synthetic ``_lunp`` fixtures and run everywhere. Tests
marked ``vienna`` invoke the real binaries and skip when they are absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import synthetic_lunp_text

from structsplice.config import (
    N_ACCESSIBILITY_CHANNELS,
    RNAfoldParams,
    RNAplfoldParams,
)
from structsplice.encoding import one_hot_dot_bracket
from structsplice.vienna import (
    ViennaRNAError,
    find_binary,
    parse_lunp,
    run_rnafold,
    run_rnaplfold,
    to_rna,
    validate_sequence,
)


def test_to_rna_converts_thymine():
    assert to_rna("ACGT") == "ACGU"
    assert to_rna("acgt") == "ACGU"


def test_validate_sequence_rejects_bad_alphabet():
    with pytest.raises(ValueError):
        validate_sequence("ACGUX")


def test_validate_sequence_rejects_empty():
    with pytest.raises(ValueError):
        validate_sequence("")


# ── _lunp parsing ──────────────────────────────────────────────────────────


def test_parse_lunp_shape():
    text = synthetic_lunp_text(seq_length=12, ulength=8)
    values, na = parse_lunp(text, seq_length=12, ulength=8)
    assert values.shape == (8, 12)
    assert na.shape == (8, 12)


def test_parse_lunp_na_appears_exactly_where_stretch_exceeds_position():
    """Row l, position i (1-based) is NA iff l > i."""
    text = synthetic_lunp_text(seq_length=12, ulength=8)
    _, na = parse_lunp(text, seq_length=12, ulength=8)
    for l in range(1, 9):
        for i in range(1, 13):
            assert na[l - 1, i - 1] == (l > i), f"l={l} i={i}"


def test_parse_lunp_na_count_is_the_triangle():
    """With ulength <= seq_length the NA block is a triangle of size u(u-1)/2."""
    u, n = 8, 20
    text = synthetic_lunp_text(seq_length=n, ulength=u)
    _, na = parse_lunp(text, seq_length=n, ulength=u)
    assert int(na.sum()) == u * (u - 1) // 2


def test_parse_lunp_na_positions_hold_zero_in_values():
    text = synthetic_lunp_text(seq_length=12, ulength=8)
    values, na = parse_lunp(text, seq_length=12, ulength=8)
    assert np.all(values[na] == 0.0)


def test_parse_lunp_na_is_distinguishable_from_measured_zero():
    """A real 0.0 and an NA must not be confusable: the mask separates them."""
    lines = ["#unpaired probabilities", " #$i\tl=1\t2"]
    lines.append("1\t0.000000\tNA")
    lines.append("2\t0.000000\t0.000000")
    values, na = parse_lunp("\n".join(lines) + "\n", seq_length=2, ulength=2)
    # Both entries are 0.0 in values ...
    assert values[1, 0] == 0.0 and values[1, 1] == 0.0
    # ... but only the first is flagged NA.
    assert na[1, 0] and not na[1, 1]


def test_parse_lunp_values_are_position_aligned():
    """Our synthetic generator writes scale*(i+l); check a few cells land right."""
    text = synthetic_lunp_text(seq_length=10, ulength=4, scale=0.01)
    values, na = parse_lunp(text, seq_length=10, ulength=4)
    for l in range(1, 5):
        for i in range(l, 11):
            assert values[l - 1, i - 1] == pytest.approx(
                min(0.999, 0.01 * (i + l)), abs=1e-6
            )


def test_parse_lunp_rejects_wrong_row_count():
    text = synthetic_lunp_text(seq_length=12, ulength=8)
    with pytest.raises(ViennaRNAError, match="position rows"):
        parse_lunp(text, seq_length=11, ulength=8)


def test_parse_lunp_rejects_wrong_column_count():
    text = synthetic_lunp_text(seq_length=12, ulength=8)
    with pytest.raises(ViennaRNAError, match="column"):
        parse_lunp(text, seq_length=12, ulength=7)


def test_parse_lunp_rejects_misordered_positions():
    lines = ["#unpaired probabilities", " #$i\tl=1"]
    lines.append("2\t0.5")
    lines.append("1\t0.5")
    with pytest.raises(ViennaRNAError, match="position"):
        parse_lunp("\n".join(lines) + "\n", seq_length=2, ulength=1)


def test_parse_lunp_rejects_non_numeric_field():
    lines = ["#unpaired probabilities", " #$i\tl=1"]
    lines.append("1\tnonsense")
    with pytest.raises(ViennaRNAError):
        parse_lunp("\n".join(lines) + "\n", seq_length=1, ulength=1)


def test_parse_lunp_rejects_impossible_na_placement():
    """An NA where the stretch is defined means the parse is misaligned."""
    lines = ["#unpaired probabilities", " #$i\tl=1\t2"]
    lines.append("1\tNA\tNA")  # l=1 at i=1 is defined, so this must fail
    lines.append("2\t0.1\t0.2")
    with pytest.raises(ViennaRNAError):
        parse_lunp("\n".join(lines) + "\n", seq_length=2, ulength=2)


# ── Real ViennaRNA ─────────────────────────────────────────────────────────


@pytest.mark.vienna
def test_rnafold_returns_valid_dot_bracket(requires_vienna):
    seqs = ["GGGGAAAACCCC", "ACGUACGUACGUACGUACGU"]
    structures, mfes = run_rnafold(seqs, RNAfoldParams())
    assert len(structures) == 2 and len(mfes) == 2
    for seq, structure in zip(seqs, structures):
        assert len(structure) == len(seq)
        assert set(structure) <= set(".()")
        one_hot_dot_bracket(structure)  # must be balanced and encodable


@pytest.mark.vienna
def test_rnafold_folds_a_known_hairpin(requires_vienna):
    """A GC-rich stem with an AAAA loop must pair, not come back all dots."""
    structures, mfes = run_rnafold(["GGGGGAAAACCCCC"], RNAfoldParams())
    assert "(" in structures[0] and ")" in structures[0]
    assert mfes[0] < 0.0


@pytest.mark.vienna
def test_rnafold_reproduces_the_shipped_pnas_structures(
    requires_vienna, real_train_npz
):
    """Our RNAfold call must match the structure field inside the npz files."""
    with np.load(real_train_npz, allow_pickle=True) as z:
        sequences = [str(s) for s in z["model_sequence"][:40]]
        expected = [str(s) for s in z["structure"][:40]]
    structures, _ = run_rnafold(sequences, RNAfoldParams())
    mismatches = [
        i for i, (a, b) in enumerate(zip(structures, expected)) if a != b
    ]
    assert not mismatches, f"rows disagreeing with the npz: {mismatches[:5]}"


@pytest.mark.vienna
def test_rnaplfold_shape_and_channel_count(requires_vienna):
    seqs = ["GGGGAAAACCCCGGGGAAAACCCC"]
    blocks, masks = run_rnaplfold(
        seqs, RNAplfoldParams(), na_policy="zero",
        n_channels=N_ACCESSIBILITY_CHANNELS,
    )
    assert blocks.shape == (1, N_ACCESSIBILITY_CHANNELS, len(seqs[0]))
    assert masks.shape == blocks.shape


@pytest.mark.vienna
def test_rnaplfold_values_are_probabilities(requires_vienna):
    blocks, _ = run_rnaplfold(
        ["GGGGAAAACCCCGGGGAAAACCCC"], RNAplfoldParams(),
        n_channels=N_ACCESSIBILITY_CHANNELS,
    )
    assert blocks.min() >= 0.0 and blocks.max() <= 1.0


@pytest.mark.vienna
def test_rnaplfold_na_mask_matches_the_boundary_rule(requires_vienna):
    """Real RNAplfold must mark NA exactly where l > i, as the parser assumes."""
    seq = "GGGGAAAACCCCGGGGAAAACCCC"
    _, masks = run_rnaplfold(
        [seq], RNAplfoldParams(), n_channels=N_ACCESSIBILITY_CHANNELS
    )
    mask = masks[0]
    for l in range(1, N_ACCESSIBILITY_CHANNELS + 1):
        for i in range(1, len(seq) + 1):
            assert mask[l - 1, i - 1] == (l > i), f"l={l} i={i}"


@pytest.mark.vienna
def test_rnaplfold_single_nucleotide_channel_differs_from_longer_stretches(
    requires_vienna,
):
    """Channel l=1 is not the whole feature: l=8 must carry different values.

    This is the l=1 versus eight-channel confusion the brief warns about.
    """
    blocks, _ = run_rnaplfold(
        ["GGGGAAAACCCCGGGGAAAACCCC"], RNAplfoldParams(),
        n_channels=N_ACCESSIBILITY_CHANNELS,
    )
    block = blocks[0]
    assert not np.allclose(block[0], block[7])


@pytest.mark.vienna
def test_rnaplfold_accessibility_decreases_with_stretch_length(requires_vienna):
    """P(unpaired stretch of length l) is non-increasing in l at a position."""
    blocks, masks = run_rnaplfold(
        ["GGGGAAAACCCCGGGGAAAACCCC"], RNAplfoldParams(),
        n_channels=N_ACCESSIBILITY_CHANNELS,
    )
    block, mask = blocks[0], masks[0]
    for i in range(N_ACCESSIBILITY_CHANNELS, block.shape[1]):
        defined = block[~mask[:, i], i]
        assert np.all(np.diff(defined) <= 1e-6), f"not monotone at position {i}"


@pytest.mark.vienna
def test_rnaplfold_edge_policy_injects_no_zeros(requires_vienna):
    """The 'edge' policy carries the last defined value instead of zeroing."""
    seq = "GGGGAAAACCCCGGGGAAAACCCC"
    blocks, masks = run_rnaplfold(
        [seq], RNAplfoldParams(), na_policy="edge",
        n_channels=N_ACCESSIBILITY_CHANNELS,
    )
    filled = blocks[0][masks[0]]
    assert filled.size > 0
    assert np.all(filled > 0.0)


def test_find_binary_returns_none_for_nonsense():
    assert find_binary("definitely-not-a-real-binary-xyz") is None
