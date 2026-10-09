"""One-hot sequence encoding, dot-bracket encoding and feature assembly."""

from __future__ import annotations

import numpy as np
import pytest

from structsplice.config import (
    DOT_BRACKET_ORDER,
    N_ACCESSIBILITY_CHANNELS,
    NUCLEOTIDE_ORDER,
    input_dim,
)
from structsplice.encoding import (
    build_features,
    channel_first,
    encode_accessibility,
    one_hot_dot_bracket,
    one_hot_sequence,
)


def test_one_hot_sequence_shape_and_values():
    oh = one_hot_sequence("ACGU")
    assert oh.shape == (4, 4)
    assert np.array_equal(oh, np.eye(4, dtype=np.float32))


def test_one_hot_channel_order_is_acgu():
    assert NUCLEOTIDE_ORDER == ("A", "C", "G", "U")
    for i, base in enumerate(NUCLEOTIDE_ORDER):
        oh = one_hot_sequence(base)
        assert oh[0, i] == 1.0
        assert oh[0].sum() == 1.0


def test_thymine_and_uracil_are_the_same_channel():
    assert np.array_equal(one_hot_sequence("ACGT"), one_hot_sequence("ACGU"))


def test_one_hot_is_lowercase_tolerant():
    assert np.array_equal(one_hot_sequence("acgu"), one_hot_sequence("ACGU"))


def test_one_hot_rows_sum_to_one_for_known_bases():
    oh = one_hot_sequence("AACCGGUU")
    assert np.allclose(oh.sum(axis=1), 1.0)


def test_unknown_base_is_all_zero_not_a_fifth_channel():
    oh = one_hot_sequence("ANA", allow_unknown=True)
    assert oh.shape == (3, 4)
    assert oh[1].sum() == 0.0
    assert oh[0].sum() == 1.0 and oh[2].sum() == 1.0


def test_unknown_base_can_be_rejected():
    with pytest.raises(ValueError):
        one_hot_sequence("AXA", allow_unknown=False)


def test_empty_sequence_is_rejected():
    with pytest.raises(ValueError):
        one_hot_sequence("")


# ── Dot-bracket ────────────────────────────────────────────────────────────


def test_dot_bracket_shape_and_order():
    assert DOT_BRACKET_ORDER == (".", "(", ")")
    oh = one_hot_dot_bracket(".()")
    assert oh.shape == (3, 3)
    assert np.array_equal(oh, np.eye(3, dtype=np.float32))


def test_dot_bracket_rows_sum_to_one():
    oh = one_hot_dot_bracket("((((....))))")
    assert np.allclose(oh.sum(axis=1), 1.0)


def test_dot_bracket_distinguishes_open_from_close():
    """A lone bracket is unbalanced, so compare the two rows of a real pair."""
    oh = one_hot_dot_bracket("()")
    assert not np.array_equal(oh[0], oh[1])
    assert oh[0, DOT_BRACKET_ORDER.index("(")] == 1.0
    assert oh[1, DOT_BRACKET_ORDER.index(")")] == 1.0


def test_dot_bracket_rejects_unknown_symbol():
    with pytest.raises(ValueError):
        one_hot_dot_bracket("..[[..]]..")


def test_dot_bracket_rejects_unbalanced_structure():
    with pytest.raises(ValueError):
        one_hot_dot_bracket("((((..")


# ── Accessibility ──────────────────────────────────────────────────────────


def test_encode_accessibility_transposes_to_position_major():
    block = np.arange(N_ACCESSIBILITY_CHANNELS * 5, dtype=np.float32).reshape(
        N_ACCESSIBILITY_CHANNELS, 5
    ) / 100.0
    encoded = encode_accessibility(block)
    assert encoded.shape == (5, N_ACCESSIBILITY_CHANNELS)
    assert np.allclose(encoded, block.T)


def test_encode_accessibility_rejects_wrong_channel_count():
    with pytest.raises(ValueError):
        encode_accessibility(np.zeros((3, 10), dtype=np.float32))


def test_encode_accessibility_rejects_out_of_range_probabilities():
    bad = np.full((N_ACCESSIBILITY_CHANNELS, 4), 1.5, dtype=np.float32)
    with pytest.raises(ValueError):
        encode_accessibility(bad)


# ── Feature assembly: the three configurations ─────────────────────────────


def test_config_a_dimension_is_four():
    features = build_features("ACGUACGU")
    assert features.shape == (8, 4)
    assert features.shape[1] == input_dim("A")


def test_config_b_dimension_is_seven():
    features = build_features("ACGUACGU", structure="((....))")
    assert features.shape == (8, 7)
    assert features.shape[1] == input_dim("B")


def test_config_c_dimension_is_twelve():
    acc = np.zeros((N_ACCESSIBILITY_CHANNELS, 8), dtype=np.float32)
    features = build_features("ACGUACGU", accessibility=acc)
    assert features.shape == (8, 12)
    assert features.shape[1] == input_dim("C")


def test_config_b_first_four_channels_match_config_a():
    seq = "ACGUACGU"
    a = build_features(seq)
    b = build_features(seq, structure="((....))")
    assert np.allclose(b[:, :4], a)


def test_config_c_first_four_channels_match_config_a():
    seq = "ACGUACGU"
    acc = np.full((N_ACCESSIBILITY_CHANNELS, 8), 0.25, dtype=np.float32)
    a = build_features(seq)
    c = build_features(seq, accessibility=acc)
    assert np.allclose(c[:, :4], a)


def test_accessibility_is_position_aligned_with_sequence():
    """Channel 0 is set only at position 3; it must land on row 3."""
    acc = np.zeros((N_ACCESSIBILITY_CHANNELS, 8), dtype=np.float32)
    acc[0, 3] = 0.7
    features = build_features("ACGUACGU", accessibility=acc)
    assert features[3, 4] == pytest.approx(0.7)
    assert features[:, 4].sum() == pytest.approx(0.7)


def test_structure_length_must_match_sequence():
    with pytest.raises(ValueError):
        build_features("ACGUACGU", structure="((..))")


def test_accessibility_length_must_match_sequence():
    acc = np.zeros((N_ACCESSIBILITY_CHANNELS, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        build_features("ACGUACGU", accessibility=acc)


def test_features_are_float32():
    assert build_features("ACGU").dtype == np.float32


def test_channel_first_round_trips():
    features = build_features("ACGUACGU", structure="((....))")
    assert channel_first(features).shape == (7, 8)
    assert np.allclose(channel_first(features).T, features)


def test_config_b_matches_shipped_struct_oh_convention(real_train_npz):
    """Our dot-bracket channels must agree with the PNAS struct_oh field."""
    import numpy as np

    with np.load(real_train_npz, allow_pickle=True) as z:
        struct_oh = z["struct_oh"][:5]
        structures = z["structure"][:5]
    for i, structure in enumerate(structures):
        ours = one_hot_dot_bracket(str(structure))
        assert np.allclose(ours.T, struct_oh[i]), f"row {i} disagrees"
