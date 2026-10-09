"""Dataset loading, PSI validation and split integrity."""

from __future__ import annotations

import numpy as np
import pytest

from structsplice.config import LEFT_FLANK, MODEL_SEQUENCE_LENGTH, RIGHT_FLANK
from structsplice.data import (
    DatasetValidationError,
    SplicingDataset,
    grouped_train_val_split,
    load_npz,
    overlapping_groups,
    validate_dataset,
)


def test_synthetic_dataset_validates(synthetic_dataset):
    validate_dataset(synthetic_dataset)
    assert len(synthetic_dataset) == 24


def test_psi_in_unit_interval(synthetic_dataset):
    assert synthetic_dataset.psi.min() >= 0.0
    assert synthetic_dataset.psi.max() <= 1.0


def test_psi_out_of_range_is_rejected(synthetic_dataset):
    bad = SplicingDataset(
        exon=synthetic_dataset.exon,
        model_sequence=synthetic_dataset.model_sequence,
        psi=synthetic_dataset.psi + 0.5,
        name="bad",
    )
    with pytest.raises(DatasetValidationError, match="PSI"):
        validate_dataset(bad)


def test_nan_psi_is_rejected(synthetic_dataset):
    psi = synthetic_dataset.psi.copy()
    psi[3] = np.nan
    bad = SplicingDataset(
        exon=synthetic_dataset.exon,
        model_sequence=synthetic_dataset.model_sequence,
        psi=psi,
        name="bad",
    )
    with pytest.raises(DatasetValidationError):
        validate_dataset(bad)


def test_length_mismatch_is_rejected(synthetic_dataset):
    bad = SplicingDataset(
        exon=synthetic_dataset.exon[:5],
        model_sequence=synthetic_dataset.model_sequence,
        psi=synthetic_dataset.psi,
        name="bad",
    )
    with pytest.raises(DatasetValidationError):
        validate_dataset(bad)


def test_wrong_sequence_length_is_rejected(synthetic_dataset):
    truncated = np.array([s[:-1] for s in synthetic_dataset.model_sequence])
    bad = SplicingDataset(
        exon=synthetic_dataset.exon,
        model_sequence=truncated,
        psi=synthetic_dataset.psi,
        name="bad",
    )
    with pytest.raises(DatasetValidationError):
        validate_dataset(bad, expect_length=MODEL_SEQUENCE_LENGTH)


def test_subset_keeps_fields_aligned(synthetic_dataset):
    idx = [0, 5, 9]
    sub = synthetic_dataset.subset(idx)
    assert len(sub) == 3
    for i, row in enumerate(idx):
        assert sub.exon[i] == synthetic_dataset.exon[row]
        assert sub.psi[i] == synthetic_dataset.psi[row]
        assert sub.structure[i] == synthetic_dataset.structure[row]
        assert sub.n_inclusion[i] == synthetic_dataset.n_inclusion[row]


def test_read_depth_is_the_psi_denominator(synthetic_dataset):
    depth = synthetic_dataset.read_depth
    expected = synthetic_dataset.n_inclusion + synthetic_dataset.n_skipping
    assert np.allclose(depth, expected)


# ── Split integrity ────────────────────────────────────────────────────────


def test_split_is_a_partition(synthetic_dataset):
    split = grouped_train_val_split(synthetic_dataset, val_fraction=0.25, seed=1)
    both = np.concatenate([split.train, split.val])
    assert sorted(both.tolist()) == list(range(len(synthetic_dataset)))
    assert set(split.train.tolist()).isdisjoint(split.val.tolist())


def test_split_is_reproducible(synthetic_dataset):
    a = grouped_train_val_split(synthetic_dataset, val_fraction=0.25, seed=7)
    b = grouped_train_val_split(synthetic_dataset, val_fraction=0.25, seed=7)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.val, b.val)


def test_different_seeds_give_different_splits(synthetic_dataset):
    a = grouped_train_val_split(synthetic_dataset, val_fraction=0.25, seed=1)
    b = grouped_train_val_split(synthetic_dataset, val_fraction=0.25, seed=2)
    assert not np.array_equal(a.val, b.val)


def test_grouping_keeps_replicates_on_one_side(synthetic_dataset):
    """The synthetic dataset repeats one exon; it must not straddle the split."""
    split = grouped_train_val_split(synthetic_dataset, val_fraction=0.25, seed=3)
    keys = np.asarray(synthetic_dataset.exon).astype(str)
    assert set(keys[split.train]).isdisjoint(keys[split.val])
    assert split.n_train_groups + split.n_val_groups == split.n_groups


def test_group_counts_match_reality(synthetic_dataset):
    split = grouped_train_val_split(synthetic_dataset, val_fraction=0.25, seed=3)
    keys = np.asarray(synthetic_dataset.exon).astype(str)
    assert len(set(keys[split.train])) == split.n_train_groups
    assert len(set(keys[split.val])) == split.n_val_groups


def test_bad_val_fraction_is_rejected(synthetic_dataset):
    for value in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            grouped_train_val_split(synthetic_dataset, val_fraction=value)


def test_unknown_group_by_is_rejected(synthetic_dataset):
    with pytest.raises(ValueError, match="group_by"):
        grouped_train_val_split(synthetic_dataset, group_by="nonsense")


def test_overlapping_groups_detects_shared_exons(synthetic_dataset):
    left = synthetic_dataset.subset(range(10))
    right = synthetic_dataset.subset(range(8, 20))
    shared = overlapping_groups(left, right)
    assert len(shared) == 2


# ── Real PNAS files ────────────────────────────────────────────────────────


@pytest.mark.real_data
def test_real_train_split_loads(real_train_npz):
    dataset = load_npz(real_train_npz, name="train")
    assert len(dataset) == 191846
    assert dataset.psi.min() >= 0.0 and dataset.psi.max() <= 1.0
    assert {len(s) for s in dataset.model_sequence} == {MODEL_SEQUENCE_LENGTH}


@pytest.mark.real_data
def test_real_test_split_loads(real_test_npz):
    dataset = load_npz(real_test_npz, name="test")
    assert len(dataset) == 47962
    assert dataset.psi.min() >= 0.0 and dataset.psi.max() <= 1.0


@pytest.mark.real_data
def test_real_psi_equals_inclusion_over_inclusion_plus_skipping(real_train_npz):
    dataset = load_npz(real_train_npz, name="train")
    depth = dataset.n_inclusion + dataset.n_skipping
    assert (depth > 0).all()
    assert np.allclose(dataset.psi, dataset.n_inclusion / depth)


@pytest.mark.real_data
def test_real_model_sequence_is_flank_exon_flank(real_train_npz):
    dataset = load_npz(real_train_npz, name="train")
    for i in range(0, len(dataset), 20000):
        assert (
            dataset.model_sequence[i]
            == LEFT_FLANK + dataset.exon[i] + RIGHT_FLANK
        )


@pytest.mark.real_data
def test_real_split_sizes_are_unmodified(real_train_npz, real_test_npz):
    """Guard against anyone re-splitting the shipped PNAS files."""
    train = load_npz(real_train_npz, name="train")
    test = load_npz(real_test_npz, name="test")
    assert len(train) == 191846
    assert len(test) == 47962
    assert len(train) + len(test) == 239808


@pytest.mark.real_data
def test_real_validation_split_never_touches_test(real_train_npz, real_test_npz):
    """The validation rows come from the training file only."""
    train = load_npz(real_train_npz, name="train")
    test = load_npz(real_test_npz, name="test")
    split = grouped_train_val_split(train, val_fraction=0.1, seed=42)
    val_sequences = set(train.model_sequence[split.val].tolist())
    test_sequences = set(test.model_sequence.tolist())
    # Any intersection can only come from the 13 sequences the published
    # splits already share, never from this code splitting the test file.
    published_overlap = set(train.model_sequence.tolist()) & test_sequences
    assert val_sequences & test_sequences <= published_overlap
