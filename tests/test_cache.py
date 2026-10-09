"""Structure caches: keying, deduplication, mutated sequences, round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from structsplice.cache import (
    AccessibilityCache,
    FoldCache,
    sequence_key,
    shard_path_for,
    shard_paths,
    unique_sequences,
)
from structsplice.config import N_ACCESSIBILITY_CHANNELS


def test_key_is_sequence_content_not_identity():
    assert sequence_key("ACGU") == sequence_key("ACGU")
    assert sequence_key("ACGU") != sequence_key("ACGA")


def test_key_treats_dna_and_rna_as_one_molecule():
    assert sequence_key("ACGT") == sequence_key("ACGU")


def test_single_point_mutation_changes_the_key():
    """A mutated variant must never reuse the wild-type structure."""
    wild = "GGGGAAAACCCC"
    mutant = "GGGGAAAACCCA"
    assert sequence_key(wild) != sequence_key(mutant)


def test_unique_sequences_deduplicates_preserving_order():
    seqs = ["ACGU", "ACGA", "ACGT", "ACGU"]
    assert unique_sequences(seqs) == ["ACGU", "ACGA"]


def test_fold_cache_round_trip(tmp_path, synthetic_fold_cache):
    path = synthetic_fold_cache.save(tmp_path / "fold.npz")
    loaded = FoldCache.load(path)
    assert len(loaded) == len(synthetic_fold_cache)
    for key, structure in synthetic_fold_cache.structures.items():
        assert loaded.structures[key] == structure


def test_fold_cache_rejects_length_mismatch():
    cache = FoldCache()
    with pytest.raises(ValueError):
        cache.add("ACGUACGU", "((..))", -1.0)


def test_fold_cache_get_raises_with_actionable_message():
    cache = FoldCache()
    with pytest.raises(KeyError, match="preprocess_structures"):
        cache.get("ACGUACGU")


def test_fold_cache_missing_reports_only_absent_sequences(synthetic_fold_cache):
    known = list(synthetic_fold_cache.structures)
    assert known  # sanity
    seqs = ["GGGGAAAACCCC"]  # not in the cache
    assert synthetic_fold_cache.missing(seqs) == ["GGGGAAAACCCC"]


def test_accessibility_cache_round_trip(tmp_path, synthetic_accessibility_cache):
    path = synthetic_accessibility_cache.save(tmp_path / "acc.npz")
    loaded = AccessibilityCache.load(path)
    assert len(loaded) == len(synthetic_accessibility_cache)
    for key, block in synthetic_accessibility_cache.values.items():
        assert np.allclose(loaded.values[key], block)
        assert np.array_equal(
            loaded.na_masks[key], synthetic_accessibility_cache.na_masks[key]
        )


def test_accessibility_cache_preserves_na_mask_separately(
    tmp_path, synthetic_accessibility_cache
):
    """The NA mask must survive a save/load, not collapse into the values."""
    path = synthetic_accessibility_cache.save(tmp_path / "acc.npz")
    loaded = AccessibilityCache.load(path)
    key = next(iter(loaded.values))
    assert loaded.na_masks[key].any()
    assert np.all(loaded.values[key][loaded.na_masks[key]] == 0.0)


def test_accessibility_cache_metadata_survives(tmp_path, synthetic_accessibility_cache):
    path = synthetic_accessibility_cache.save(tmp_path / "acc.npz")
    loaded = AccessibilityCache.load(path)
    assert loaded.na_policy == synthetic_accessibility_cache.na_policy
    assert loaded.n_channels == N_ACCESSIBILITY_CHANNELS


def test_accessibility_cache_rejects_wrong_shape():
    cache = AccessibilityCache(n_channels=N_ACCESSIBILITY_CHANNELS)
    bad = np.zeros((3, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        cache.add("ACGUACGUAC", bad, np.zeros_like(bad, dtype=bool))


def test_cache_kind_mismatch_is_detected(tmp_path, synthetic_fold_cache):
    path = synthetic_fold_cache.save(tmp_path / "fold.npz")
    with pytest.raises(ValueError, match="cache"):
        AccessibilityCache.load(path)


def test_missing_cache_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FoldCache.load(tmp_path / "nope.npz")


def test_shard_paths_merge(tmp_path, synthetic_fold_cache):
    base = tmp_path / "fold.npz"
    keys = list(synthetic_fold_cache.structures)
    half = len(keys) // 2

    first = FoldCache(params_tag="synthetic")
    second = FoldCache(params_tag="synthetic")
    for i, key in enumerate(keys):
        target = first if i < half else second
        target.structures[key] = synthetic_fold_cache.structures[key]
        target.mfe[key] = synthetic_fold_cache.mfe[key]

    first.save(shard_path_for(base, 0))
    second.save(shard_path_for(base, 1))

    found = shard_paths(base)
    assert len(found) == 2
    merged = FoldCache.load(found)
    assert len(merged) == len(synthetic_fold_cache)


def test_duplicate_sequences_are_folded_once(synthetic_dataset):
    """The dataset repeats one exon, so unique count must be lower than rows."""
    seqs = synthetic_dataset.model_sequence.tolist()
    assert len(unique_sequences(seqs)) == len(seqs) - 1
