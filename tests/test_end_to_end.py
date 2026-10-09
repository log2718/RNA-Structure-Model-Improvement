"""End-to-end smoke tests: train a couple of batches, then evaluate.

These are deliberately tiny: two epochs over a 24-row synthetic dataset,
capped at two batches each. They verify that the whole pipeline runs and that
its outputs are well formed, not that the model is accurate.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from structsplice.config import FEATURE_CONFIGS, input_dim
from structsplice.evaluate import aggregate_runs, evaluate_checkpoint
from structsplice.train import TrainSettings, train


def tiny_settings(config: str, **overrides) -> TrainSettings:
    """Return settings small enough for a CPU smoke test."""
    kwargs = dict(
        config=config,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        val_fraction=0.25,
        seed=0,
        patience=0,            # never stop early in a 2-epoch run
        max_train_batches=2,   # two batches per epoch
        num_workers=0,
        device="cpu",
        dropout=0.1,
    )
    kwargs.update(overrides)
    return TrainSettings(**kwargs)


@pytest.mark.parametrize("config", sorted(FEATURE_CONFIGS))
def test_train_then_evaluate(
    config, tmp_path, synthetic_dataset, synthetic_fold_cache,
    synthetic_accessibility_cache,
):
    """The full loop must run and produce coherent artefacts for A, B and C."""
    out = tmp_path / f"train_{config}"
    summary = train(
        synthetic_dataset,
        tiny_settings(config),
        out,
        fold_cache=synthetic_fold_cache,
        accessibility_cache=synthetic_accessibility_cache,
        use_npz_structure=False,
    )

    # Training artefacts.
    assert (out / "best.pt").is_file()
    assert (out / "last.pt").is_file()
    assert (out / "history.jsonl").is_file()
    assert (out / "train_summary.json").is_file()
    assert summary["input_dim"] == input_dim(config)
    assert summary["epochs_run"] == 2
    assert len(summary["history"]) == 2
    for record in summary["history"]:
        assert np.isfinite(record["train_loss"])
        assert np.isfinite(record["val_loss"])
        assert record["train_batches"] == 2
    assert summary["best_epoch"] in (1, 2)

    # The JSONL log must be parseable line by line.
    lines = (out / "history.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)

    # train_summary.json must be strictly valid JSON (no bare NaN).
    text = (out / "train_summary.json").read_text()
    assert "NaN" not in text
    json.loads(text)

    # Evaluation on a held-out dataset, same procedure for every config.
    results = tmp_path / f"eval_{config}"
    payload = evaluate_checkpoint(
        out / "best.pt",
        synthetic_dataset,
        results,
        fold_cache=synthetic_fold_cache,
        accessibility_cache=synthetic_accessibility_cache,
        use_npz_structure=False,
        batch_size=8,
        device="cpu",
        split_name="synthetic_holdout",
    )
    assert payload["config"] == config
    assert payload["input_dim"] == input_dim(config)
    assert np.isfinite(payload["metrics"]["mse"])
    assert np.isfinite(payload["metrics"]["mae"])
    assert payload["metrics"]["n"] == len(synthetic_dataset)
    assert (results / "metrics.json").is_file()
    assert (results / "predictions.csv").is_file()


def test_predictions_csv_is_complete_and_bounded(
    tmp_path, synthetic_dataset, synthetic_fold_cache
):
    out = tmp_path / "train"
    train(
        synthetic_dataset, tiny_settings("B"), out,
        fold_cache=synthetic_fold_cache, use_npz_structure=False,
    )
    results = tmp_path / "eval"
    evaluate_checkpoint(
        out / "best.pt", synthetic_dataset, results,
        fold_cache=synthetic_fold_cache, use_npz_structure=False,
        device="cpu",
    )

    with (results / "predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(synthetic_dataset)
    assert set(rows[0]) == {
        "row_index", "exon_id", "model_sequence", "psi_true", "psi_pred",
    }
    for row in rows:
        idx = int(row["row_index"])
        # Identifiers must stay attached to the right row.
        assert row["exon_id"] == synthetic_dataset.exon[idx]
        assert row["model_sequence"] == synthetic_dataset.model_sequence[idx]
        assert float(row["psi_true"]) == pytest.approx(
            synthetic_dataset.psi[idx], abs=1e-6
        )
        # Predicted PSI must be a valid fraction.
        assert 0.0 <= float(row["psi_pred"]) <= 1.0

    # Every row appears exactly once.
    assert sorted(int(r["row_index"]) for r in rows) == list(
        range(len(synthetic_dataset))
    )


def test_same_seed_reproduces_the_run(tmp_path, synthetic_dataset):
    first = train(synthetic_dataset, tiny_settings("A", seed=3), tmp_path / "a")
    second = train(synthetic_dataset, tiny_settings("A", seed=3), tmp_path / "b")
    for x, y in zip(first["history"], second["history"]):
        assert x["train_loss"] == pytest.approx(y["train_loss"], abs=1e-9)
        assert x["val_loss"] == pytest.approx(y["val_loss"], abs=1e-9)


def test_different_seeds_give_different_runs(tmp_path, synthetic_dataset):
    first = train(synthetic_dataset, tiny_settings("A", seed=1), tmp_path / "a")
    second = train(synthetic_dataset, tiny_settings("A", seed=2), tmp_path / "b")
    assert (
        first["history"][0]["train_loss"] != second["history"][0]["train_loss"]
    )


def test_validation_rows_are_disjoint_from_training_rows(
    tmp_path, synthetic_dataset
):
    summary = train(synthetic_dataset, tiny_settings("A"), tmp_path / "a")
    assert summary["n_train_rows"] + summary["n_val_rows"] == len(
        synthetic_dataset
    )
    assert summary["n_train_groups"] + summary["n_val_groups"] == len(
        set(synthetic_dataset.exon.tolist())
    )


def test_early_stopping_can_trigger(tmp_path, synthetic_dataset):
    """With patience 1 and a frozen model the run must stop before the cap."""
    settings = tiny_settings("A", epochs=10, patience=1, learning_rate=1e-12)
    summary = train(synthetic_dataset, settings, tmp_path / "a")
    assert summary["epochs_run"] < 10
    assert summary["stopped_early"] is True


def test_kl_loss_runs_end_to_end(tmp_path, synthetic_dataset):
    summary = train(
        synthetic_dataset, tiny_settings("A", loss="kl"), tmp_path / "a"
    )
    for record in summary["history"]:
        assert np.isfinite(record["val_loss"])
        # Bernoulli KL is non-negative, unlike BCE which bottoms out at H(y).
        assert record["val_loss"] >= -1e-6


def test_depth_weighted_training_runs(tmp_path, synthetic_dataset):
    summary = train(
        synthetic_dataset,
        tiny_settings("A", weight_by_depth=True),
        tmp_path / "a",
    )
    assert np.isfinite(summary["history"][-1]["train_loss"])


def test_depth_weighting_without_counts_is_rejected(tmp_path, synthetic_dataset):
    stripped = synthetic_dataset.subset(range(len(synthetic_dataset)))
    stripped.n_inclusion = None
    stripped.n_skipping = None
    with pytest.raises(ValueError, match="weight_by_depth"):
        train(stripped, tiny_settings("A", weight_by_depth=True), tmp_path / "a")


def test_config_c_requires_a_cache(tmp_path, synthetic_dataset):
    with pytest.raises(ValueError, match="accessibility"):
        train(synthetic_dataset, tiny_settings("C"), tmp_path / "a")


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        TrainSettings(config="Z")
    with pytest.raises(ValueError):
        TrainSettings(config="A", epochs=0)
    with pytest.raises(ValueError):
        TrainSettings(config="A", learning_rate=0.0)
    with pytest.raises(ValueError):
        TrainSettings(config="A", val_fraction=1.5)


def test_aggregate_runs_reports_mean_and_spread():
    runs = [
        {"config": "A", "seed": 0, "metrics": {"mse": 0.10, "mae": 0.2, "n": 5}},
        {"config": "A", "seed": 1, "metrics": {"mse": 0.20, "mae": 0.3, "n": 5}},
    ]
    summary = aggregate_runs(runs)
    assert summary["n_runs"] == 2
    assert summary["configs"] == ["A"]
    assert summary["metrics"]["mse"]["mean"] == pytest.approx(0.15)
    assert summary["metrics"]["mse"]["std"] == pytest.approx(0.05)
    assert summary["metrics"]["mse"]["values"] == [0.10, 0.20]


def test_aggregate_runs_skips_undefined_metrics():
    runs = [
        {"config": "A", "seed": 0,
         "metrics": {"mse": 0.1, "pearson_r": None, "n": 5}},
    ]
    summary = aggregate_runs(runs)
    assert "mse" in summary["metrics"]
    assert "pearson_r" not in summary["metrics"]


def test_aggregate_runs_rejects_empty():
    with pytest.raises(ValueError):
        aggregate_runs([])
