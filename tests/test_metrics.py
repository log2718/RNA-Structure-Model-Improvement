"""Regression metrics, including constant-target and degenerate cases."""

from __future__ import annotations

import numpy as np
import pytest

from structsplice.metrics import (
    METRIC_KEYS,
    format_metrics,
    pearson_correlation,
    r_squared,
    regression_metrics,
)


def test_perfect_predictions():
    y = np.array([0.1, 0.4, 0.6, 0.9])
    m = regression_metrics(y.copy(), y)
    assert m["mse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["pearson_r"] == pytest.approx(1.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["n"] == 4


def test_mse_and_mae_values():
    pred = np.array([0.0, 1.0])
    true = np.array([1.0, 1.0])
    m = regression_metrics(pred, true)
    assert m["mse"] == pytest.approx(0.5)
    assert m["mae"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(np.sqrt(0.5))


def test_pearson_detects_perfect_anticorrelation():
    pred = np.array([1.0, 0.75, 0.5, 0.25])
    true = np.array([0.0, 0.25, 0.5, 0.75])
    r, note = pearson_correlation(pred, true)
    assert r == pytest.approx(-1.0)
    assert note is None


def test_pearson_r2_is_the_square_of_r():
    pred = np.array([0.1, 0.5, 0.2, 0.9])
    true = np.array([0.2, 0.4, 0.3, 0.8])
    m = regression_metrics(pred, true)
    assert m["pearson_r2"] == pytest.approx(m["pearson_r"] ** 2)


def test_r2_is_zero_for_the_mean_predictor():
    true = np.array([0.0, 0.25, 0.75, 1.0])
    pred = np.full_like(true, true.mean())
    r2, note = r_squared(pred, true)
    assert r2 == pytest.approx(0.0)
    assert note is None


def test_r2_is_negative_for_a_worse_than_mean_predictor():
    true = np.array([0.0, 1.0])
    pred = np.array([1.0, 0.0])
    r2, _ = r_squared(pred, true)
    assert r2 < 0


def test_constant_targets_do_not_crash_and_are_explained():
    """Zero target variance makes Pearson and R2 undefined, never silently 0."""
    true = np.full(5, 0.5)
    pred = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
    m = regression_metrics(pred, true)
    # Error metrics stay well defined.
    assert np.isfinite(m["mse"]) and np.isfinite(m["mae"])
    # Correlation-style metrics are nan, with a stated reason.
    assert np.isnan(m["pearson_r"])
    assert np.isnan(m["r2"])
    assert np.isnan(m["pearson_r2"])
    assert isinstance(m["pearson_note"], str)
    assert "constant" in m["pearson_note"].lower()
    assert isinstance(m["r2_note"], str)


def test_constant_predictions_make_pearson_undefined():
    true = np.array([0.1, 0.4, 0.7])
    pred = np.full(3, 0.5)
    r, note = pearson_correlation(pred, true)
    assert np.isnan(r)
    assert note is not None and "predictions are constant" in note
    # R2 stays well defined: the target still has variance to explain.
    r2, r2_note = r_squared(pred, true)
    assert np.isfinite(r2) and r2_note is None


def test_single_row_is_handled():
    """One row gives error metrics but no correlation."""
    m = regression_metrics(np.array([0.4]), np.array([0.6]))
    assert m["n"] == 1
    assert m["mse"] == pytest.approx(0.04)
    assert np.isnan(m["pearson_r"])
    assert "fewer than 2" in m["pearson_note"]


def test_empty_input_is_rejected():
    """An empty evaluation set is a bug upstream, so it raises rather than
    reporting a meaningless zero."""
    with pytest.raises(ValueError, match="empty"):
        regression_metrics(np.zeros(0), np.zeros(0))


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        regression_metrics(np.zeros(3), np.zeros(4))


def test_nan_predictions_are_rejected():
    with pytest.raises(ValueError):
        regression_metrics(np.array([np.nan, 0.5]), np.array([0.5, 0.5]))


def test_all_metric_keys_present():
    m = regression_metrics(np.array([0.2, 0.8]), np.array([0.3, 0.7]))
    for key in METRIC_KEYS:
        assert key in m


def test_format_metrics_is_readable():
    m = regression_metrics(np.array([0.2, 0.8]), np.array([0.3, 0.7]))
    text = format_metrics(m)
    assert "mse" in text and "mae" in text


def test_format_metrics_handles_undefined_values():
    m = regression_metrics(np.array([0.2, 0.8]), np.full(2, 0.5))
    text = format_metrics(m)
    assert isinstance(text, str) and text
