"""Regression metrics for PSI predictions.

Every metric is computed in float64 on NumPy arrays and every degenerate case
returns ``float("nan")`` with a recorded reason rather than raising or silently
returning a misleading number. Pearson correlation and :math:`R^2` are both
undefined when a variance is zero, which happens for a constant target, a
constant prediction, or a single-row evaluation set.
"""

from __future__ import annotations

import math

import numpy as np

#: Metric keys produced by :func:`regression_metrics`, in report order.
METRIC_KEYS = ("n", "mse", "rmse", "mae", "pearson_r", "pearson_r2", "r2")


def _as_pair(
    predictions: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce predictions and targets to aligned finite float64 arrays.

    Args:
        predictions: Predicted values.
        targets: Observed values.

    Returns:
        ``(predictions, targets)`` as 1-D float64 arrays.

    Raises:
        ValueError: If the arrays differ in length or are not 1-D, or if any
            value is non-finite.
    """
    pred = np.asarray(predictions, dtype=np.float64).ravel()
    true = np.asarray(targets, dtype=np.float64).ravel()
    if pred.shape != true.shape:
        raise ValueError(
            f"predictions have {pred.shape[0]} entries but targets have "
            f"{true.shape[0]}"
        )
    if not np.isfinite(pred).all():
        raise ValueError(
            f"predictions contain {int((~np.isfinite(pred)).sum())} non-finite "
            f"value(s)"
        )
    if not np.isfinite(true).all():
        raise ValueError(
            f"targets contain {int((~np.isfinite(true)).sum())} non-finite "
            f"value(s)"
        )
    return pred, true


def pearson_correlation(
    predictions: np.ndarray, targets: np.ndarray
) -> tuple[float, str | None]:
    """Return Pearson's r and a reason string when it is undefined.

    Args:
        predictions: Predicted values.
        targets: Observed values.

    Returns:
        ``(r, reason)``. ``reason`` is ``None`` on success; otherwise ``r`` is
        ``nan`` and ``reason`` names the degeneracy.
    """
    pred, true = _as_pair(predictions, targets)
    if pred.size < 2:
        return math.nan, "fewer than 2 samples"
    pred_sd = pred.std()
    true_sd = true.std()
    if true_sd == 0.0 and pred_sd == 0.0:
        return math.nan, "both targets and predictions are constant"
    if true_sd == 0.0:
        return math.nan, "targets are constant (zero variance)"
    if pred_sd == 0.0:
        return math.nan, "predictions are constant (zero variance)"
    r = float(
        np.corrcoef(pred, true)[0, 1]
    )
    if not math.isfinite(r):
        return math.nan, "correlation underflowed to a non-finite value"
    return r, None


def r_squared(
    predictions: np.ndarray, targets: np.ndarray
) -> tuple[float, str | None]:
    """Return the coefficient of determination and a reason when undefined.

    This is the standard :math:`1 - \\mathrm{SS_{res}}/\\mathrm{SS_{tot}}`
    against the target mean, so it is negative whenever the model does worse
    than predicting that mean. It is *not* the square of Pearson's r, which is
    reported separately as ``pearson_r2``; the two agree only for an optimally
    rescaled prediction.

    Args:
        predictions: Predicted values.
        targets: Observed values.

    Returns:
        ``(r2, reason)``. ``reason`` is ``None`` on success; otherwise ``r2``
        is ``nan``.
    """
    pred, true = _as_pair(predictions, targets)
    if pred.size < 2:
        return math.nan, "fewer than 2 samples"
    ss_tot = float(((true - true.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return math.nan, "targets are constant (zero total sum of squares)"
    ss_res = float(((true - pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot, None


def regression_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, float | int | str | None]:
    """Compute the full metric set for a set of PSI predictions.

    Args:
        predictions: Predicted PSI values.
        targets: Measured PSI values.

    Returns:
        Dict with keys :data:`METRIC_KEYS` plus ``pearson_note`` and
        ``r2_note``, which hold ``None`` or a reason string when the
        corresponding metric is ``nan``.

    Raises:
        ValueError: If the inputs are empty, misaligned or non-finite.
    """
    pred, true = _as_pair(predictions, targets)
    if pred.size == 0:
        raise ValueError("cannot compute metrics on an empty array")

    residual = pred - true
    mse = float(np.mean(residual**2))
    r, pearson_note = pearson_correlation(pred, true)
    r2, r2_note = r_squared(pred, true)

    return {
        "n": int(pred.size),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(residual))),
        "pearson_r": r,
        "pearson_r2": math.nan if math.isnan(r) else r * r,
        "r2": r2,
        "pearson_note": pearson_note,
        "r2_note": r2_note,
    }


def format_metrics(metrics: dict[str, float | int | str | None]) -> str:
    """Render a metric dict as one aligned human-readable line block.

    Args:
        metrics: Output of :func:`regression_metrics`.

    Returns:
        A multi-line string suitable for a log file.
    """
    lines = []
    for key in METRIC_KEYS:
        value = metrics.get(key)
        if isinstance(value, int):
            lines.append(f"  {key:<11s} {value}")
        elif value is None or (
            isinstance(value, float) and math.isnan(value)
        ):
            note = ""
            if key.startswith("pearson") and metrics.get("pearson_note"):
                note = f"  ({metrics['pearson_note']})"
            elif key == "r2" and metrics.get("r2_note"):
                note = f"  ({metrics['r2_note']})"
            lines.append(f"  {key:<11s} nan{note}")
        else:
            lines.append(f"  {key:<11s} {value:.6f}")
    return "\n".join(lines)


def format_metrics_line(metrics: dict[str, float | int | str | None]) -> str:
    """Render a metric dict as one compact line, for log messages.

    Undefined metrics print as ``nan`` without their reason string; call
    :func:`format_metrics` for the multi-line form that includes reasons.

    Args:
        metrics: Output of :func:`regression_metrics`.

    Returns:
        A single-line string such as ``n=40 mse=0.0712 mae=0.2210 r=0.133``.
    """
    parts: list[str] = []
    for key in METRIC_KEYS:
        value = metrics.get(key)
        if isinstance(value, int):
            parts.append(f"{key}={value}")
        elif value is None or (
            isinstance(value, float) and math.isnan(value)
        ):
            parts.append(f"{key}=nan")
        else:
            parts.append(f"{key}={value:.4f}")
    return " ".join(parts)
