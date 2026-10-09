r"""Objectives for fractional PSI targets.

Why binary cross-entropy on a fractional target
-----------------------------------------------

The original PNAS model trains on the KL divergence between two Bernoulli
distributions: one with parameter :math:`y` (the measured PSI) and one with
parameter :math:`\hat p = \sigma(z)` (the model's prediction). That divergence
is

.. math::

   \mathrm{KL}(y \,\|\, \hat p)
     = y \log \frac{y}{\hat p} + (1-y) \log \frac{1-y}{1-\hat p}
     = \underbrace{-\,[\,y \log \hat p + (1-y)\log(1-\hat p)\,]}_{H(y,\hat p)
       \;=\; \text{BCE}}
       \;-\; \underbrace{(-\,[\,y\log y + (1-y)\log(1-y)\,])}_{H(y)
       \;=\; \text{target entropy}} .

So **Bernoulli KL = BCE − H(y)**. The subtracted term depends only on the
target, not on any model parameter, so:

* the two objectives have *identical gradients* with respect to the weights,
  and therefore define the same optimisation problem and the same optimum;
* they differ by a per-example constant, so their *numeric values* differ.
  KL is non-negative and reaches 0 at a perfect fit, whereas BCE bottoms out at
  :math:`H(y) > 0` for any non-saturated target.

This matters in two places. Early stopping and checkpoint selection are
unaffected, because adding a fixed constant to every epoch's validation score
cannot change which epoch is best. Comparing a logged loss against a number
reported for the PNAS model *is* affected, so :func:`bernoulli_kl_with_logits`
is provided and selectable with ``--loss kl``; it reproduces the PNAS
``kl_divergence_from_logits`` convention exactly.

The default is ``bce``, as requested: it is the numerically safer of the two,
since ``BCEWithLogitsLoss`` fuses the sigmoid and the log into one stable
expression and never evaluates :math:`\log 0` for saturated predictions.

Read-depth weighting
--------------------

This corpus *does* carry the underlying counts. ``metadata_PSI`` equals
``num_exon_inclusion / (num_exon_inclusion + num_exon_skipping)`` exactly, and
the denominator ranges over roughly two orders of magnitude across the library.
A PSI of 0.5 from 6 reads and a PSI of 0.5 from 600 reads therefore carry very
different evidence, and unweighted BCE treats them as equal.

Scaling each example's BCE by its read depth turns the objective into the exact
negative log-likelihood of a Binomial observation (up to a binomial
coefficient that is constant in the parameters), which is the statistically
correct likelihood for this measurement process. It is available through
``--weight-by-depth`` and :func:`read_depth_weights`.

It is **off by default** on purpose. The published PNAS objective is
unweighted, so leaving it off keeps this project's baseline comparable to the
paper and keeps the A/B/C comparison about features rather than about a change
of objective. Turning it on changes the estimand, so it should be a deliberate,
separately reported experiment rather than a silent default.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

LOSS_NAMES = ("bce", "kl")


def _check_targets(targets: torch.Tensor) -> None:
    """Validate that targets are finite probabilities.

    Args:
        targets: Tensor of fractional PSI values.

    Raises:
        ValueError: If any entry is non-finite or outside ``[0, 1]``.
    """
    if not torch.isfinite(targets).all():
        raise ValueError("targets contain non-finite values")
    if float(targets.min()) < 0.0 or float(targets.max()) > 1.0:
        raise ValueError(
            f"targets must lie in [0, 1]; observed range "
            f"[{float(targets.min())}, {float(targets.max())}]"
        )


def _reduce(
    per_example: torch.Tensor,
    weights: torch.Tensor | None,
    reduction: str,
) -> torch.Tensor:
    """Reduce a per-example loss, optionally weighted.

    With weights, ``"mean"`` is the weighted mean ``sum(w * l) / sum(w)``, so
    the loss scale stays comparable to the unweighted case and the learning
    rate does not have to be retuned when weighting is switched on.

    Args:
        per_example: ``(batch,)`` tensor of per-example losses.
        weights: Optional ``(batch,)`` non-negative weights.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Returns:
        The reduced loss, or the per-example tensor when ``reduction`` is
        ``"none"``.

    Raises:
        ValueError: On an unknown reduction, a shape mismatch, negative
            weights, or weights summing to zero.
    """
    if weights is not None:
        if weights.shape != per_example.shape:
            raise ValueError(
                f"weights shape {tuple(weights.shape)} does not match loss "
                f"shape {tuple(per_example.shape)}"
            )
        if float(weights.min()) < 0.0:
            raise ValueError("loss weights must be non-negative")
        per_example = per_example * weights

    if reduction == "none":
        return per_example
    if reduction == "sum":
        return per_example.sum()
    if reduction != "mean":
        raise ValueError(
            f"unknown reduction {reduction!r}; expected 'mean', 'sum' or 'none'"
        )

    if weights is None:
        return per_example.mean()
    total = weights.sum()
    if float(total) <= 0.0:
        raise ValueError("weighted mean is undefined: weights sum to zero")
    return per_example.sum() / total


def bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary cross-entropy on fractional targets, computed from logits.

    Wraps :func:`torch.nn.functional.binary_cross_entropy_with_logits`, which
    accepts any target in ``[0, 1]``, not just 0 and 1, and fuses the sigmoid
    with the log for numerical stability.

    Args:
        logits: ``(batch,)`` raw model outputs, pre-sigmoid.
        targets: ``(batch,)`` measured PSI values in ``[0, 1]``.
        weights: Optional ``(batch,)`` non-negative per-example weights.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Returns:
        Scalar loss, or ``(batch,)`` when ``reduction`` is ``"none"``.

    Raises:
        ValueError: If shapes disagree or targets are invalid.
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} does not match targets shape "
            f"{tuple(targets.shape)}"
        )
    _check_targets(targets)
    per_example = F.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), reduction="none"
    )
    return _reduce(per_example, weights, reduction)


def bernoulli_entropy(targets: torch.Tensor) -> torch.Tensor:
    """Return the Bernoulli entropy ``H(y)`` of each target, in nats.

    Zero at ``y`` of exactly 0 or 1, where the limit of ``y log y`` is 0. The
    ``log`` is only evaluated on a clamped copy, so no ``nan`` is produced for
    saturated targets.

    Args:
        targets: ``(batch,)`` values in ``[0, 1]``.

    Returns:
        ``(batch,)`` entropies, each in ``[0, log 2]``.
    """
    y = targets.to(torch.float32)
    safe = y.clamp(min=torch.finfo(torch.float32).tiny, max=1.0)
    safe_c = (1.0 - y).clamp(min=torch.finfo(torch.float32).tiny, max=1.0)
    terms = y * torch.log(safe) + (1.0 - y) * torch.log(safe_c)
    return -terms


def bernoulli_kl_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL divergence between the target and predicted Bernoulli distributions.

    Equals :func:`bce_with_logits` minus the target entropy, matching the
    ``kl_divergence_from_logits`` helper used by the reference PNAS training
    script. Gradients are identical to BCE; only the reported value differs.

    Args:
        logits: ``(batch,)`` raw model outputs, pre-sigmoid.
        targets: ``(batch,)`` measured PSI values in ``[0, 1]``.
        weights: Optional ``(batch,)`` non-negative per-example weights.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Returns:
        Scalar loss, or ``(batch,)`` when ``reduction`` is ``"none"``.
    """
    per_example = bce_with_logits(logits, targets, None, reduction="none")
    per_example = per_example - bernoulli_entropy(targets).to(per_example.dtype)
    return _reduce(per_example, weights, reduction)


def get_loss_fn(name: str):
    """Return a loss function by name.

    Args:
        name: ``"bce"`` or ``"kl"``.

    Returns:
        A callable ``(logits, targets, weights, reduction) -> Tensor``.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name == "bce":
        return bce_with_logits
    if name == "kl":
        return bernoulli_kl_with_logits
    raise ValueError(
        f"unknown loss {name!r}; expected one of {list(LOSS_NAMES)}"
    )


def read_depth_weights(
    n_inclusion: np.ndarray,
    n_skipping: np.ndarray,
    normalise: bool = True,
) -> np.ndarray:
    """Build per-example loss weights from inclusion/skipping read counts.

    Weighting BCE by read depth makes the objective the Binomial negative
    log-likelihood of the observed counts, up to a parameter-free constant.

    Args:
        n_inclusion: Inclusion read counts.
        n_skipping: Skipping read counts.
        normalise: Divide by the mean depth so the weights average to 1.0,
            which keeps the loss magnitude and a tuned learning rate usable.

    Returns:
        Float32 array of weights, one per example.

    Raises:
        ValueError: If the arrays differ in length, hold negative counts, or
            any example has zero total depth.
    """
    inc = np.asarray(n_inclusion, dtype=np.float64)
    skp = np.asarray(n_skipping, dtype=np.float64)
    if inc.shape != skp.shape:
        raise ValueError(
            f"count arrays have differing shapes {inc.shape} and {skp.shape}"
        )
    if (inc < 0).any() or (skp < 0).any():
        raise ValueError("read counts must be non-negative")
    depth = inc + skp
    if (depth <= 0).any():
        raise ValueError(
            f"{int((depth <= 0).sum())} example(s) have zero inclusion+skipping "
            f"reads, so their PSI and weight are undefined"
        )
    if normalise:
        depth = depth / depth.mean()
    return depth.astype(np.float32)
