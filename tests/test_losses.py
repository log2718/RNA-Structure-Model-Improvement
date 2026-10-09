"""Loss behaviour on fractional PSI targets, and the BCE/KL relationship."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from structsplice.losses import (
    LOSS_NAMES,
    bce_with_logits,
    bernoulli_entropy,
    bernoulli_kl_with_logits,
    get_loss_fn,
    read_depth_weights,
)


def test_bce_accepts_fractional_targets():
    logits = torch.zeros(4)
    targets = torch.tensor([0.0, 0.3, 0.75, 1.0])
    loss = bce_with_logits(logits, targets)
    assert torch.isfinite(loss)
    # At logit 0 the prediction is 0.5, so the loss is log 2 for every target.
    assert float(loss) == pytest.approx(math.log(2.0), abs=1e-6)


def test_bce_matches_torch_reference():
    torch.manual_seed(0)
    logits = torch.randn(16)
    targets = torch.rand(16)
    expected = torch.nn.BCEWithLogitsLoss()(logits, targets)
    assert torch.allclose(bce_with_logits(logits, targets), expected, atol=1e-6)


def test_bce_is_minimised_at_the_true_target():
    target = torch.tensor([0.3])
    best = torch.logit(target)
    losses = [
        float(bce_with_logits(best + delta, target))
        for delta in (-1.0, -0.1, 0.0, 0.1, 1.0)
    ]
    assert losses[2] == min(losses)


def test_saturated_targets_stay_finite():
    """PSI of exactly 0 and 1 occur in this corpus and must not produce NaN."""
    logits = torch.tensor([20.0, -20.0])
    targets = torch.tensor([1.0, 0.0])
    assert torch.isfinite(bce_with_logits(logits, targets))
    targets_flipped = torch.tensor([0.0, 1.0])
    assert torch.isfinite(bce_with_logits(logits, targets_flipped))


def test_bernoulli_entropy_is_zero_at_the_extremes():
    h = bernoulli_entropy(torch.tensor([0.0, 1.0]))
    assert torch.allclose(h, torch.zeros(2), atol=1e-6)


def test_bernoulli_entropy_peaks_at_one_half():
    h = bernoulli_entropy(torch.tensor([0.1, 0.5, 0.9]))
    assert float(h[1]) == pytest.approx(math.log(2.0), abs=1e-6)
    assert float(h[1]) > float(h[0]) and float(h[1]) > float(h[2])


def test_kl_equals_bce_minus_target_entropy():
    """The documented identity: Bernoulli KL = BCE - H(y)."""
    torch.manual_seed(0)
    logits = torch.randn(32)
    targets = torch.rand(32)
    kl = bernoulli_kl_with_logits(logits, targets)
    bce = bce_with_logits(logits, targets)
    entropy = bernoulli_entropy(targets).mean()
    assert float(kl) == pytest.approx(float(bce - entropy), abs=1e-5)


def test_kl_is_non_negative_and_zero_at_a_perfect_fit():
    targets = torch.tensor([0.2, 0.5, 0.8])
    perfect = torch.logit(targets)
    kl = bernoulli_kl_with_logits(perfect, targets)
    assert float(kl) == pytest.approx(0.0, abs=1e-6)
    assert float(kl) >= -1e-7


def test_bce_does_not_reach_zero_at_a_perfect_fit():
    """BCE bottoms out at H(y) > 0, which is why the two values differ."""
    targets = torch.tensor([0.3])
    perfect = torch.logit(targets)
    bce = float(bce_with_logits(perfect, targets))
    assert bce > 0.1
    assert bce == pytest.approx(float(bernoulli_entropy(targets).mean()), abs=1e-6)


def test_bce_and_kl_have_identical_gradients():
    """Differing by a target-only constant means the same optimisation."""
    torch.manual_seed(0)
    targets = torch.rand(10)

    a = torch.randn(10, requires_grad=True)
    bce_with_logits(a, targets).backward()

    b = a.detach().clone().requires_grad_(True)
    bernoulli_kl_with_logits(b, targets).backward()

    assert torch.allclose(a.grad, b.grad, atol=1e-6)


@pytest.mark.parametrize("name", LOSS_NAMES)
def test_get_loss_fn_returns_a_usable_callable(name):
    fn = get_loss_fn(name)
    loss = fn(torch.randn(5), torch.rand(5), torch.ones(5))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_get_loss_fn_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_loss_fn("hinge")


def test_weights_scale_the_loss():
    logits = torch.zeros(2)
    targets = torch.tensor([0.5, 0.5])
    plain = bce_with_logits(logits, targets)
    doubled = bce_with_logits(logits, targets, weights=torch.tensor([2.0, 2.0]))
    assert float(doubled) == pytest.approx(float(plain), abs=1e-6)


def test_zero_weight_removes_an_example():
    logits = torch.tensor([0.0, 50.0])
    targets = torch.tensor([0.5, 0.0])
    weights = torch.tensor([1.0, 0.0])
    loss = bce_with_logits(logits, targets, weights=weights)
    assert float(loss) == pytest.approx(math.log(2.0), abs=1e-5)


def test_read_depth_weights_are_normalised_and_proportional():
    inclusion = np.array([5, 50, 500])
    skipping = np.array([5, 50, 500])
    w = read_depth_weights(inclusion, skipping)
    assert w.mean() == pytest.approx(1.0, abs=1e-6)
    assert w[2] > w[1] > w[0]
    assert w[1] / w[0] == pytest.approx(10.0, abs=1e-5)


def test_read_depth_weights_can_stay_unnormalised():
    w = read_depth_weights(np.array([3]), np.array([7]), normalise=False)
    assert float(w[0]) == pytest.approx(10.0)


# ── Backpropagation and optimiser steps ───────────────────────────────────


def test_backward_populates_all_gradients():
    from structsplice.model import BiLSTMPSIRegressor

    torch.manual_seed(0)
    model = BiLSTMPSIRegressor.from_config("A")
    x = torch.randn(8, 90, 4)
    targets = torch.rand(8)
    loss = bce_with_logits(model(x), targets)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())


def test_optimizer_step_changes_parameters_and_reduces_loss():
    from structsplice.model import BiLSTMPSIRegressor

    torch.manual_seed(0)
    model = BiLSTMPSIRegressor.from_config("B")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(16, 90, 7)
    targets = torch.rand(16)

    before = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        first = float(bce_with_logits(model(x), targets))
    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        bce_with_logits(model(x), targets).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        last = float(bce_with_logits(model(x), targets))

    assert any(
        not torch.allclose(a, b)
        for a, b in zip(before, model.parameters())
    )
    assert last < first


def test_model_can_overfit_a_tiny_batch():
    """Sanity check that the architecture can actually learn a mapping."""
    from structsplice.model import BiLSTMPSIRegressor

    torch.manual_seed(0)
    model = BiLSTMPSIRegressor(input_dim=4, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    x = torch.randn(4, 20, 4)
    targets = torch.tensor([0.05, 0.35, 0.65, 0.95])
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        bce_with_logits(model(x), targets).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        psi = model.predict_psi(x)
    assert torch.allclose(psi, targets, atol=0.1), psi
