"""Model shapes, PSI bounds, padding/mask correctness and gradient flow."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from structsplice.config import FEATURE_CONFIGS, input_dim
from structsplice.dataset import PSIFeatureDataset, collate_batch
from structsplice.model import BiLSTMPSIRegressor


@pytest.mark.parametrize("config,dim", [("A", 4), ("B", 7), ("C", 12)])
def test_input_dim_per_config(config, dim):
    assert input_dim(config) == dim


@pytest.mark.parametrize("config", sorted(FEATURE_CONFIGS))
def test_forward_output_shape(config):
    model = BiLSTMPSIRegressor.from_config(config)
    x = torch.randn(5, 90, input_dim(config))
    logits = model(x)
    assert logits.shape == (5,)


@pytest.mark.parametrize("config", sorted(FEATURE_CONFIGS))
def test_predict_psi_is_in_unit_interval(config):
    model = BiLSTMPSIRegressor.from_config(config)
    x = torch.randn(7, 90, input_dim(config)) * 50.0  # push toward saturation
    psi = model.predict_psi(x)
    assert psi.shape == (7,)
    assert torch.all(psi >= 0.0) and torch.all(psi <= 1.0)


@pytest.mark.parametrize("config", sorted(FEATURE_CONFIGS))
def test_architecture_identical_except_input_layer(config):
    """Only the input projection may differ in size across configurations."""
    model = BiLSTMPSIRegressor.from_config(config)
    assert model.input_proj.in_features == input_dim(config)
    assert model.input_proj.out_features == 32
    assert model.lstm.hidden_size == 64
    assert model.lstm.bidirectional is True
    assert model.lstm.num_layers == 1
    # Head is 128 -> 64 -> 1.
    linears = [m for m in model.head if isinstance(m, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(128, 64), (64, 1)]


def test_wrong_input_dim_is_rejected():
    """Feeding config B features to a config A model must fail loudly."""
    model = BiLSTMPSIRegressor.from_config("A")
    with pytest.raises(ValueError, match="input_dim"):
        model(torch.randn(2, 90, 7))


def test_bad_hyperparameters_are_rejected():
    with pytest.raises(ValueError):
        BiLSTMPSIRegressor(input_dim=0)
    with pytest.raises(ValueError):
        BiLSTMPSIRegressor(input_dim=4, dropout=1.0)
    with pytest.raises(ValueError):
        BiLSTMPSIRegressor(input_dim=4, pooling="nonsense")


def test_parameter_count_grows_with_input_dim():
    a = BiLSTMPSIRegressor.from_config("A").parameter_count()
    c = BiLSTMPSIRegressor.from_config("C").parameter_count()
    assert c > a
    # The only difference is the projection weight: (12-4) * 32.
    assert c - a == (12 - 4) * 32


# ── Variable length and padding ────────────────────────────────────────────


def test_padding_does_not_change_the_prediction():
    """A padded batch must give the same answer as the unpadded sequence."""
    torch.manual_seed(0)
    model = BiLSTMPSIRegressor.from_config("A").eval()
    short = torch.randn(1, 30, 4)

    with torch.no_grad():
        alone = model(
            short, mask=torch.ones(1, 30, dtype=torch.bool),
            lengths=torch.tensor([30]),
        )
        padded_x = torch.cat([short, torch.randn(1, 60, 4)], dim=1)
        mask = torch.zeros(1, 90, dtype=torch.bool)
        mask[0, :30] = True
        padded = model(padded_x, mask=mask, lengths=torch.tensor([30]))

    assert torch.allclose(alone, padded, atol=1e-5)


def test_batch_order_does_not_change_predictions():
    torch.manual_seed(0)
    model = BiLSTMPSIRegressor.from_config("A").eval()
    x = torch.randn(4, 90, 4)
    mask = torch.ones(4, 90, dtype=torch.bool)
    lengths = torch.tensor([90, 90, 90, 90])
    with torch.no_grad():
        forward = model(x, mask=mask, lengths=lengths)
        reversed_ = model(
            x.flip(0), mask=mask, lengths=lengths
        ).flip(0)
    assert torch.allclose(forward, reversed_, atol=1e-5)


def test_ragged_batch_each_length_respected():
    torch.manual_seed(0)
    model = BiLSTMPSIRegressor.from_config("A").eval()
    lengths = torch.tensor([10, 45, 90])
    x = torch.randn(3, 90, 4)
    mask = torch.zeros(3, 90, dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, : int(n)] = True

    with torch.no_grad():
        batched = model(x, mask=mask, lengths=lengths)
        singles = torch.cat([
            model(
                x[i : i + 1, : int(n)],
                mask=torch.ones(1, int(n), dtype=torch.bool),
                lengths=torch.tensor([int(n)]),
            )
            for i, n in enumerate(lengths)
        ])
    assert torch.allclose(batched, singles, atol=1e-5)


@pytest.mark.parametrize("pooling", ["mean", "max", "last"])
def test_all_pooling_modes_respect_the_mask(pooling):
    torch.manual_seed(0)
    model = BiLSTMPSIRegressor(input_dim=4, pooling=pooling).eval()
    short = torch.randn(1, 20, 4)
    with torch.no_grad():
        alone = model(
            short, mask=torch.ones(1, 20, dtype=torch.bool),
            lengths=torch.tensor([20]),
        )
        padded_x = torch.cat([short, torch.full((1, 40, 4), 99.0)], dim=1)
        mask = torch.zeros(1, 60, dtype=torch.bool)
        mask[0, :20] = True
        padded = model(padded_x, mask=mask, lengths=torch.tensor([20]))
    assert torch.allclose(alone, padded, atol=1e-5)


def test_forward_without_mask_treats_all_positions_as_real():
    model = BiLSTMPSIRegressor.from_config("A").eval()
    x = torch.randn(2, 15, 4)
    with torch.no_grad():
        implicit = model(x)
        explicit = model(x, mask=torch.ones(2, 15, dtype=torch.bool))
    assert torch.allclose(implicit, explicit, atol=1e-6)


# ── Dataset integration: tensor dimensions for all three configs ───────────


def test_config_a_batch_dimensions(synthetic_dataset):
    ds = PSIFeatureDataset(synthetic_dataset, "A")
    batch = collate_batch([ds[i] for i in range(4)])
    assert batch["features"].shape == (4, 90, 4)
    assert batch["mask"].shape == (4, 90)
    assert batch["targets"].shape == (4,)
    assert batch["mask"].all()


def test_config_b_batch_dimensions(synthetic_dataset, synthetic_fold_cache):
    ds = PSIFeatureDataset(
        synthetic_dataset, "B", fold_cache=synthetic_fold_cache,
        use_npz_structure=False,
    )
    batch = collate_batch([ds[i] for i in range(4)])
    assert batch["features"].shape == (4, 90, 7)


def test_config_c_batch_dimensions(
    synthetic_dataset, synthetic_accessibility_cache
):
    ds = PSIFeatureDataset(
        synthetic_dataset, "C",
        accessibility_cache=synthetic_accessibility_cache,
    )
    batch = collate_batch([ds[i] for i in range(4)])
    assert batch["features"].shape == (4, 90, 12)


@pytest.mark.parametrize("config", sorted(FEATURE_CONFIGS))
def test_dataset_batch_feeds_model(
    config, synthetic_dataset, synthetic_fold_cache,
    synthetic_accessibility_cache,
):
    ds = PSIFeatureDataset(
        synthetic_dataset, config,
        fold_cache=synthetic_fold_cache,
        accessibility_cache=synthetic_accessibility_cache,
        use_npz_structure=False,
    )
    batch = collate_batch([ds[i] for i in range(6)])
    model = BiLSTMPSIRegressor.from_config(config)
    logits = model(
        batch["features"], mask=batch["mask"], lengths=batch["lengths"]
    )
    assert logits.shape == (6,)
    psi = torch.sigmoid(logits)
    assert torch.all((psi >= 0) & (psi <= 1))


def test_config_c_without_cache_is_rejected(synthetic_dataset):
    with pytest.raises(ValueError, match="accessibility"):
        PSIFeatureDataset(synthetic_dataset, "C")


def test_config_b_without_any_structure_source_is_rejected(synthetic_dataset):
    stripped = synthetic_dataset.subset(range(5))
    stripped.structure = None
    with pytest.raises(ValueError, match="dot-bracket"):
        PSIFeatureDataset(stripped, "B", use_npz_structure=True)


def test_unknown_config_is_rejected(synthetic_dataset):
    with pytest.raises(ValueError):
        PSIFeatureDataset(synthetic_dataset, "Z")


def test_collate_rejects_mixed_feature_dims(
    synthetic_dataset, synthetic_accessibility_cache
):
    a = PSIFeatureDataset(synthetic_dataset, "A")
    c = PSIFeatureDataset(
        synthetic_dataset, "C",
        accessibility_cache=synthetic_accessibility_cache,
    )
    with pytest.raises(ValueError, match="feature dimensions"):
        collate_batch([a[0], c[1]])


def test_collate_rejects_empty_batch():
    with pytest.raises(ValueError):
        collate_batch([])


def test_collate_pads_ragged_examples():
    items = []
    for n in (10, 25, 40):
        items.append({
            "features": torch.randn(n, 4),
            "target": torch.tensor(0.5),
            "weight": torch.tensor(1.0),
            "length": torch.tensor(n),
            "index": torch.tensor(0),
        })
    batch = collate_batch(items)
    assert batch["features"].shape == (3, 40, 4)
    assert batch["mask"][0].sum() == 10
    assert batch["mask"][1].sum() == 25
    assert batch["mask"][2].sum() == 40
    # Padding must be exactly zero.
    assert torch.all(batch["features"][0, 10:] == 0)
