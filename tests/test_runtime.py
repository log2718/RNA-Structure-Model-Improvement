"""Seeding, device resolution, checkpoint round-trips and JSON output."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from structsplice.model import BiLSTMPSIRegressor
from structsplice.runtime import (
    describe_device,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    seeded_generator,
    set_seed,
    write_json,
)


def test_set_seed_makes_torch_reproducible():
    set_seed(123)
    a = torch.randn(10)
    set_seed(123)
    b = torch.randn(10)
    assert torch.equal(a, b)


def test_set_seed_makes_numpy_and_random_reproducible():
    import random

    set_seed(7)
    a = (np.random.rand(5).tolist(), random.random())
    set_seed(7)
    b = (np.random.rand(5).tolist(), random.random())
    assert a == b


def test_different_seeds_diverge():
    set_seed(1)
    a = torch.randn(10)
    set_seed(2)
    b = torch.randn(10)
    assert not torch.equal(a, b)


def test_model_init_is_seed_reproducible():
    set_seed(42)
    first = BiLSTMPSIRegressor.from_config("A")
    set_seed(42)
    second = BiLSTMPSIRegressor.from_config("A")
    for p, q in zip(first.parameters(), second.parameters()):
        assert torch.equal(p, q)


def test_seeded_generator_is_reproducible():
    a = torch.randperm(20, generator=seeded_generator(5))
    b = torch.randperm(20, generator=seeded_generator(5))
    assert torch.equal(a, b)


def test_resolve_device_defaults_to_something_available():
    device = resolve_device(None)
    assert device.type in {"cpu", "cuda", "mps"}


def test_resolve_device_honours_cpu_request():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_rejects_unavailable_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available here, so the error path cannot be hit")
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_describe_device_mentions_the_type():
    assert "cpu" in describe_device(torch.device("cpu")).lower()


# ── Checkpoints ────────────────────────────────────────────────────────────


def test_checkpoint_round_trip_restores_weights(tmp_path):
    set_seed(0)
    model = BiLSTMPSIRegressor.from_config("C")
    path = save_checkpoint(
        tmp_path / "ck.pt", model, epoch=3,
        metadata={"settings": {"config": "C", "seed": 0}, "val_loss": 0.42},
    )
    assert path.is_file()

    restored, payload = load_checkpoint(path, device="cpu")
    # epoch and hparams sit at the top level; caller provenance is nested.
    assert payload["epoch"] == 3
    assert payload["hparams"]["input_dim"] == 12
    assert payload["metadata"]["settings"]["config"] == "C"
    assert payload["metadata"]["val_loss"] == pytest.approx(0.42)
    for p, q in zip(model.parameters(), restored.parameters()):
        assert torch.equal(p, q)


def test_restored_model_gives_identical_predictions(tmp_path):
    set_seed(0)
    model = BiLSTMPSIRegressor.from_config("B").eval()
    x = torch.randn(4, 90, 7)
    with torch.no_grad():
        before = model.predict_psi(x)

    path = save_checkpoint(
        tmp_path / "ck.pt", model, epoch=1,
        metadata={"settings": {"config": "B"}},
    )
    restored, _ = load_checkpoint(path, device="cpu")
    restored.eval()
    with torch.no_grad():
        after = restored.predict_psi(x)
    assert torch.allclose(before, after, atol=1e-6)


def test_checkpoint_rebuilds_architecture_from_hparams(tmp_path):
    """Loading must not require the caller to know the architecture."""
    model = BiLSTMPSIRegressor(
        input_dim=12, proj_dim=16, hidden_size=8, head_dim=4, pooling="max"
    )
    path = save_checkpoint(tmp_path / "ck.pt", model, epoch=0)
    restored, _ = load_checkpoint(path, device="cpu")
    assert restored.hparams == model.hparams
    assert restored.pooling == "max"


def test_checkpoint_can_load_into_an_existing_model(tmp_path):
    set_seed(0)
    source = BiLSTMPSIRegressor.from_config("A")
    path = save_checkpoint(tmp_path / "ck.pt", source, epoch=0)

    set_seed(999)
    target = BiLSTMPSIRegressor.from_config("A")
    assert not all(
        torch.equal(p, q)
        for p, q in zip(source.parameters(), target.parameters())
    )
    loaded, _ = load_checkpoint(path, device="cpu", model=target)
    assert loaded is target
    for p, q in zip(source.parameters(), target.parameters()):
        assert torch.equal(p, q)


def test_checkpoint_stores_optimizer_state(tmp_path):
    model = BiLSTMPSIRegressor.from_config("A")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.randn(2, 20, 4)).sum().backward()
    optimizer.step()

    path = save_checkpoint(
        tmp_path / "ck.pt", model, epoch=1, optimizer=optimizer
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" in payload


def test_loading_a_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


# ── JSON output ────────────────────────────────────────────────────────────


def test_write_json_round_trip(tmp_path):
    payload = {"a": 1, "b": [1, 2, 3], "c": {"d": "text"}}
    path = write_json(tmp_path / "out.json", payload)
    assert json.loads(path.read_text()) == payload


def test_write_json_converts_nan_to_null(tmp_path):
    """NaN is not valid JSON; undefined metrics must serialise as null."""
    path = write_json(
        tmp_path / "m.json",
        {"pearson_r": math.nan, "mse": 0.25, "note": "targets are constant"},
    )
    text = path.read_text()
    assert "NaN" not in text
    loaded = json.loads(text)
    assert loaded["pearson_r"] is None
    assert loaded["mse"] == 0.25


def test_write_json_handles_numpy_types(tmp_path):
    path = write_json(
        tmp_path / "n.json",
        {"i": np.int64(3), "f": np.float32(0.5), "arr": np.arange(3)},
    )
    loaded = json.loads(path.read_text())
    assert loaded["i"] == 3
    assert loaded["f"] == pytest.approx(0.5)
    assert loaded["arr"] == [0, 1, 2]


def test_write_json_creates_parent_directories(tmp_path):
    path = write_json(tmp_path / "deep" / "nested" / "x.json", {"ok": True})
    assert path.is_file()
