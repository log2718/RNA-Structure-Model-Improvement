"""Seeding, device selection and checkpoint I/O.

Kept separate from :mod:`structsplice.train` so evaluation and tests can reuse it
without importing the training loop.
"""

from __future__ import annotations

import json
import math
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch, and optionally force deterministic kernels.

    Determinism caveat: cuDNN's LSTM backward pass has no deterministic
    implementation, so a CUDA run is reproducible only up to that kernel.
    Setting ``deterministic`` leaves ``warn_only=True`` rather than raising, so
    training still runs on GPU while reporting where exactness is lost. CPU runs
    are bitwise reproducible.

    Args:
        seed: Seed applied to all three generators.
        deterministic: Request deterministic algorithms and disable cuDNN
            benchmark autotuning.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seeded_generator(seed: int) -> torch.Generator:
    """Return a CPU ``torch.Generator`` seeded for DataLoader shuffling.

    Passing an explicit generator to ``DataLoader`` makes the batch order a
    function of the seed alone, independent of how many other random draws the
    process has made.

    Args:
        seed: Seed value.

    Returns:
        A seeded generator.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick a torch device, preferring CUDA when available.

    Args:
        requested: Explicit device string such as ``"cpu"``, ``"cuda"`` or
            ``"cuda:1"``. When ``None`` or ``"auto"``, CUDA is used if present,
            otherwise Apple MPS, otherwise CPU.

    Returns:
        The resolved device.

    Raises:
        RuntimeError: If CUDA was explicitly requested but is unavailable.
    """
    if requested and requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"device {requested!r} was requested but torch reports no CUDA "
                f"support in this build or no visible GPU"
            )
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Return a human-readable description of a device.

    Args:
        device: The device to describe.

    Returns:
        A short string naming the device and, for CUDA, the GPU model.
    """
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        return f"{device} ({torch.cuda.get_device_name(index)})"
    return str(device)


def setup_logging(
    log_path: Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Configure and return the project logger.

    Args:
        log_path: Optional file to tee log records to. Parent directories are
            created.
        level: Logging level.

    Returns:
        The ``structsplice`` logger.
    """
    logger = logging.getLogger("structsplice")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    epoch: int,
    metadata: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    """Write a checkpoint holding weights, hyperparameters and provenance.

    The model's own ``hparams`` dict is stored alongside the weights so a
    checkpoint can be rebuilt without the caller having to remember which
    feature configuration produced it.

    Args:
        path: Destination ``.pt`` path; parent directories are created.
        model: Model to save. Must expose a ``hparams`` dict.
        epoch: Epoch index the weights come from.
        metadata: Extra JSON-serialisable provenance, e.g. seed and config.
        optimizer: Optional optimizer state, for resuming training.

    Returns:
        The resolved destination path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "hparams": dict(getattr(model, "hparams", {})),
        "epoch": epoch,
        "metadata": metadata or {},
        "format_version": 1,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, out)
    return out


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
    model: torch.nn.Module | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a checkpoint, rebuilding the model from stored hyperparameters.

    Args:
        path: Checkpoint path.
        device: Device to map tensors onto.
        model: Optional pre-built model to load weights into. When ``None``, a
            :class:`~structsplice.model.BiLSTMPSIRegressor` is constructed from
            the checkpoint's ``hparams``.

    Returns:
        ``(model, payload)`` where ``payload`` is the full checkpoint dict
        minus the tensors it is unsafe to keep around.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        KeyError: If the file is not a checkpoint written by
            :func:`save_checkpoint`.
    """
    from .model import BiLSTMPSIRegressor

    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" not in payload:
        raise KeyError(
            f"{ckpt_path} does not look like a structsplice checkpoint; it has "
            f"keys {sorted(payload)}"
        )

    if model is None:
        hparams = payload.get("hparams")
        if not hparams:
            raise KeyError(
                f"{ckpt_path} carries no 'hparams', so the model cannot be "
                f"rebuilt automatically. Pass an explicit model."
            )
        model = BiLSTMPSIRegressor(**hparams)

    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return model, payload


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    ``nan`` and ``inf`` are not valid JSON, and the bare ``NaN`` token Python
    emits by default is rejected by strict parsers, pandas' ``read_json`` and
    most non-Python tooling. Undefined metrics are legitimate here (a constant
    target makes Pearson's r undefined), so they are written as ``null`` and
    the companion ``*_note`` field carries the reason.

    Args:
        value: Any JSON-ish structure.

    Returns:
        The same structure with non-finite floats replaced by ``None``.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as indented, strictly valid JSON.

    Non-finite floats become ``null`` so the file parses everywhere; see
    :func:`_json_safe`.

    Args:
        path: Destination path.
        payload: JSON-serialisable object.

    Returns:
        The resolved destination path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(_json_safe(payload), indent=2, default=str, allow_nan=False)
        + "\n"
    )
    return out
