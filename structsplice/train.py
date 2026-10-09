"""Training loop, early stopping and epoch logging.

The same loop trains all three feature configurations. Only the per-position
channel count differs between them, so any accuracy gap is attributable to the
features rather than to capacity or schedule. See ``README.md`` for the
fairness argument.

The original PNAS test split is never read here. Validation comes from a
grouped split of the training rows only, so model selection and early stopping
cannot see held-out data.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cache import AccessibilityCache, FoldCache
from .config import CONFIG_DESCRIPTIONS, FEATURE_CONFIGS, input_dim
from .data import SplicingDataset, grouped_train_val_split
from .dataset import PSIFeatureDataset, collate_batch
from .losses import get_loss_fn, read_depth_weights
from .metrics import format_metrics_line, regression_metrics
from .model import BiLSTMPSIRegressor
from .runtime import (
    describe_device,
    resolve_device,
    save_checkpoint,
    seeded_generator,
    set_seed,
    write_json,
)

LOGGER = logging.getLogger("structsplice.train")


@dataclasses.dataclass
class TrainSettings:
    """Everything that defines one training run.

    Attributes:
        config: Feature configuration ``"A"``, ``"B"`` or ``"C"``.
        epochs: Maximum number of epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        loss: ``"bce"`` or ``"kl"``; both share the same gradient.
        weight_by_depth: Weight each row by its sequencing depth.
        val_fraction: Fraction of training groups held out for validation.
        seed: Random seed for the split, initialisation and shuffling.
        patience: Early-stopping patience in epochs; ``0`` disables it.
        min_delta: Minimum val-loss improvement that counts as progress.
        grad_clip: Gradient-norm clip; ``0`` disables it.
        num_workers: DataLoader worker processes.
        device: ``"cpu"``, ``"cuda"``, ``"mps"`` or ``None`` to auto-detect.
        max_train_batches: Stop each training epoch after this many batches.
            Used by the smoke tests; ``0`` means no limit.
        proj_dim: Input projection width.
        hidden_size: LSTM units per direction.
        head_dim: Prediction-head hidden width.
        dropout: Dropout probability.
        pooling: Pooling mode.
        num_layers: Stacked LSTM layers.
    """

    config: str
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loss: str = "bce"
    weight_by_depth: bool = False
    val_fraction: float = 0.1
    seed: int = 42
    patience: int = 5
    min_delta: float = 0.0
    grad_clip: float = 0.0
    num_workers: int = 0
    device: str | None = None
    max_train_batches: int = 0
    proj_dim: int = 32
    hidden_size: int = 64
    head_dim: int = 64
    dropout: float = 0.2
    pooling: str = "mean"
    num_layers: int = 1

    def __post_init__(self) -> None:
        if self.config not in FEATURE_CONFIGS:
            raise ValueError(
                f"unknown feature config {self.config!r}; expected one of "
                f"{sorted(FEATURE_CONFIGS)}"
            )
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive; got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive; got {self.batch_size}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive; got {self.learning_rate}"
            )
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(
                f"val_fraction must lie in (0, 1); got {self.val_fraction}"
            )

    def model_kwargs(self) -> dict[str, Any]:
        """Return the architecture keyword arguments."""
        return {
            "proj_dim": self.proj_dim,
            "hidden_size": self.hidden_size,
            "head_dim": self.head_dim,
            "dropout": self.dropout,
            "pooling": self.pooling,
            "num_layers": self.num_layers,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the settings."""
        return dataclasses.asdict(self)


def build_feature_dataset(
    split: SplicingDataset,
    settings: TrainSettings,
    fold_cache: FoldCache | None = None,
    accessibility_cache: AccessibilityCache | None = None,
    use_npz_structure: bool = True,
) -> PSIFeatureDataset:
    """Wrap a split in a feature dataset, attaching depth weights if asked.

    Args:
        split: The rows to wrap.
        settings: Run settings; supplies the config and weighting choice.
        fold_cache: RNAfold cache for config ``"B"``.
        accessibility_cache: RNAplfold cache for config ``"C"``.
        use_npz_structure: Prefer the dataset's precomputed structures.

    Returns:
        The feature dataset.

    Raises:
        ValueError: If depth weighting is requested but read counts are absent.
    """
    weights = None
    if settings.weight_by_depth:
        if split.n_inclusion is None or split.n_skipping is None:
            raise ValueError(
                "weight_by_depth requires the metadata_num_exon_inclusion and "
                "metadata_num_exon_skipping fields, which this split lacks"
            )
        weights = read_depth_weights(split.n_inclusion, split.n_skipping)
    return PSIFeatureDataset(
        split,
        settings.config,
        fold_cache=fold_cache,
        accessibility_cache=accessibility_cache,
        use_npz_structure=use_npz_structure,
        weights=weights,
    )


def run_epoch(
    model: BiLSTMPSIRegressor,
    loader: Iterable[dict[str, torch.Tensor]],
    loss_fn,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
    max_batches: int = 0,
) -> dict[str, Any]:
    """Run one pass over a loader, training if an optimizer is given.

    Args:
        model: The model.
        loader: Batches from :func:`~structsplice.dataset.collate_batch`.
        loss_fn: Callable ``(logits, targets, weights) -> scalar``.
        device: Device to move batches to.
        optimizer: If given, backpropagate and step; otherwise evaluate.
        grad_clip: Gradient-norm clip; ``0`` disables it.
        max_batches: Stop after this many batches; ``0`` means no limit.

    Returns:
        Dict with ``loss`` (weighted mean over examples), ``n_batches``,
        ``predictions`` and ``targets``.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_n = 0
    n_batches = 0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            target = batch["targets"].to(device, non_blocking=True)
            weight = batch["weights"].to(device, non_blocking=True)

            logits = model(features, mask=mask, lengths=lengths)
            loss = loss_fn(logits, target, weight)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip
                    )
                optimizer.step()

            batch_n = int(target.numel())
            total_loss += float(loss.detach()) * batch_n
            total_n += batch_n
            n_batches += 1

            preds.append(torch.sigmoid(logits.detach()).cpu().numpy())
            targets.append(target.detach().cpu().numpy())

            if max_batches and n_batches >= max_batches:
                break

    return {
        "loss": total_loss / total_n if total_n else float("nan"),
        "n_batches": n_batches,
        "n_examples": total_n,
        "predictions": (
            np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)
        ),
        "targets": (
            np.concatenate(targets)
            if targets
            else np.zeros(0, dtype=np.float32)
        ),
    }


def train(
    dataset: SplicingDataset,
    settings: TrainSettings,
    output_dir: str | Path,
    fold_cache: FoldCache | None = None,
    accessibility_cache: AccessibilityCache | None = None,
    use_npz_structure: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Train one model and write checkpoints, logs and a run summary.

    Args:
        dataset: The **training** split only. The PNAS test split must not be
            passed here.
        settings: Run settings.
        output_dir: Directory for checkpoints, the epoch log and the summary.
        fold_cache: RNAfold cache for config ``"B"``.
        accessibility_cache: RNAplfold cache for config ``"C"``.
        use_npz_structure: Prefer the dataset's precomputed structures.
        logger: Logger; defaults to the module logger.

    Returns:
        Dict with the run ``history``, ``best_epoch``, ``best_val_loss``, the
        checkpoint paths, and the resolved ``settings``.
    """
    log = logger or LOGGER
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    set_seed(settings.seed)
    device = resolve_device(settings.device)
    log.info("device: %s", describe_device(device))
    log.info(
        "config %s: %s (input_dim=%d)",
        settings.config,
        CONFIG_DESCRIPTIONS[settings.config],
        input_dim(settings.config),
    )

    split = grouped_train_val_split(
        dataset,
        val_fraction=settings.val_fraction,
        seed=settings.seed,
    )
    log.info(
        "grouped split: %d train rows, %d val rows, %d/%d groups, "
        "no group on both sides",
        len(split.train),
        len(split.val),
        split.n_train_groups,
        split.n_val_groups,
    )

    train_split = dataset.subset(split.train)
    val_split = dataset.subset(split.val)

    train_features = build_feature_dataset(
        train_split,
        settings,
        fold_cache,
        accessibility_cache,
        use_npz_structure,
    )
    val_features = build_feature_dataset(
        val_split,
        settings,
        fold_cache,
        accessibility_cache,
        use_npz_structure,
    )

    generator = seeded_generator(settings.seed)
    train_loader = DataLoader(
        train_features,
        batch_size=settings.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=settings.num_workers,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_features,
        batch_size=settings.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=settings.num_workers,
        drop_last=False,
    )

    model = BiLSTMPSIRegressor.from_config(
        settings.config, **settings.model_kwargs()
    ).to(device)
    log.info("model parameters: %d", model.parameter_count())

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    loss_fn = get_loss_fn(settings.loss)

    best_path = out / "best.pt"
    last_path = out / "last.pt"
    history_path = out / "history.jsonl"
    history: list[dict[str, Any]] = []
    best_val = math.inf
    best_epoch = -1
    epochs_without_gain = 0
    stopped_early = False

    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, settings.epochs + 1):
            t0 = time.time()
            tr = run_epoch(
                model,
                train_loader,
                loss_fn,
                device,
                optimizer=optimizer,
                grad_clip=settings.grad_clip,
                max_batches=settings.max_train_batches,
            )
            va = run_epoch(model, val_loader, loss_fn, device)
            val_metrics = regression_metrics(va["predictions"], va["targets"])
            elapsed = time.time() - t0

            improved = va["loss"] < best_val - settings.min_delta
            if improved:
                best_val = va["loss"]
                best_epoch = epoch
                epochs_without_gain = 0
                save_checkpoint(
                    best_path,
                    model,
                    epoch,
                    metadata={
                        "settings": settings.to_dict(),
                        "val_loss": va["loss"],
                        "train_loss": tr["loss"],
                        "val_metrics": val_metrics,
                    },
                    optimizer=optimizer,
                )
            else:
                epochs_without_gain += 1

            record = {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "val_loss": va["loss"],
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "val_pearson_r": val_metrics["pearson_r"],
                "val_r2": val_metrics["r2"],
                "train_batches": tr["n_batches"],
                "seconds": round(elapsed, 2),
                "improved": improved,
            }
            history.append(record)
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()

            log.info(
                "epoch %d/%d train_loss=%.5f val_loss=%.5f %s (%.1fs)%s",
                epoch,
                settings.epochs,
                tr["loss"],
                va["loss"],
                format_metrics_line(val_metrics),
                elapsed,
                " *" if improved else "",
            )

            if settings.patience and epochs_without_gain >= settings.patience:
                log.info(
                    "early stopping: no val improvement for %d epochs "
                    "(best epoch %d, val_loss=%.5f)",
                    epochs_without_gain,
                    best_epoch,
                    best_val,
                )
                stopped_early = True
                break

    save_checkpoint(
        last_path,
        model,
        history[-1]["epoch"] if history else 0,
        metadata={"settings": settings.to_dict(), "final": True},
        optimizer=optimizer,
    )

    summary = {
        "settings": settings.to_dict(),
        "input_dim": input_dim(settings.config),
        "device": describe_device(device),
        "parameters": model.parameter_count(),
        "n_train_rows": len(split.train),
        "n_val_rows": len(split.val),
        "n_train_groups": split.n_train_groups,
        "n_val_groups": split.n_val_groups,
        "best_epoch": best_epoch,
        "best_val_loss": None if best_epoch < 0 else best_val,
        "stopped_early": stopped_early,
        "epochs_run": len(history),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history": history,
    }
    write_json(out / "train_summary.json", summary)
    return summary
