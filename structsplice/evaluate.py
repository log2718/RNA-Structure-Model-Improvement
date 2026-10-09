"""Evaluation on a held-out split, with per-row prediction export.

The procedure here is identical for all three feature configurations: the
same loader, the same metric set, the same CSV layout. Only the checkpoint and
its feature config change, which is what makes the three runs comparable.

A single run is never evidence of an improvement. :func:`aggregate_runs`
summarises several seeds so the eventual comparison can report a spread
instead of one number.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cache import AccessibilityCache, FoldCache
from .config import CONFIG_DESCRIPTIONS, input_dim
from .data import SplicingDataset
from .dataset import PSIFeatureDataset, collate_batch
from .losses import get_loss_fn
from .metrics import METRIC_KEYS, regression_metrics
from .model import BiLSTMPSIRegressor
from .runtime import describe_device, load_checkpoint, resolve_device, write_json

LOGGER = logging.getLogger("structsplice.evaluate")

#: Columns written to the predictions CSV.
PREDICTION_COLUMNS = ("row_index", "exon_id", "model_sequence", "psi_true", "psi_pred")


def predict(
    model: BiLSTMPSIRegressor,
    features: PSIFeatureDataset,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 0,
    loss_name: str | None = None,
) -> dict[str, Any]:
    """Run the model over a feature dataset in row order.

    Args:
        model: A trained model.
        features: The feature dataset to score.
        device: Device to run on.
        batch_size: Evaluation batch size.
        num_workers: DataLoader workers.
        loss_name: If given, also report the mean loss under this objective.

    Returns:
        Dict with ``predictions``, ``targets``, ``row_index`` and, when
        ``loss_name`` is given, ``loss``.
    """
    loader = DataLoader(
        features,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
    )
    loss_fn = get_loss_fn(loss_name) if loss_name else None

    model.eval().to(device)
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"].to(device)
            target = batch["targets"].to(device)

            logits = model(feats, mask=mask, lengths=lengths)
            if loss_fn is not None:
                weight = batch["weights"].to(device)
                n = int(target.numel())
                total_loss += float(loss_fn(logits, target, weight)) * n
                total_n += n

            preds.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(target.cpu().numpy())
            indices.append(batch["indices"].cpu().numpy())

    result = {
        "predictions": np.concatenate(preds) if preds else np.zeros(0, np.float32),
        "targets": np.concatenate(targets) if targets else np.zeros(0, np.float32),
        "row_index": np.concatenate(indices) if indices else np.zeros(0, np.int64),
    }
    if loss_fn is not None:
        result["loss"] = total_loss / total_n if total_n else float("nan")
    return result


def write_predictions(
    path: str | Path,
    dataset: SplicingDataset,
    row_index: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> Path:
    """Write per-row predictions to CSV, carrying exon identifiers.

    Args:
        path: Destination CSV path; parent directories are created.
        dataset: The evaluated split, used to look up identifiers.
        row_index: Row indices into ``dataset``, as returned by :func:`predict`.
        targets: True PSI values aligned with ``row_index``.
        predictions: Predicted PSI values aligned with ``row_index``.

    Returns:
        The resolved destination path.

    Raises:
        ValueError: If the three arrays have differing lengths.
    """
    if not (len(row_index) == len(targets) == len(predictions)):
        raise ValueError(
            f"length mismatch: row_index={len(row_index)}, "
            f"targets={len(targets)}, predictions={len(predictions)}"
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exon = dataset.exon
    sequence = dataset.model_sequence
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PREDICTION_COLUMNS)
        for i, true, pred in zip(row_index, targets, predictions):
            idx = int(i)
            writer.writerow(
                [
                    idx,
                    str(exon[idx]) if exon is not None else "",
                    str(sequence[idx]),
                    f"{float(true):.6f}",
                    f"{float(pred):.6f}",
                ]
            )
    return out


def evaluate_checkpoint(
    checkpoint: str | Path,
    dataset: SplicingDataset,
    output_dir: str | Path,
    config: str | None = None,
    fold_cache: FoldCache | None = None,
    accessibility_cache: AccessibilityCache | None = None,
    use_npz_structure: bool = True,
    batch_size: int = 256,
    num_workers: int = 0,
    device: str | None = None,
    loss_name: str = "bce",
    split_name: str = "test",
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint on one split and write metrics plus predictions.

    Args:
        checkpoint: Path to a checkpoint written by
            :func:`~structsplice.runtime.save_checkpoint`.
        dataset: The split to evaluate.
        output_dir: Directory for ``metrics.json`` and ``predictions.csv``.
        config: Feature configuration. Defaults to the one recorded in the
            checkpoint metadata.
        fold_cache: RNAfold cache for config ``"B"``.
        accessibility_cache: RNAplfold cache for config ``"C"``.
        use_npz_structure: Prefer the dataset's precomputed structures.
        batch_size: Evaluation batch size.
        num_workers: DataLoader workers.
        device: Device override; ``None`` auto-detects.
        loss_name: Objective used to report the held-out loss.
        split_name: Label recorded in the output files.
        logger: Logger; defaults to the module logger.

    Returns:
        Dict with ``metrics``, ``config``, the written file paths and the
        training history recorded in the checkpoint, when present.

    Raises:
        ValueError: If no feature config is given and none is recorded in the
            checkpoint.
    """
    log = logger or LOGGER
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device)
    model, payload = load_checkpoint(checkpoint, device=resolved_device)

    # save_checkpoint nests caller-supplied provenance under "metadata";
    # "epoch" and "hparams" live at the top level of the payload.
    metadata = payload.get("metadata") or {}
    settings = metadata.get("settings") or {}
    resolved_config = config or settings.get("config")
    if resolved_config is None:
        raise ValueError(
            f"{checkpoint} records no feature config; pass config explicitly"
        )

    expected = input_dim(resolved_config)
    actual = model.hparams["input_dim"]
    if actual != expected:
        raise ValueError(
            f"checkpoint was built with input_dim={actual} but config "
            f"{resolved_config!r} implies {expected}"
        )

    log.info(
        "evaluating %s on %s: config %s, %s, device %s",
        Path(checkpoint).name,
        split_name,
        resolved_config,
        CONFIG_DESCRIPTIONS[resolved_config],
        describe_device(resolved_device),
    )

    features = PSIFeatureDataset(
        dataset,
        resolved_config,
        fold_cache=fold_cache,
        accessibility_cache=accessibility_cache,
        use_npz_structure=use_npz_structure,
    )
    result = predict(
        model,
        features,
        resolved_device,
        batch_size=batch_size,
        num_workers=num_workers,
        loss_name=loss_name,
    )
    metrics = regression_metrics(result["predictions"], result["targets"])
    metrics[f"{loss_name}_loss"] = result["loss"]

    csv_path = write_predictions(
        out / "predictions.csv",
        dataset,
        result["row_index"],
        result["targets"],
        result["predictions"],
    )

    train_history = metadata.get("val_metrics")
    payload = {
        "split": split_name,
        "config": resolved_config,
        "config_description": CONFIG_DESCRIPTIONS[resolved_config],
        "input_dim": expected,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "seed": settings.get("seed"),
        "loss_name": loss_name,
        "train_loss_at_checkpoint": metadata.get("train_loss"),
        "val_loss_at_checkpoint": metadata.get("val_loss"),
        "val_metrics_at_checkpoint": train_history,
        "metrics": metrics,
        "predictions_csv": str(csv_path),
        "settings": settings,
    }
    json_path = write_json(out / "metrics.json", payload)
    payload["metrics_json"] = str(json_path)
    return payload


def aggregate_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarise several evaluation payloads across seeds.

    Args:
        runs: Payloads returned by :func:`evaluate_checkpoint`.

    Returns:
        Dict with ``n_runs``, the ``seeds`` seen, and per-metric ``mean``,
        ``std`` (population) and ``values``. Non-numeric metric entries, such
        as the explanatory string a constant target produces, are skipped.

    Raises:
        ValueError: If ``runs`` is empty.
    """
    if not runs:
        raise ValueError("aggregate_runs needs at least one run")
    configs = {r.get("config") for r in runs}
    summary: dict[str, Any] = {
        "n_runs": len(runs),
        "configs": sorted(c for c in configs if c is not None),
        "seeds": [r.get("seed") for r in runs],
        "metrics": {},
    }
    for key in METRIC_KEYS:
        values = [
            float(r["metrics"][key])
            for r in runs
            if isinstance(r.get("metrics", {}).get(key), (int, float))
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        summary["metrics"][key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "values": values,
        }
    return summary
