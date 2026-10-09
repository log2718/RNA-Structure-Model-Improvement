#!/usr/bin/env python3
"""Train one BiLSTM configuration on the PNAS training split.

Validation is carved out of the training split by exon group, so the PNAS test
split is never read by this script.

Examples
--------
Sequence-only baseline::

    python scripts/train.py --config A --out-dir outputs/A_seed0 --seed 0

Sequence + dot-bracket, using the structures shipped in the npz files::

    python scripts/train.py --config B --out-dir outputs/B_seed0 --seed 0

Sequence + accessibility, needing a preprocessed cache::

    python scripts/train.py --config C --out-dir outputs/C_seed0 --seed 0 \
        --accessibility-cache cache/accessibility.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from structsplice.cache import AccessibilityCache, FoldCache, shard_paths
from structsplice.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    FEATURE_CONFIGS,
)
from structsplice.data import load_npz
from structsplice.losses import LOSS_NAMES
from structsplice.model import POOLING_MODES
from structsplice.runtime import setup_logging
from structsplice.train import TrainSettings, train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, choices=sorted(FEATURE_CONFIGS),
                   help="feature configuration: A, B or C")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--train-npz", type=Path, default=None,
                   help="override the training npz path")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output directory (default: <outputs>/<config>_seed<seed>)")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--fold-cache", type=Path, default=None,
                   help="RNAfold cache; config B falls back to the npz structures")
    p.add_argument("--accessibility-cache", type=Path, default=None,
                   help="RNAplfold cache; required for config C")
    p.add_argument("--no-npz-structure", action="store_true",
                   help="for config B, require a fold cache instead of the npz field")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", "--learning-rate", dest="learning_rate",
                   type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--loss", choices=LOSS_NAMES, default="bce")
    p.add_argument("--weight-by-depth", action="store_true",
                   help="weight each row by its inclusion+skipping read count")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=5,
                   help="early-stopping patience in epochs; 0 disables")
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None,
                   help="cpu, cuda, mps; default auto-detects")
    p.add_argument("--max-train-batches", type=int, default=0,
                   help="cap training batches per epoch (smoke tests only)")
    p.add_argument("--limit-rows", type=int, default=0,
                   help="use only a reproducible random subsample of this many "
                        "training rows (smoke tests only, not for real runs)")

    p.add_argument("--proj-dim", type=int, default=32)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--pooling", choices=POOLING_MODES, default="mean")
    p.add_argument("--num-layers", type=int, default=1)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)

    out_dir = args.out_dir or (
        Path(args.output_root) / f"{args.config}_seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir / "train.log")

    train_npz = args.train_npz or (Path(args.data_dir) / "train_data.npz")
    dataset = load_npz(train_npz, name="train")
    log.info("train split: %s", dataset.summary())

    if args.limit_rows and args.limit_rows < len(dataset):
        rng = np.random.default_rng(args.seed)
        keep = np.sort(
            rng.choice(len(dataset), size=args.limit_rows, replace=False)
        )
        dataset = dataset.subset(keep)
        log.warning(
            "--limit-rows %d: training on a subsample, NOT a full run. %s",
            args.limit_rows, dataset.summary(),
        )

    fold_cache = None
    if args.fold_cache is not None:
        paths = shard_paths(args.fold_cache)
        if not paths:
            log.error("no RNAfold cache found at %s", args.fold_cache)
            return 1
        fold_cache = FoldCache.load(paths)
        log.info("loaded RNAfold cache: %d sequences", len(fold_cache))

    accessibility_cache = None
    if args.accessibility_cache is not None:
        paths = shard_paths(args.accessibility_cache)
        if not paths:
            log.error("no RNAplfold cache found at %s", args.accessibility_cache)
            return 1
        accessibility_cache = AccessibilityCache.load(paths)
        log.info(
            "loaded RNAplfold cache: %d sequences, params %s, NA policy %r",
            len(accessibility_cache),
            accessibility_cache.params_tag,
            accessibility_cache.na_policy,
        )

    if args.config == "C" and accessibility_cache is None:
        log.error(
            "config C needs --accessibility-cache. Build one with:\n"
            "  python scripts/preprocess_structures.py --tool rnaplfold "
            "--out cache/accessibility.npz"
        )
        return 1

    settings = TrainSettings(
        config=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss=args.loss,
        weight_by_depth=args.weight_by_depth,
        val_fraction=args.val_fraction,
        seed=args.seed,
        patience=args.patience,
        min_delta=args.min_delta,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        device=args.device,
        max_train_batches=args.max_train_batches,
        proj_dim=args.proj_dim,
        hidden_size=args.hidden_size,
        head_dim=args.head_dim,
        dropout=args.dropout,
        pooling=args.pooling,
        num_layers=args.num_layers,
    )

    summary = train(
        dataset,
        settings,
        out_dir,
        fold_cache=fold_cache,
        accessibility_cache=accessibility_cache,
        use_npz_structure=not args.no_npz_structure,
        logger=log,
    )
    log.info(
        "done: best epoch %s, best val loss %s, checkpoints in %s",
        summary["best_epoch"], summary["best_val_loss"], out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
