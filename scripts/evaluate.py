#!/usr/bin/env python3
"""Evaluate trained checkpoints on the held-out PNAS test split.

The same procedure, metric set and CSV layout are used for every
configuration, so the three models are directly comparable. Passing several
checkpoints also writes an across-seed summary, because a single run does not
establish an improvement.

Examples
--------
One checkpoint::

    python scripts/evaluate.py --checkpoint outputs/A_seed0/best.pt \
        --out-dir results/A_seed0

Several seeds of one configuration, with an aggregate::

    python scripts/evaluate.py --checkpoint outputs/A_seed*/best.pt \
        --out-dir results/A --aggregate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from structsplice.cache import AccessibilityCache, FoldCache, shard_paths
from structsplice.config import DEFAULT_DATA_DIR, FEATURE_CONFIGS
from structsplice.data import load_npz, overlapping_groups
from structsplice.evaluate import aggregate_runs, evaluate_checkpoint
from structsplice.losses import LOSS_NAMES
from structsplice.metrics import format_metrics_line
from structsplice.runtime import setup_logging, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", nargs="+", required=True, type=Path,
                   help="one or more checkpoint files")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="directory for metrics.json and predictions.csv")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--split", default="test", choices=("test", "train"),
                   help="which split to evaluate; defaults to the held-out test split")
    p.add_argument("--config", default=None, choices=sorted(FEATURE_CONFIGS),
                   help="override the config recorded in the checkpoint")
    p.add_argument("--fold-cache", type=Path, default=None)
    p.add_argument("--accessibility-cache", type=Path, default=None)
    p.add_argument("--no-npz-structure", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--loss", choices=LOSS_NAMES, default="bce")
    p.add_argument("--aggregate", action="store_true",
                   help="also write an across-checkpoint mean/std summary")
    p.add_argument("--report-overlap", action="store_true",
                   help="log exon groups shared between train and test")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_root / "evaluate.log")

    dataset = load_npz(
        Path(args.data_dir) / f"{args.split}_data.npz", name=args.split
    )
    log.info("%s split: %s", args.split, dataset.summary())

    if args.report_overlap and args.split == "test":
        train_split = load_npz(
            Path(args.data_dir) / "train_data.npz", name="train"
        )
        shared = overlapping_groups(train_split, dataset)
        log.warning(
            "%d exon group(s) appear in both the PNAS train and test splits. "
            "The split is preserved as published and not modified here.",
            len(shared),
        )

    fold_cache = None
    if args.fold_cache is not None:
        paths = shard_paths(args.fold_cache)
        if not paths:
            log.error("no RNAfold cache at %s", args.fold_cache)
            return 1
        fold_cache = FoldCache.load(paths)

    accessibility_cache = None
    if args.accessibility_cache is not None:
        paths = shard_paths(args.accessibility_cache)
        if not paths:
            log.error("no RNAplfold cache at %s", args.accessibility_cache)
            return 1
        accessibility_cache = AccessibilityCache.load(paths)

    payloads = []
    multiple = len(args.checkpoint) > 1
    for checkpoint in args.checkpoint:
        if not Path(checkpoint).is_file():
            log.error("checkpoint not found: %s", checkpoint)
            return 1
        sub = out_root / Path(checkpoint).parent.name if multiple else out_root
        payload = evaluate_checkpoint(
            checkpoint,
            dataset,
            sub,
            config=args.config,
            fold_cache=fold_cache,
            accessibility_cache=accessibility_cache,
            use_npz_structure=not args.no_npz_structure,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            loss_name=args.loss,
            split_name=args.split,
            logger=log,
        )
        payloads.append(payload)
        log.info(
            "config %s seed %s: %s",
            payload["config"], payload["seed"],
            format_metrics_line(payload["metrics"]),
        )

    if args.aggregate or multiple:
        summary = aggregate_runs(payloads)
        write_json(out_root / "aggregate.json", summary)
        log.info("aggregate over %d run(s), configs %s:",
                 summary["n_runs"], summary["configs"])
        for key, stats in summary["metrics"].items():
            log.info("  %-11s mean=%.5f std=%.5f", key,
                     stats["mean"], stats["std"])
        if summary["n_runs"] < 2:
            log.warning(
                "only one run: this is not evidence of an improvement. "
                "Train several seeds before comparing configurations."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
