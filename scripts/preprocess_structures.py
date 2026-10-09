#!/usr/bin/env python3
"""Build RNAfold / RNAplfold feature caches for the PNAS splits.

Caches are keyed by a hash of the folded sequence, so duplicate sequences are
folded once and the train and test splits can safely share one cache file:
nothing but the nucleotide string is read, and no PSI value or split label
enters preprocessing. That is why a shared cache is not leakage.

Examples
--------
Dot-bracket structures for both splits::

    python scripts/preprocess_structures.py --tool rnafold \
        --splits train test --out cache/rnafold.npz

Accessibility, sharded for a SLURM array job::

    python scripts/preprocess_structures.py --tool rnaplfold \
        --splits train test --out cache/accessibility.npz \
        --shard "$SLURM_ARRAY_TASK_ID" --n-shards 20

    python scripts/preprocess_structures.py --tool rnaplfold \
        --out cache/accessibility.npz --merge-shards
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from structsplice.cache import (
    AccessibilityCache,
    FoldCache,
    shard_path_for,
    shard_paths,
    unique_sequences,
)
from structsplice.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_NA_POLICY,
    NA_POLICIES,
    N_ACCESSIBILITY_CHANNELS,
    RNAfoldParams,
    RNAplfoldParams,
)
from structsplice.data import load_npz
from structsplice.runtime import setup_logging
from structsplice.vienna import (
    INSTALL_HINT,
    ViennaRNANotFound,
    iter_batches,
    require_binary,
    run_rnafold,
    run_rnaplfold,
    vienna_version,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tool",
        choices=("rnafold", "rnaplfold"),
        required=True,
        help="which ViennaRNA tool to run",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory holding train_data.npz / test_data.npz",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="which splits to collect sequences from",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output cache path (default: <cache-dir>/<tool>.npz)",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help="default parent directory for the cache",
    )
    p.add_argument("--limit", type=int, default=0,
                   help="fold only the first N unique sequences (debugging)")
    p.add_argument("--batch-size", type=int, default=500,
                   help="sequences per ViennaRNA subprocess call")
    p.add_argument("--shard", type=int, default=-1,
                   help="zero-based shard index for array jobs")
    p.add_argument("--n-shards", type=int, default=1,
                   help="total number of shards")
    p.add_argument("--merge-shards", action="store_true",
                   help="merge existing shards into --out and exit")
    p.add_argument("--na-policy", choices=NA_POLICIES,
                   default=DEFAULT_NA_POLICY,
                   help="how to fill RNAplfold boundary NA entries")
    p.add_argument("--winsize", type=int, default=RNAplfoldParams.winsize)
    p.add_argument("--span", type=int, default=RNAplfoldParams.span)
    p.add_argument("--ulength", type=int, default=RNAplfoldParams.ulength)
    p.add_argument("--temperature", type=float, default=37.0)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--force", action="store_true",
                   help="refold sequences already present in the cache")
    return p.parse_args(argv)


def collect_sequences(data_dir: Path, splits: list[str], log) -> list[str]:
    """Return the unique RNA-normalised model sequences across the splits."""
    all_sequences: list[str] = []
    for split in splits:
        path = Path(data_dir) / f"{split}_data.npz"
        dataset = load_npz(path, name=split)
        log.info("%s: %s", split, dataset.summary())
        all_sequences.extend(dataset.model_sequence.tolist())
    unique = unique_sequences(all_sequences)
    log.info(
        "%d rows across %s -> %d unique sequences (%.1f%% duplicates)",
        len(all_sequences),
        "+".join(splits),
        len(unique),
        100.0 * (1.0 - len(unique) / max(len(all_sequences), 1)),
    )
    return unique


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    log = setup_logging(args.log_file)

    out = args.out or (Path(args.cache_dir) / f"{args.tool}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    cache_cls = FoldCache if args.tool == "rnafold" else AccessibilityCache

    if args.merge_shards:
        found = shard_paths(out)
        if not found:
            log.error("no shards found for %s", out)
            return 1
        merged = cache_cls.load(found)
        merged.save(out)
        log.info("merged %d shard(s) -> %s (%d sequences)",
                 len(found), out, len(merged))
        return 0

    try:
        binary = require_binary(
            "RNAfold" if args.tool == "rnafold" else "RNAplfold"
        )
    except ViennaRNANotFound:
        log.error("ViennaRNA not found on PATH.\n%s", INSTALL_HINT)
        return 2
    log.info("using %s (ViennaRNA %s)", binary, vienna_version() or "unknown")

    sequences = collect_sequences(args.data_dir, args.splits, log)
    if args.limit:
        sequences = sequences[: args.limit]
        log.info("limited to %d sequences", len(sequences))

    target = out
    if args.shard >= 0:
        if not 0 <= args.shard < args.n_shards:
            log.error("shard %d outside [0, %d)", args.shard, args.n_shards)
            return 1
        sequences = sequences[args.shard :: args.n_shards]
        target = shard_path_for(out, args.shard)
        log.info("shard %d/%d: %d sequences -> %s",
                 args.shard, args.n_shards, len(sequences), target)

    if args.tool == "rnafold":
        params = RNAfoldParams(temperature=args.temperature, binary=binary)
        cache = FoldCache(params_tag=params.cache_tag())
    else:
        params = RNAplfoldParams(
            winsize=args.winsize,
            span=args.span,
            ulength=args.ulength,
            temperature=args.temperature,
            binary=binary,
        )
        cache = AccessibilityCache(
            params_tag=params.cache_tag(),
            na_policy=args.na_policy,
            n_channels=N_ACCESSIBILITY_CHANNELS,
        )
        if args.ulength < N_ACCESSIBILITY_CHANNELS:
            log.error("--ulength %d is below the %d channels the model needs",
                      args.ulength, N_ACCESSIBILITY_CHANNELS)
            return 1

    if target.is_file() and not args.force:
        existing = cache_cls.load(target)
        todo = existing.missing(sequences)
        log.info("%s already holds %d sequences; %d still to fold",
                 target, len(existing), len(todo))
        cache = existing
        sequences = todo

    if not sequences:
        log.info("nothing to do; cache is complete")
        cache.save(target)
        return 0

    done = 0
    for batch in iter_batches(sequences, args.batch_size):
        if args.tool == "rnafold":
            structures, mfes = run_rnafold(batch, params)
            for seq, structure, mfe in zip(batch, structures, mfes):
                cache.add(seq, structure, float(mfe))
        else:
            blocks, masks = run_rnaplfold(
                batch, params, na_policy=args.na_policy,
                n_channels=N_ACCESSIBILITY_CHANNELS,
            )
            for i, seq in enumerate(batch):
                cache.add(seq, blocks[i], masks[i])
        done += len(batch)
        log.info("folded %d/%d", done, len(sequences))

    cache.save(target)
    log.info("wrote %s (%d sequences, params %s)",
             target, len(cache), cache.params_tag)
    if args.tool == "rnaplfold":
        any_key = next(iter(cache.values))
        block = cache.values[any_key]
        mask = cache.na_masks[any_key]
        log.info(
            "sanity: block shape %s, %d NA entries filled by policy %r",
            block.shape, int(mask.sum()), args.na_policy,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
