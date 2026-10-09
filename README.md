# RNA-Structure-Model-Improvement

Does RNA secondary structure help predict exon inclusion? This repository
compares three BiLSTM models on the splicing library from Liao, Sudarshan and
Regev, *Deciphering RNA splicing logic with interpretable machine learning*
(PNAS, 2023). The three models differ **only** in their input features:
sequence alone, sequence plus RNAfold dot-bracket structure, and sequence plus
RNAplfold accessibility.

The code is complete and tested. No full training run has been performed; that
is intended for an NYU HPC GPU allocation.

## Research objective

Predict one continuous PSI (percent spliced in) value per sequence, as a
regression onto `[0, 1]`. This is a sequence-level target, not nucleotide-level
splice-site classification. The question is whether explicit structure features
add predictive signal over raw sequence, which a BiLSTM could in principle
learn implicitly.

## Dataset

Both files were supplied with the repository and are **used exactly as
published**. Nothing in this codebase rewrites, re-splits or regenerates them.

| Split | Rows | File |
|---|---|---|
| Train | 191,846 | `data/train_data.npz` |
| Test | 47,962 | `data/test_data.npz` |

Characteristics confirmed by inspection:

- **Input sequence** is `model_sequence`, a fixed **90 nt** string: a constant
  10 nt left flank `CATCCAGGTT`, the variable **70 nt exon**, and a constant
  10 nt right flank `CAGGTCTGAC`. Every row in both splits follows this layout.
  The flanks are the vector context of the PNAS minigene reporter, so they are
  short constant padding rather than real flanking introns.
- **Target** is `metadata_PSI`, a float in `[0, 1]`. Saturated labels are
  common: the training split has 29,736 rows at exactly 0 and 12,131 at exactly
  1, that is 21.8% of rows sitting on a boundary (test: 7,556 and 3,004, or
  22.0%). The loss must tolerate targets of 0 and 1 without taking `log 0`.
- **Read counts are present.** `metadata_PSI` equals
  `metadata_num_exon_inclusion / (metadata_num_exon_inclusion +
  metadata_num_exon_skipping)` exactly, for every row. Depth ranges from 60 to
  1,399 reads with a median of 159 in training, and 60 to 1,321 with a median
  of 160 in test. See *Loss function* for what this implies.
- **Variants** are distinct exons, not annotated mutation families. There is no
  wild-type/mutant grouping column. 23 `model_sequence` values appear twice in
  the training split, which is why the train/validation split is grouped.
- **The files already ship RNAfold output.** The `structure` field holds the
  MFE dot-bracket string and matches a fresh `RNAfold -T 37` call on all 40
  sequences tested, so Model B needs no preprocessing by default. The `struct_oh`
  field matches this project's dot-bracket channel order, verified in the tests.
- No accessibility features ship with the data, so Model C requires RNAplfold.

### Known dataset caveats

- **13 `model_sequence` values appear in both the published train and test
  splits.** This is a property of the data as released. The split is preserved
  as published; `scripts/evaluate.py --report-overlap` reports the overlap
  rather than repairing it. Any eventual write-up should mention it.
- The orientation of `model_sequence` is taken as given, 5' to 3' as stored. The
  shipped `structure` field folds consistently with that reading, which is
  corroborating but not proof.
- Sequences are stored in the DNA alphabet (`T`). `T` and `U` are treated as the
  same symbol and share one channel.

### Train/validation/test discipline

The PNAS test split is used **only** for final evaluation. Validation comes
from a grouped split of the *training* rows, carved out by exon so the 23
duplicated sequences cannot straddle the boundary. The split is a deterministic
function of the seed and the sorted group keys, so models A, B and C see
byte-identical training and validation rows. Default is 10% of groups.

## The three models

| Model | Features | Channels |
|---|---|---|
| A | one-hot sequence (A, C, G, U) | 4 |
| B | A + RNAfold dot-bracket (`.`, `(`, `)`) | 7 |
| C | A + RNAplfold accessibility, stretch lengths l = 1..8 | 12 |

All three use one shared class, `BiLSTMPSIRegressor`, whose `input_dim` is the
only thing that changes.

Model C's eight channels are the probability that an unpaired stretch of length
l **ends** at that position, for l = 1 through 8. Channel 0 alone is the
single-nucleotide unpaired probability; it is not the whole feature. RNAplfold
is run with the parameters from the MLCB abstract:

```
RNAplfold -W 200 -L 150 -u 31 -T 37
```

`-u 31` is kept as specified and the first eight columns are handed to the
model, so the cache stays faithful to the abstract.

## Architecture

```
input (L, C)  ->  Linear(C, 32) + ReLU + Dropout
              ->  BiLSTM(32 -> 64 per direction, 1 layer)
              ->  masked pooling over positions  ->  (128,)
              ->  Dropout -> Linear(128, 64) -> ReLU -> Dropout -> Linear(64, 1)
              ->  logit;  PSI = sigmoid(logit)
```

Defaults: dropout 0.2, masked mean pooling, Adam at 1e-3, batch size 128. These
are starting points, not tuned optima.

`forward` returns a **logit**, and PSI is `sigmoid(logit)`, which keeps the
prediction in `(0, 1)` by construction and lets the loss use the fused, stable
`BCEWithLogitsLoss`. Use `predict_psi` for bounded predictions.

### Fairness of the comparison

Points that could otherwise bias an A/B/C comparison, and how they are handled:

- **Only the input projection differs.** Parameter counts are 58,657 (A),
  58,753 (B) and 58,913 (C). Every difference is exactly `(C - 4) x 32` weights
  in the first `Linear`. The LSTM, pooling and head are identical.
- **Identical data order.** The train/validation split and the DataLoader
  shuffling depend on the seed only, not on the feature configuration.
- **Identical schedule.** Same optimizer, learning rate, batch size, epoch cap,
  early-stopping rule and loss for all three.
- **Extra channels are not free capacity.** Models B and C get 3 and 8 more
  input channels, so they have marginally more parameters. The projection is
  the same width (32) in all three, which keeps the bottleneck equal.
- **Unnormalised accessibility.** The eight accessibility channels are
  probabilities in `[0, 1]`, on the same scale as the one-hot channels, so no
  feature scaling is applied that A does not also get.
- A single run proves nothing. Use several seeds and compare distributions;
  `scripts/evaluate.py --aggregate` reports mean and standard deviation.

## Loss function

Default is binary cross-entropy on the fractional target, via
`BCEWithLogitsLoss`. The PNAS work used KL divergence, and for Bernoulli
distributions the two differ only by a constant that depends on the target:

```
KL(y || p) = y*log(y/p) + (1-y)*log((1-y)/(1-p))
           = BCE(y, p) - H(y)
```

where `H(y) = -[y*log y + (1-y)*log(1-y)]` is the entropy of the target. `H(y)`
contains no model parameter, so:

- the gradients with respect to the weights are **identical**, making the two
  the same optimisation problem with the same optimum. The test suite asserts
  this.
- the reported numbers differ. KL is non-negative and hits 0 at a perfect fit;
  BCE bottoms out at `H(y) > 0`. Early stopping is unaffected, since a constant
  offset cannot change which epoch is best, but a logged BCE is not comparable
  to a published KL. Pass `--loss kl` to get the PNAS convention.

### On read-depth weighting

This corpus does carry inclusion and skipping counts, and PSI is exactly their
ratio. A PSI of 0.5 from 60 reads and from 1,399 reads carry very different
evidence, and unweighted BCE treats them identically. Weighting each example by
its depth makes the objective the exact Binomial negative log-likelihood, up to
a binomial coefficient that is constant in the parameters. That is the
statistically correct likelihood for this measurement process.

It is **off by default** on purpose: the published PNAS objective is
unweighted, so leaving it off keeps the baseline comparable and keeps the A/B/C
contrast about features rather than about a change of objective. Turning it on
changes the estimand, so treat it as a separate, separately reported
experiment: `--weight-by-depth`.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

ViennaRNA supplies `RNAfold` and `RNAplfold` and is **not** a pip dependency:

```bash
conda install -c bioconda viennarna      # conda
brew install viennarna                   # macOS
sudo apt-get install vienna-rna          # Debian/Ubuntu
module load viennarna                    # typical HPC
```

Verify with `RNAfold --version`. Model A needs nothing; Model B falls back to
the structures already in the npz files; **only Model C requires ViennaRNA.**

All paths are configurable by flag or environment variable, with no hard-coded
local paths:

| Variable | Default | Purpose |
|---|---|---|
| `STRUCTSPLICE_DATA_DIR` | `./data` | input npz files |
| `STRUCTSPLICE_CACHE_DIR` | `./cache` | structure caches |
| `STRUCTSPLICE_OUTPUT_DIR` | `./outputs` | checkpoints and logs |

## Preprocessing structures

Caches are keyed by a SHA-256 hash of the folded sequence, never by row index,
so duplicate sequences are folded once and train and test can share one file.
Nothing but the nucleotide string is read during preprocessing: no PSI value and
no split label, which is why a shared cache is not leakage.

```bash
# Accessibility for Model C. Both splits, one shared cache. ~30 min, 1 core.
python scripts/preprocess_structures.py --tool rnaplfold \
    --splits train test --out cache/accessibility.npz

# Optional: refold dot-brackets instead of trusting the npz field. ~11 min.
python scripts/preprocess_structures.py --tool rnafold \
    --splits train test --out cache/rnafold.npz
```

Shard it across a SLURM array if preferred, then merge once:

```bash
python scripts/preprocess_structures.py --tool rnaplfold \
    --out cache/accessibility.npz --shard "$SLURM_ARRAY_TASK_ID" --n-shards 20
python scripts/preprocess_structures.py --tool rnaplfold \
    --out cache/accessibility.npz --merge-shards
```

### Boundaries, missing values and padding

- **Structures are folded on the full 90 nt `model_sequence`**, flanks
  included, not on the 70 nt exon alone. Folding the exon in isolation would
  give a different structure. Any future mutated sequence is hashed in full, so
  a variant can never reuse a neighbour's structure.
- **RNAplfold emits `NA`** where a stretch of length l would run off the 5' end,
  that is whenever `l > i` for 1-based position `i`. That is 28 entries per
  sequence for eight channels. These are *undefined*, not measured zeros.
  The parser records them in an explicit boolean mask stored alongside the
  values, so a filled boundary is always distinguishable from a real 0.0.
  Two policies are available: `zero` (default) substitutes 0.0, and `edge`
  carries the longest defined length at that position, injecting no artificial
  zeros.
- **RNAplfold caps its column count at the sequence length**, so `-u 31` yields
  31 columns for a 90 nt sequence but only 24 for a 24 nt one. The parser
  expects `min(ulength, len(seq))` and marks longer stretches undefined.
- **Sequences are a uniform 90 nt**, so padding is currently a no-op. The
  collate function still pads to the longest sequence per batch and emits a
  mask, and the LSTM uses packed sequences, so longer or ragged context would
  work unchanged. The tests exercise the ragged path directly.
- The code never confuses dot-bracket structure with base-pairing probabilities
  (`RNAfold -p` is not used), nor single-nucleotide accessibility with the
  eight-channel profile.

## Running the tests

```bash
python -m pytest                      # everything
python -m pytest -m "not vienna"      # skip tests needing RNAfold/RNAplfold
python -m pytest tests/test_vienna.py # structure parsing and real folding
```

Unit tests use synthetic fixtures and run without ViennaRNA. Tests that need
the real binaries are marked `vienna` and skip, rather than silently pass, when
they are unavailable.

## Training

Nothing below has been run to completion; these are the intended commands.

```bash
# Model A: sequence only
python scripts/train.py --config A --seed 0 --out-dir outputs/A_seed0

# Model B: sequence + dot-bracket (uses the structures in the npz files)
python scripts/train.py --config B --seed 0 --out-dir outputs/B_seed0

# Model C: sequence + accessibility (needs the cache)
python scripts/train.py --config C --seed 0 --out-dir outputs/C_seed0 \
    --accessibility-cache cache/accessibility.npz
```

Useful flags: `--epochs`, `--batch-size`, `--lr`, `--patience`, `--dropout`,
`--pooling {mean,max,last}`, `--hidden-size`, `--loss {bce,kl}`,
`--weight-by-depth`, `--device {cpu,cuda,mps}`, `--num-workers`,
`--val-fraction`, `--seed`. Add `--limit-rows N` for a quick smoke test on a
subsample; it is not for real runs.

Each run writes `best.pt`, `last.pt`, `history.jsonl` with per-epoch train and
validation loss, `train_summary.json`, and `train.log`.

## Evaluation

```bash
# One checkpoint on the held-out PNAS test split
python scripts/evaluate.py --checkpoint outputs/A_seed0/best.pt \
    --out-dir results/A_seed0 --report-overlap

# Several seeds, with mean and standard deviation
python scripts/evaluate.py --checkpoint outputs/A_seed*/best.pt \
    --out-dir results/A --aggregate

# Model C needs the same cache it trained with
python scripts/evaluate.py --checkpoint outputs/C_seed0/best.pt \
    --out-dir results/C_seed0 --accessibility-cache cache/accessibility.npz
```

Reports MSE, RMSE, MAE, Pearson r, Pearson r squared, R squared and the
held-out loss. Writes `metrics.json` and `predictions.csv`, the latter carrying
`row_index`, `exon_id`, `model_sequence`, `psi_true` and `psi_pred` so
predictions stay traceable to individual exons.

Degenerate cases are reported honestly rather than papered over: a constant
target makes Pearson r and R squared undefined, so they are written as `null`
with a companion note explaining why, never as 0.

## HPC

`scripts/slurm_train_example.sbatch` and
`scripts/slurm_preprocess_example.sbatch` are **examples**. Every account,
partition, GPU type, memory and time value is marked `<<< PLACEHOLDER >>>` and
must be set for your own allocation. Neither has been run against a live
scheduler. Build the accessibility cache before submitting Model C, and do not
run training or installs on a login node.

## Project layout

```
structsplice/
  config.py     paths, channel orders, ViennaRNA parameter sets
  data.py       npz loading, validation, grouped train/val split
  encoding.py   one-hot sequence, dot-bracket, accessibility assembly
  vienna.py     RNAfold/RNAplfold subprocess calls and _lunp parsing
  cache.py      on-disk structure caches, sharding and merging
  dataset.py    torch Dataset and padding-aware collation
  model.py      BiLSTMPSIRegressor, shared by all three configurations
  losses.py     BCE and Bernoulli KL on fractional targets, depth weights
  metrics.py    regression metrics with explicit degenerate-case handling
  runtime.py    seeding, device resolution, checkpoints, JSON output
  train.py      training loop, early stopping, epoch logging
  evaluate.py   held-out evaluation, prediction export, seed aggregation
scripts/        preprocess_structures.py, train.py, evaluate.py, SLURM examples
tests/          193 tests
```

## Limitations and assumptions

1. **No training has been run.** Every number in this README is a dataset fact,
   a parameter count or a timing measurement. There are no accuracy results,
   and the hyperparameters are untuned starting points.
2. **13 sequences appear in both published splits.** Preserved as released;
   reported by `--report-overlap`.
3. **The 10 nt flanks are vector context, not introns.** The model sees no real
   flanking intron and no splice-site motif beyond what sits inside the 90 nt
   window.
4. **No mutation grouping exists in the data.** Grouping uses the exon sequence,
   which is the only identifier available. If a variant-family column exists
   elsewhere in the PNAS supplement, the grouped split should be revisited.
5. **RNAfold MFE only.** A single minimum-free-energy structure, not an
   ensemble or partition-function representation. Model C's accessibility is
   the ensemble-derived complement to this.
6. **Boundary `NA` handling is a modelling choice.** The `zero` default is
   explicit and masked, but `edge` is also provided; the choice is worth a
   sensitivity check.
7. **`--weight-by-depth` is off by default,** which means the default objective
   ignores the fact that labels have very different read support.
8. **Sequence orientation is assumed as stored** and was not independently
   confirmed against a genome annotation.
9. **The PNAS model itself is not reimplemented.** This is a BiLSTM comparison
   on the PNAS data, not a reproduction of the PNAS architecture, and its
   published metrics are not directly comparable to a BCE number logged here.
