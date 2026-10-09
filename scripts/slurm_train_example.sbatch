#!/bin/bash
#
# EXAMPLE SLURM script. Every value marked <<< PLACEHOLDER >>> must be set to
# match your own NYU HPC allocation before this will run. Nothing here has
# been tested against a live scheduler.
#
#SBATCH --job-name=structsplice
#SBATCH --account=<<< PLACEHOLDER: your Slurm account >>>
#SBATCH --partition=<<< PLACEHOLDER: your GPU partition >>>
#SBATCH --gres=gpu:1                  # <<< PLACEHOLDER: GPU type/count >>>
#SBATCH --cpus-per-task=4             # <<< PLACEHOLDER >>>
#SBATCH --mem=32G                     # <<< PLACEHOLDER >>>
#SBATCH --time=04:00:00               # <<< PLACEHOLDER >>>
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-2                   # one task per feature configuration
#
# Submit from a login node with:   sbatch scripts/slurm_train_example.sbatch
# Do not run training or installs on a login node directly.

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────
# Replace with however you load Python and ViennaRNA on your cluster, e.g.
# a module load, a Singularity container, or a conda environment.
#
#   module purge
#   module load <<< PLACEHOLDER: python module >>>
#   module load <<< PLACEHOLDER: viennarna module >>>
#
# Or activate a prebuilt virtualenv (built earlier on a compute node, not
# on the login node):
#
#   source <<< PLACEHOLDER: /path/to/venv >>>/bin/activate

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"

# All paths are configurable; nothing is hard-coded to a local machine.
export STRUCTSPLICE_DATA_DIR="${STRUCTSPLICE_DATA_DIR:-$PROJECT_DIR/data}"
export STRUCTSPLICE_CACHE_DIR="${STRUCTSPLICE_CACHE_DIR:-$PROJECT_DIR/cache}"
export STRUCTSPLICE_OUTPUT_DIR="${STRUCTSPLICE_OUTPUT_DIR:-$PROJECT_DIR/outputs}"

mkdir -p logs "$STRUCTSPLICE_OUTPUT_DIR"

# ── Pick the configuration for this array task ─────────────────────────────
CONFIGS=(A B C)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEED:-0}"
OUT_DIR="$STRUCTSPLICE_OUTPUT_DIR/${CONFIG}_seed${SEED}"

echo "host=$(hostname) config=$CONFIG seed=$SEED out=$OUT_DIR"
nvidia-smi || true

# Config C needs the accessibility cache; build it once beforehand with
# scripts/preprocess_structures.py (see README), not inside this job.
EXTRA=()
if [[ "$CONFIG" == "C" ]]; then
  EXTRA+=(--accessibility-cache "$STRUCTSPLICE_CACHE_DIR/accessibility.npz")
fi

srun python scripts/train.py \
  --config "$CONFIG" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR" \
  --epochs 30 \
  --batch-size 128 \
  --lr 0.001 \
  --patience 5 \
  --num-workers "${SLURM_CPUS_PER_TASK:-0}" \
  --device cuda \
  "${EXTRA[@]}"

srun python scripts/evaluate.py \
  --checkpoint "$OUT_DIR/best.pt" \
  --out-dir "$PROJECT_DIR/results/${CONFIG}_seed${SEED}" \
  --device cuda \
  "${EXTRA[@]}"
