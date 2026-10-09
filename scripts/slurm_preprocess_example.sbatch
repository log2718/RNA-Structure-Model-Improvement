#!/bin/bash
#
# EXAMPLE SLURM array job that builds the RNAplfold accessibility cache.
# Every <<< PLACEHOLDER >>> must be set for your own allocation. CPU only:
# ViennaRNA does not use the GPU.
#
#SBATCH --job-name=structsplice-plfold
#SBATCH --account=<<< PLACEHOLDER: your Slurm account >>>
#SBATCH --partition=<<< PLACEHOLDER: your CPU partition >>>
#SBATCH --cpus-per-task=1             # <<< PLACEHOLDER >>>
#SBATCH --mem=8G                      # <<< PLACEHOLDER >>>
#SBATCH --time=02:00:00               # <<< PLACEHOLDER >>>
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-19                  # <<< PLACEHOLDER: shard count >>>

set -euo pipefail

#   module load <<< PLACEHOLDER: python module >>>
#   module load <<< PLACEHOLDER: viennarna module >>>

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
export STRUCTSPLICE_CACHE_DIR="${STRUCTSPLICE_CACHE_DIR:-$PROJECT_DIR/cache}"
mkdir -p logs "$STRUCTSPLICE_CACHE_DIR"

N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-20}"

srun python scripts/preprocess_structures.py \
  --tool rnaplfold \
  --splits train test \
  --out "$STRUCTSPLICE_CACHE_DIR/accessibility.npz" \
  --shard "$SLURM_ARRAY_TASK_ID" \
  --n-shards "$N_SHARDS"

# After the whole array finishes, merge the shards once:
#   python scripts/preprocess_structures.py --tool rnaplfold \
#       --out cache/accessibility.npz --merge-shards
