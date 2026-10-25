#!/bin/bash
#SBATCH --job-name=sim_lambda
#SBATCH --output=/home/gdehol/logs/sim_lambda_%A_%a.txt
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# End-to-end lambda recovery: simulate betas with a known lambda from each
# voxel's real session-shift fit, refit (simulate_lambda_pipeline.py).
# Array: 4 conditions x N participants; task t -> participant (t-1) % N,
# condition (t-1) / N in: lambda 0 all runs, lambda 0 odd runs,
# lambda 1 all runs, lambda 1 odd runs; 4-7: noise x2 / x3 for lambda 0, 1 (all runs).
#   sbatch --array=1-140 --exclude=u24-chaihm0-651 simulate_lambda_pipeline.sh
set -eo pipefail

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values
export KERAS_BACKEND=tensorflow
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"

mapfile -t SUBS < <(ls "$BIDS_FOLDER/derivatives/encoding_models/aprf-session-shift" \
                    | grep '^sub-' | sed 's/^sub-//' | sort)
N=${#SUBS[@]}
t=$(( SLURM_ARRAY_TASK_ID - 1 ))
SUB=${SUBS[$(( t % N ))]}
case $(( t / N )) in
    0) ARGS=(--lam 0) ;;
    1) ARGS=(--lam 0 --runs odd) ;;
    2) ARGS=(--lam 1) ;;
    3) ARGS=(--lam 1 --runs odd) ;;
    4) ARGS=(--lam 0 --noise-scale 2) ;;
    5) ARGS=(--lam 0 --noise-scale 3) ;;
    6) ARGS=(--lam 1 --noise-scale 2) ;;
    7) ARGS=(--lam 1 --noise-scale 3) ;;
esac

exec "$HOME/data/conda/envs/abstract_values/bin/python" -u \
    -m abstract_values.encoding_models.simulate_lambda_pipeline "$SUB" \
    --bids-folder "$BIDS_FOLDER" "${ARGS[@]}"
