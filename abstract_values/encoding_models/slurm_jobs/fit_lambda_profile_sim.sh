#!/bin/bash
#SBATCH --job-name=lamprof_sim
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/lamprof_sim_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

# cvR^2 lambda profile on betas simulated with a known lambda
# (fit_lambda_profile.py --sim-lam, simulate_lambda_pipeline.simulated_betas).
# Array: 6 conditions x N participants; task t -> participant (t-1) % N,
# condition (t-1) / N: lambda {0, 1} x noise {1, 2, 3}.
#   sbatch --array=1-210 --exclude=u24-chaihm0-651 fit_lambda_profile_sim.sh
set -eo pipefail
export PYTHONUNBUFFERED=1
export KERAS_BACKEND=tensorflow
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO="${REPO:-$HOME/git/abstract_values}"
cd "$REPO"
mapfile -t SUBS < <(ls "$BIDS_FOLDER/derivatives/encoding_models/aprf-session-shift" \
                    | grep '^sub-' | sed 's/^sub-//' | sort)
N=${#SUBS[@]}
t=$(( SLURM_ARRAY_TASK_ID - 1 ))
SUB=${SUBS[$(( t % N ))]}
c=$(( t / N ))
LAMS=(0 0 0 1 1 1); NOISE=(1 2 3 1 2 3)
exec $HOME/data/conda/envs/abstract_values/bin/python -u -W ignore -m \
    abstract_values.encoding_models.fit_lambda_profile "$SUB" --bids-folder "$BIDS_FOLDER" \
    --sim-lam "${LAMS[$c]}" --noise-scale "${NOISE[$c]}"
