#!/bin/bash
#SBATCH --job-name=lamprof
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/lamprof_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

# Fine lambda profile of the mix coding model, bootstrap CI over runs
# (encoding_models/fit_lambda_profile.py).
#   sbatch --export=ALL,PARTICIPANT_LABEL=05,SMOOTHED=1 fit_lambda_profile.sh

set -eo pipefail
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
REPO="${REPO:-$HOME/git/abstract_values}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
ARGS=("$PARTICIPANT_LABEL" --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue)
[ "${SMOOTHED:-0}" = "1" ] && ARGS+=(--smoothed)
[ "${BIDS:-0}" = "1" ] && ARGS+=(--bids)
[ "${WHOLE_BRAIN:-0}" = "1" ] && ARGS+=(--whole-brain)
exec $HOME/data/conda/envs/abstract_values/bin/python -u -W ignore -m \
    abstract_values.encoding_models.fit_lambda_profile "${ARGS[@]}"
