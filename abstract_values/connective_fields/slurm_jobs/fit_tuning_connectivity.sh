#!/bin/bash
#SBATCH --job-name=tuncon
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/tuncon_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

# Coding / connectivity / joint models of IPS voxels with linearly inverted V1
# input (fit_tuning_connectivity.py).
#   sbatch --export=ALL,PARTICIPANT_LABEL=05,SMOOTHED=1 fit_tuning_connectivity.sh

set -eo pipefail
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
REPO="${REPO:-$HOME/git/abstract_values}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
ARGS=("$PARTICIPANT_LABEL" --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue)
[ "${SMOOTHED:-0}" = "1" ] && ARGS+=(--smoothed)
exec $HOME/data/conda/envs/abstract_values/bin/python -u -W ignore -m \
    abstract_values.connective_fields.fit_tuning_connectivity "${ARGS[@]}"
