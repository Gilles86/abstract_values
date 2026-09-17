#!/bin/bash
#SBATCH --job-name=cf_peaks
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/cf_peaks_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00

# Per-voxel most-connected V1 orientation (peaks.py).
#
# Usage:
#   sbatch --array=3-30 peaks.sh
#   sbatch --export=ALL,PARTICIPANT_LABEL=pil01 peaks.sh
#
# Optional overrides: N_BASIS, KAPPA, LAGS, REPO (default ~/git/abstract_values)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi
REPO="${REPO:-$HOME/git/abstract_values}"

. $HOME/init_conda.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
conda run --no-capture-output -n abstract_values python -u -W ignore \
    -m abstract_values.connective_fields.peaks "$PARTICIPANT_LABEL" \
    --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue \
    --n-basis "${N_BASIS:-24}" --kappa "${KAPPA:-16}" --lags "${LAGS:-0}"
