#!/bin/bash
#SBATCH --job-name=fit_null_model
#SBATCH --output=/home/gdehol/logs/fit_null_model_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=zne.uzh

# Fit null (mean) encoding model and compute CV R².
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=03 fit_null_model.sh
#   sbatch --array=1-30 fit_null_model.sh
#
# Optional exports:
#   SESSION       session number (default: all)
#   SMOOTHED      set to "1" for smoothed betas
#   FMRIPREP_DERIV  (default: fmriprep)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi

SESSION="${SESSION:-}"
SMOOTHED="${SMOOTHED:-0}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
)

[ -n "$SESSION" ] && ARGS+=(--sessions "$SESSION")
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

export PYTHONUNBUFFERED=1

. $HOME/init_conda.sh
conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/fit_null_model.py" \
    "${ARGS[@]}"
