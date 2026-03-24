#!/bin/bash
#SBATCH --job-name=fit_aprf_weighted_cv
#SBATCH --output=/home/gdehol/logs/fit_aprf_weighted_cv_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Leave-one-run-out CV for the weighted abstract pRF (log-Gaussian basis set).
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=pil02 fit_aprf_weighted_cv.sh
#
# Optional overrides (--export key=value):
#   SESSION         space-separated session numbers (default: all sessions)
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep)
#   N_BASIS         number of log-Gaussian basis pRFs (default: 8)
#   SMOOTHED        set to "1" to use smoothed betas (default: off)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

SESSION="${SESSION:-}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
N_BASIS="${N_BASIS:-8}"
SMOOTHED="${SMOOTHED:-0}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
    --n-basis "$N_BASIS"
)

[ -n "$SESSION" ] && ARGS+=(--sessions $SESSION)
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

echo "fit_aprf_weighted_cv: sub-${PARTICIPANT_LABEL}  deriv=${FMRIPREP_DERIV}  n_basis=${N_BASIS}  smoothed=${SMOOTHED}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/fit_aprf_weighted_cv.py" \
    "${ARGS[@]}"
