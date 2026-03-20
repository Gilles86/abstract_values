#!/bin/bash
#SBATCH --job-name=fit_aprf_shift_cv
#SBATCH --output=/home/gdehol/logs/fit_aprf_shift_cv_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# Leave-one-run-out CV for the session-shifted abstract pRF
# (SessionShiftedLogGaussianPRF: mode_1, mode_2, fwhm, amplitude, baseline).
# Requires at least 2 sessions.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=pil01,SESSION="1 2" fit_aprf_shift_cv.sh
#
# Optional overrides (--export key=value):
#   SESSION         space-separated session numbers (default: all sessions)
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep)
#   SMOOTHED        set to "1" to use smoothed betas (default: off)
#   N_ITERATIONS    max gradient descent iterations per fold (default: 1000)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

SESSION="${SESSION:-}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
SMOOTHED="${SMOOTHED:-0}"
N_ITERATIONS="${N_ITERATIONS:-1000}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
    --n-iterations "$N_ITERATIONS"
)

[ -n "$SESSION" ] && ARGS+=(--sessions $SESSION)
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

echo "fit_aprf_shift_cv: sub-${PARTICIPANT_LABEL}  deriv=${FMRIPREP_DERIV}  smoothed=${SMOOTHED}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/fit_aprf_shift_cv.py" \
    "${ARGS[@]}"
