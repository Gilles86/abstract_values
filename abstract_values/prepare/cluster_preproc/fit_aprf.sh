#!/bin/bash
#SBATCH --job-name=fit_aprf
#SBATCH --output=/home/gdehol/logs/fit_aprf_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Fit abstract pRF (LogGaussianPRF) encoding model to single-trial GLMsingle
# betas using the objective CHF value of each gabor stimulus as the paradigm.
#
# Two ways to run:
#   Array job:  sbatch --array=1-30 fit_aprf.sh
#   By name:    sbatch --export=PARTICIPANT_LABEL=pil01 fit_aprf.sh
#
# Optional overrides (--export key=value):
#   SESSION       session number, e.g. SESSION=1 (default: all sessions)
#   SMOOTHED      set to "1" to fit on spatially smoothed betas (default: off)
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep)
#   N_ITERATIONS  max gradient descent iterations (default: 1000)
#
# Example (pilot, session 1, smoothed):
#   sbatch --export=PARTICIPANT_LABEL=pil01,SESSION=1,SMOOTHED=1 fit_aprf.sh

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

SESSION="${SESSION:-}"
SMOOTHED="${SMOOTHED:-0}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
N_ITERATIONS="${N_ITERATIONS:-1000}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=/home/gdehol/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
    --n-iterations "$N_ITERATIONS"
)

[ -n "$SESSION" ] && ARGS+=(--sessions "$SESSION")
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

echo "fit_aprf: sub-${PARTICIPANT_LABEL}  deriv=${FMRIPREP_DERIV}  smoothed=${SMOOTHED}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/fit_aprf.py" \
    "${ARGS[@]}"
