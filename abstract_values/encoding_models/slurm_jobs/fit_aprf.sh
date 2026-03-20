#!/bin/bash
#SBATCH --job-name=fit_aprf
#SBATCH --output=/home/gdehol/logs/fit_aprf_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Fit abstract pRF (LogGaussianPRF) encoding model to single-trial GLMsingle
# betas using the objective CHF value of each gabor stimulus as the paradigm.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=pil02 fit_aprf.sh
#
# Optional overrides (--export key=value):
#   SESSION         session number (default: all sessions)
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep)
#   SMOOTHED        set to "1" to use smoothed betas (default: off)
#   MODEL           model type: standard or session-shift (default: standard)
#   N_ITERATIONS    max gradient descent iterations (default: 1000)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

SESSION="${SESSION:-}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
SMOOTHED="${SMOOTHED:-0}"
MODEL="${MODEL:-standard}"
N_ITERATIONS="${N_ITERATIONS:-1000}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
    --n-iterations "$N_ITERATIONS"
)

[ -n "$SESSION" ] && ARGS+=(--sessions "$SESSION")
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)
[ "$MODEL" != "standard" ] && ARGS+=(--model "$MODEL")

echo "fit_aprf: sub-${PARTICIPANT_LABEL}  deriv=${FMRIPREP_DERIV}  smoothed=${SMOOTHED}  model=${MODEL}"
echo "Args: ${ARGS[*]}"

# Load environment
. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/fit_aprf.py" \
    "${ARGS[@]}"
