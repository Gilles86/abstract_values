#!/bin/bash
#SBATCH --job-name=fit_vonmises
#SBATCH --output=/home/gdehol/logs/fit_vonmises_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Fit Von Mises basis set encoding model to single-trial GLMsingle betas.
#
# Two ways to run:
#   Array job:  sbatch --array=1-30 fit_vonmises.sh
#   By name:    sbatch --export=PARTICIPANT_LABEL=pil01 fit_vonmises.sh
#
# Optional overrides (--export key=value):
#   SESSION       session number, e.g. SESSION=1 (default: all sessions)
#   SMOOTHED      set to "1" to fit on spatially smoothed betas (default: off)
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep)
#   N_BASIS       number of Von Mises basis functions (default: 8)
#   KAPPA         Von Mises concentration parameter (default: 2.0)
#
# Example (pilot, session 1):
#   sbatch --export=PARTICIPANT_LABEL=pil01,SESSION=1 fit_vonmises.sh
# Smoothed:
#   sbatch --export=PARTICIPANT_LABEL=pil01,SESSION=1,SMOOTHED=1 fit_vonmises.sh

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

SESSION="${SESSION:-}"
SMOOTHED="${SMOOTHED:-0}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
N_BASIS="${N_BASIS:-8}"
KAPPA="${KAPPA:-2.0}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=/home/gdehol/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
    --n-basis "$N_BASIS"
    --kappa "$KAPPA"
)

[ -n "$SESSION" ] && ARGS+=(--sessions "$SESSION")
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

echo "fit_vonmises: sub-${PARTICIPANT_LABEL}  deriv=${FMRIPREP_DERIV}  smoothed=${SMOOTHED}  n_basis=${N_BASIS}  kappa=${KAPPA}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/fit_vonmises_model.py" \
    "${ARGS[@]}"
