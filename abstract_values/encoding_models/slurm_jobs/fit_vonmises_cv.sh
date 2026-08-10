#!/bin/bash
#SBATCH --job-name=fit_vonmises_cv
#SBATCH --output=/home/gdehol/logs/fit_vonmises_cv_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00

# Project models are TF-native; cluster keras.json defaults to jax — pin TF.
export KERAS_BACKEND=tensorflow

# Leave-one-run-out CV for the Von Mises basis-set (orientation/Gabor) model.
# Always fits jointly across all of a subject's MRI sessions.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=pil01 fit_vonmises_cv.sh
#
# Optional overrides (--export key=value):
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep)
#   SMOOTHED        set to "1" to use smoothed betas (default: off)
#   N_BASIS         number of Von Mises basis functions (default: 8)
#   KAPPA           concentration parameter (default: 2.0)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
SMOOTHED="${SMOOTHED:-0}"
SESSION_SHIFT="${SESSION_SHIFT:-0}"
N_BASIS="${N_BASIS:-8}"
KAPPA="${KAPPA:-2.0}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
    --n-basis "$N_BASIS"
    --kappa "$KAPPA"
)

[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)
[ "$SESSION_SHIFT" = "1" ] && ARGS+=(--session-shift)

echo "fit_vonmises_cv: sub-${PARTICIPANT_LABEL}  deriv=${FMRIPREP_DERIV}  smoothed=${SMOOTHED}  n_basis=${N_BASIS}  kappa=${KAPPA}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/fit_vonmises_cv.py" \
    "${ARGS[@]}"
