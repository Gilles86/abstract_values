#!/bin/bash
#SBATCH --job-name=fit_aprf_cv
#SBATCH --output=/home/gdehol/logs/fit_aprf_cv_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# 12h matches the Snakefile's _aprf_cv_resources() budget — the heavier
# shift-ladder CV variants (fwhm-shift, fully-shifted; ~16 LORO folds each)
# were timing out at the old 6h default (see backfill 2026-08-12/13).

# Project models are TF-native; cluster keras.json defaults to jax — pin TF.
export KERAS_BACKEND=tensorflow

# Leave-one-run-out CV for the abstract pRF encoding model.
# Supports all four model variants from fit_aprf_cv.py:
#   standard             — LogGaussianPRF
#   session-shift        — SessionShiftedLogGaussianPRF (requires ≥2 sessions)
#   gaussian             — symmetric GaussianValuePRF
#   gauss-session-shift  — symmetric SessionShiftedGaussianValuePRF
#
# Always fits jointly across all of a subject's MRI sessions.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=pil01 fit_aprf_cv.sh
#   sbatch --export=PARTICIPANT_LABEL=pil01,MODEL=session-shift fit_aprf_cv.sh
#   sbatch --export=PARTICIPANT_LABEL=pil01,MODEL=gaussian fit_aprf_cv.sh
#
# Optional overrides (--export key=value):
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep)
#   SMOOTHED        set to "1" to use smoothed betas (default: off)
#   N_ITERATIONS    max gradient descent iterations per fold (default: 1000)
#   MODEL           standard|session-shift|fwhm-only-shift|fwhm-shift|
#                   fully-shifted|gaussian|gauss-session-shift|linear
#                   (default: standard)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
SMOOTHED="${SMOOTHED:-0}"
N_ITERATIONS="${N_ITERATIONS:-1000}"
MODEL="${MODEL:-standard}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
    --n-iterations "$N_ITERATIONS"
)

[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)
[ "$MODEL" != "standard" ] && ARGS+=(--model "$MODEL")

echo "fit_aprf_cv: sub-${PARTICIPANT_LABEL}  deriv=${FMRIPREP_DERIV}  smoothed=${SMOOTHED}  model=${MODEL}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/fit_aprf_cv.py" \
    "${ARGS[@]}"
