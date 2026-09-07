#!/bin/bash
#SBATCH --job-name=eu_value_weighted
#SBATCH --output=/home/gdehol/logs/eu_value_weighted_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=zne.uzh

# Project models are TF-native; cluster keras.json defaults to jax — pin TF.
export KERAS_BACKEND=tensorflow

# Expected decoded VALUE from the weighted log-Gaussian basis, per session.
# The value-space twin of compute_expected_decoded_orientation_vonmises.sh.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=03 compute_expected_decoded_value_weighted.sh
#
# Optional overrides (--export key=value):
#   ROI             ROI label (default: NPCr)
#   HEMI            hemisphere: LR, L, R, None (default: None)
#   N_VOXELS        top voxels by joint R² (default: 100)
#   N_BASIS         log-Gaussian basis functions (default: 8)
#   BASIS_FWHM      basis FWHM in CHF (default: 2x inter-basis spacing)
#   N_SIMULATIONS   noisy repeats per stimulus (default: 1000)
#   SMOOTHED        set to "1" for smoothed betas (default: off)
#   SPHERICAL       set to "0" for full-covariance noise (default: on)
#
# The ridge penalty is not an override: it is the project default (see
# ridge_alpha.py) and the script refuses another value without an explicit
# --allow-nondefault-alpha.

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi

ROI="${ROI:-NPCr}"
HEMI="${HEMI:-None}"
N_VOXELS="${N_VOXELS:-100}"
N_BASIS="${N_BASIS:-8}"
BASIS_FWHM="${BASIS_FWHM:-}"
N_SIMULATIONS="${N_SIMULATIONS:-1000}"
SMOOTHED="${SMOOTHED:-0}"
SPHERICAL="${SPHERICAL:-1}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --roi "$ROI"
    --hemi "$HEMI"
    --n-voxels "$N_VOXELS"
    --n-basis "$N_BASIS"
    --n-simulations "$N_SIMULATIONS"
    --bids-folder "$BIDS_FOLDER"
)
[ -n "$BASIS_FWHM" ] && ARGS+=(--basis-fwhm "$BASIS_FWHM")
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)
[ "$SPHERICAL" = "1" ] && ARGS+=(--spherical-noise) || ARGS+=(--no-spherical-noise)

echo "eu_value_weighted: sub-${PARTICIPANT_LABEL}  roi=${ROI}  n_voxels=${N_VOXELS}  n_basis=${N_BASIS}  spherical=${SPHERICAL}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh
conda activate abstract_values

PYTHONUNBUFFERED=1 python -u \
    "$REPO/abstract_values/encoding_models/compute_expected_decoded_value_weighted.py" \
    "${ARGS[@]}"
