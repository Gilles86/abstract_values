#!/bin/bash
#SBATCH --job-name=cross_condition_decoding_aprf
#SBATCH --output=/home/gdehol/logs/cross_condition_decoding_aprf_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=zne.uzh

# Project models are TF-native; cluster keras.json defaults to jax — pin TF.
export KERAS_BACKEND=tensorflow

# Decode each mapping condition with the encoding model fitted on the OTHER
# one (aprf-session-shift). The mappings are inverted, so a voxel coding
# orientation carries value tuning that flips between them and cross-decodes
# at chance or worse, while a voxel coding value transfers. Compare against
# compute_identical_decoding_aprf.py (one shared model for both conditions).
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=11 compute_cross_condition_decoding_aprf.sh
#
# Optional overrides (--export key=value):
#   ROI                 ROI label (default: NPCr)
#   HEMI                hemisphere: LR, L, R, None (default: None)
#   N_VOXELS            top voxels by R² (default: 100)
#   N_NOISE_ITERATIONS  noise-model Adam iterations (default: 1000)
#   N_VALUES            decode grid points (default: 200)
#   SMOOTHED            set to "1" for smoothed betas (default: off)
#   SPHERICAL           set to "0" for full-covariance noise (default: on)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

ROI="${ROI:-NPCr}"
HEMI="${HEMI:-None}"
N_VOXELS="${N_VOXELS:-100}"
N_NOISE_ITERATIONS="${N_NOISE_ITERATIONS:-1000}"
N_VALUES="${N_VALUES:-200}"
SMOOTHED="${SMOOTHED:-0}"
SPHERICAL="${SPHERICAL:-1}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --roi "$ROI"
    --hemi "$HEMI"
    --n-voxels "$N_VOXELS"
    --n-noise-iterations "$N_NOISE_ITERATIONS"
    --n-values "$N_VALUES"
    --bids-folder "$BIDS_FOLDER"
)
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)
[ "$SPHERICAL" = "1" ] && ARGS+=(--spherical-noise) || ARGS+=(--no-spherical-noise)

echo "cross_condition_decoding_aprf: sub-${PARTICIPANT_LABEL}  roi=${ROI}  n_voxels=${N_VOXELS}  spherical=${SPHERICAL}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh
conda activate abstract_values

PYTHONUNBUFFERED=1 python -u \
    "$REPO/abstract_values/encoding_models/compute_cross_condition_decoding_aprf.py" \
    "${ARGS[@]}"
