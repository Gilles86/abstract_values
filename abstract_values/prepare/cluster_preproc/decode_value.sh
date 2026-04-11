#!/bin/bash
#SBATCH --job-name=decode_value
#SBATCH --output=/home/gdehol/logs/decode_value_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# Bayesian leave-one-run-out abstract value (CHF) decoding.
# Requires MASK and MASK_DESC to be set (no whole-brain default).
#
# Two ways to run:
#   Array job:  sbatch --array=1-30 decode_value.sh
#   By name:    sbatch --export=PARTICIPANT_LABEL=pil01,MASK=...,MASK_DESC=BensonV1 decode_value.sh
#
# Required overrides (--export key=value):
#   MASK          full path to brain mask NIfTI
#   MASK_DESC     short label, e.g. BensonV1 (used in output filename)
#
# Optional overrides:
#   SESSION       session number (default: all sessions)
#   SMOOTHED      set to "1" to use spatially smoothed betas (default: off)
#   SPHERICAL     set to "1" for spherical (diagonal) noise model (default: full)
#   N_VOXELS      top voxels to decode with (default: 100)
#   LAMBD         ResidualFitter regularisation λ (default: 0.0)
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep-flair)
#
# Example — V1 bilateral, full noise model, pilot ses-1:
#   MASK=/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/masks/sub-pil01/ses-1/anat/sub-pil01_ses-1_space-T1w_hemi-LR_desc-BensonV1_mask.nii.gz
#   sbatch --export=PARTICIPANT_LABEL=pil01,SESSION=1,MASK=$MASK,MASK_DESC=BensonV1 decode_value.sh
#   sbatch --export=PARTICIPANT_LABEL=pil01,SESSION=1,MASK=$MASK,MASK_DESC=BensonV1,SPHERICAL=1 decode_value.sh

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

if [ -z "$MASK" ] || [ -z "$MASK_DESC" ]; then
    echo "ERROR: MASK and MASK_DESC must be set via --export."
    exit 1
fi

SESSION="${SESSION:-}"
SMOOTHED="${SMOOTHED:-0}"
SPHERICAL="${SPHERICAL:-0}"
N_VOXELS="${N_VOXELS:-100}"
LAMBD="${LAMBD:-0.0}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep-flair}"
MODEL="${MODEL:-loggauss}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --mask "$MASK"
    --mask-desc "$MASK_DESC"
    --n-voxels "$N_VOXELS"
    --lambd "$LAMBD"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
)

[ -n "$SESSION" ] && ARGS+=(--sessions "$SESSION")
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)
[ "$SPHERICAL" = "1" ] && ARGS+=(--spherical-noise)
[ "$MODEL" != "loggauss" ] && ARGS+=(--model "$MODEL")

echo "decode_value: sub-${PARTICIPANT_LABEL}  mask=${MASK_DESC}  smoothed=${SMOOTHED}  spherical=${SPHERICAL}  λ=${LAMBD}  model=${MODEL}"
echo "Args: ${ARGS[*]}"

# Load environment
. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/decode_value.py" \
    "${ARGS[@]}"
