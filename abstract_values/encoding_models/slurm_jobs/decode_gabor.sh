#!/bin/bash
#SBATCH --job-name=decode_gabor
#SBATCH --output=/home/gdehol/logs/decode_gabor_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00   # bumped from 30 min — matches decode_value.sh; fdr05

# Project models are TF-native; cluster keras.json defaults to jax — pin TF.
export KERAS_BACKEND=tensorflow
                          # selections can be large and 30 min runs out

# Bayesian leave-one-run-out gabor orientation decoding.
# Requires MASK and MASK_DESC to be set (no whole-brain default).
#
# Two ways to run:
#   Array job:  sbatch --array=1-30 decode_gabor.sh
#   By name:    sbatch --export=PARTICIPANT_LABEL=pil01,MASK=...,MASK_DESC=BensonV1 decode_gabor.sh
#
# Required overrides (--export key=value):
#   MASK          full path to brain mask NIfTI
#   MASK_DESC     short label, e.g. BensonV1 (used in output filename)
#
# Optional overrides:
#   SESSION       session number (default: all sessions)
#   SMOOTHED      set to "1" to use spatially smoothed betas (default: off)
#   SPHERICAL     set to "1" for spherical (diagonal) noise model (default: full)
#   GEODESIC      set to "1" for geodesic-distance spatial noise model
#                 (mutually exclusive with SPHERICAL; single-hemisphere ROI)
#   GEODESIC_HEMI hemisphere for geodesic surface distance (default: R)
#   N_VOXELS      top voxels to decode with (default: 100)
#   FDR_ALPHA     FDR-control α on whole-brain R² mixture (mutually exclusive
#                 with P_SIGNAL_THR). Output filename: nvoxels-fdrNN.
#   P_SIGNAL_THR  P(signal|r²) cutoff on whole-brain R² mixture (mutually
#                 exclusive with FDR_ALPHA). Output filename: nvoxels-psigNN.
#   RIVAL_VAL     set to "1" to keep only voxels the orientation model beats
#                 the value (log-Gaussian basis) rival on, fold-wise.
#                 Requires N_VOXELS=0. Output filename gets a -vsval tag.
#   LAMBD         ResidualFitter regularisation λ (default: 0.1)
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep-flair)
#   MODEL         vonmises (default, tuned bump) | linear (no tuning bump,
#                 cos(2x)/sin(2x) closed-form fit). Output subdir:
#                 derivatives/decoding/{gabor,gabor-linear}/.
#
# Example — V1 bilateral, both noise models, pilot ses-1:
#   MASK=/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/masks/sub-pil01/ses-1/anat/sub-pil01_ses-1_space-T1w_hemi-LR_desc-BensonV1_mask.nii.gz
#   sbatch --export=PARTICIPANT_LABEL=pil01,SESSION=1,MASK=$MASK,MASK_DESC=BensonV1 decode_gabor.sh
#   sbatch --export=PARTICIPANT_LABEL=pil01,SESSION=1,MASK=$MASK,MASK_DESC=BensonV1,SPHERICAL=1 decode_gabor.sh

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
GEODESIC="${GEODESIC:-0}"
GEODESIC_HEMI="${GEODESIC_HEMI:-R}"
N_VOXELS="${N_VOXELS:-100}"
RIVAL_VAL="${RIVAL_VAL:-0}"
FDR_ALPHA="${FDR_ALPHA:-}"
P_SIGNAL_THR="${P_SIGNAL_THR:-}"
# lambd>0 silently overrides the geodesic Omega in braincoder's
# ResidualFitter (see decode_gabor.py's assert) — default to 0 whenever
# GEODESIC=1 unless the caller explicitly overrides LAMBD themselves.
if [ "$GEODESIC" = "1" ]; then
    LAMBD="${LAMBD:-0}"
else
    LAMBD="${LAMBD:-0.1}"
fi
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
MODEL="${MODEL:-vonmises}"

if [ "$RIVAL_VAL" = "1" ] && [ "$N_VOXELS" != "0" ]; then
    echo "ERROR: RIVAL_VAL=1 requires N_VOXELS=0."
    exit 1
fi

if [ -n "$FDR_ALPHA" ] && [ -n "$P_SIGNAL_THR" ]; then
    echo "ERROR: FDR_ALPHA and P_SIGNAL_THR are mutually exclusive."
    exit 1
fi

if [ "$SPHERICAL" = "1" ] && [ "$GEODESIC" = "1" ]; then
    echo "ERROR: SPHERICAL and GEODESIC are mutually exclusive."
    exit 1
fi

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
[ "$GEODESIC" = "1" ] && ARGS+=(--geodesic-noise --geodesic-hemi "$GEODESIC_HEMI")
[ -n "$FDR_ALPHA" ] && ARGS+=(--fdr-alpha "$FDR_ALPHA")
[ -n "$P_SIGNAL_THR" ] && ARGS+=(--p-signal-thr "$P_SIGNAL_THR")
[ "$MODEL" != "vonmises" ] && ARGS+=(--model "$MODEL")
[ "$RIVAL_VAL" = "1" ] && ARGS+=(--rival-value)

echo "decode_gabor: sub-${PARTICIPANT_LABEL}  mask=${MASK_DESC}  smoothed=${SMOOTHED}  spherical=${SPHERICAL}  geodesic=${GEODESIC}  λ=${LAMBD}  fdr=${FDR_ALPHA}  psig=${P_SIGNAL_THR}  model=${MODEL}"
echo "Args: ${ARGS[*]}"

# Load environment
. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/decode_gabor.py" \
    "${ARGS[@]}"
