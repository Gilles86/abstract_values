#!/bin/bash
#SBATCH --job-name=decode_value_fullfit
#SBATCH --output=/home/gdehol/logs/decode_value_fullfit_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Project models are TF-native; cluster keras.json defaults to jax — pin TF.
export KERAS_BACKEND=tensorflow

# Decode EVERY trial using the already-fit, full-data (non-cross-validated)
# encoding model — no LORO refitting, no ParameterFitter call at all (params
# are loaded straight from derivatives/encoding_models/{aprf,aprf-linear,
# aprf-session-shift}). Fast — the only fitting here is the noise model.
#
# Requires MASK and MASK_DESC to be set (no whole-brain default).
#
#   sbatch --export=PARTICIPANT_LABEL=pil01,MASK=...,MASK_DESC=BensonV1,MODEL=loggauss decode_value_fullfit.sh
#
# Optional overrides (--export key=value):
#   SESSION       session number (default: all sessions)
#   SMOOTHED      set to "1" to use spatially smoothed betas (default: off)
#   SPHERICAL     set to "1" for spherical (diagonal) noise model (default: full)
#   N_VOXELS      top voxels by the full-fit R² (default: 100; 0 = all voxels in mask)
#   LAMBD         ResidualFitter regularisation λ (default: 0)
#   MODEL         loggauss (default) | linear | session-shift
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep)

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
LAMBD="${LAMBD:-0}"
MODEL="${MODEL:-loggauss}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --mask "$MASK"
    --mask-desc "$MASK_DESC"
    --n-voxels "$N_VOXELS"
    --lambd "$LAMBD"
    --model "$MODEL"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
)

[ -n "$SESSION" ] && ARGS+=(--sessions "$SESSION")
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)
[ "$SPHERICAL" = "1" ] && ARGS+=(--spherical-noise)

echo "decode_value_fullfit: sub-${PARTICIPANT_LABEL}  mask=${MASK_DESC}  model=${MODEL}  smoothed=${SMOOTHED}  spherical=${SPHERICAL}  λ=${LAMBD}  n_voxels=${N_VOXELS}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/decode_value_fullfit.py" \
    "${ARGS[@]}"
