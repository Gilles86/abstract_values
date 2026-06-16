#!/bin/bash
#SBATCH --job-name=decode_value_mc
#SBATCH --output=/home/gdehol/logs/decode_value_mc_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=zne.uzh

# Out-of-sample VALUE decoding tiebreaker: single vs weighted x joint vs
# separate, on the union FDR voxel set. CPU-only. Requires
# select_value_voxels_fdr to have run (union voxel TSV).
#
# Usage:
#   sbatch --array=3-14 decode_value_modelcomp.sh
#
# Optional: N_BASIS / FWHM (weighted config), N_ITER, NOISE_ITER, SMOOTHED

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi

N_BASIS="${N_BASIS:-8}"
FWHM="${FWHM:-6}"
N_ITER="${N_ITER:-500}"
NOISE_ITER="${NOISE_ITER:-1000}"
SMOOTHED="${SMOOTHED:-0}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values
ENV=$HOME/data/conda/envs/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --n-basis "$N_BASIS" --fwhm "$FWHM"
    --n-iter "$N_ITER" --noise-iter "$NOISE_ITER"
)
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

echo "decode_value_mc: sub-${PARTICIPANT_LABEL}  k=${N_BASIS} fwhm=${FWHM}"
echo "Args: ${ARGS[*]}"

exec ${ENV}/bin/python -u \
    "$REPO/abstract_values/encoding_models/decode_value_modelcomp.py" \
    "${ARGS[@]}"
