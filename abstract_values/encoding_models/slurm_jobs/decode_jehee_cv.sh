#!/bin/bash
#SBATCH --job-name=decode_jeheecv
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/gdehol/logs/decode_jeheecv_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Production decoding (vonmises orientation / loggauss value, top-N voxels by
# training R², leave-one-run-out) decoded with spherical AND van Bergen & Jehee
# (rho, tau, sigma² WW') noise on the same voxels, N = 50, 250 and the whole ROI.
# Writes derivatives/decoding_jeheecv/.
#
# CPU only. Array index -> subject (1..36 = pil01 pil02 03..36). Required: QUANTITY, ROI.
#   sbatch --array=1-36 --export=ALL,QUANTITY=orientation,ROI=BensonV1 decode_jehee_cv.sh
set -eo pipefail

SUBJECTS=(pil01 pil02 $(seq -f "%02g" 3 36))
if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}
fi
if [ -z "$QUANTITY" ] || [ -z "$ROI" ]; then
    echo "ERROR: QUANTITY and ROI must be set via --export=ALL,..."; exit 1
fi

export KERAS_BACKEND=tensorflow
# same braincoder as decode_lowrank_cv.sh (decode_lowrank_cv imports LowRankNoise)
export PYTHONPATH="${BRAINCODER_PATH:-$HOME/git/braincoder-lowrank}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

sleep $(( (${SLURM_ARRAY_TASK_ID:-0} % 12) * 5 ))

echo "decode_jeheecv: sub-${PARTICIPANT_LABEL} ${QUANTITY} ${ROI} N=${N_VOXELS:-50 250 all}"
exec $HOME/data/conda/envs/abstract_values/bin/python -u \
    $HOME/git/abstract_values/abstract_values/encoding_models/decode_jehee_cv.py \
    "$PARTICIPANT_LABEL" --quantity "$QUANTITY" --roi "$ROI" \
    --n-voxels ${N_VOXELS:-50 250 all} ${EXTRA_ARGS} \
    --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue
