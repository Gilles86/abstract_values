#!/bin/bash
#SBATCH --job-name=decode_lowrankcv
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/gdehol/logs/decode_lowrankcv_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Production decoding (vonmises orientation / loggauss value, top-N voxels by
# training R², leave-one-run-out) decoded with spherical AND low-rank (nested
# LORO k, T) noise on the same voxels. Writes derivatives/decoding_lowrankcv/.
#
# CPU only (no GPU needed). Array index -> subject (1..36 = pil01 pil02 03..36). Required: QUANTITY, ROI.
#   sbatch --array=1-36 --export=ALL,QUANTITY=orientation,ROI=BensonV1 decode_lowrank_cv.sh
#   sbatch --array=1-35 --export=ALL,QUANTITY=value,ROI=IPS0 decode_lowrank_cv.sh
#   sbatch --export=ALL,PARTICIPANT_LABEL=36,QUANTITY=value,ROI=IPS12 decode_lowrank_cv.sh
set -eo pipefail

SUBJECTS=(pil01 pil02 $(seq -f "%02g" 3 36))
if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}
fi
if [ -z "$QUANTITY" ] || [ -z "$ROI" ]; then
    echo "ERROR: QUANTITY and ROI must be set via --export=ALL,..."; exit 1
fi

export KERAS_BACKEND=tensorflow
# Native braincoder.noise.LowRankNoise: braincoder v0.6.0 + branch
# lowrank-inner-weights (inner-fold weights for basis-set models), checked out
# at ~/git/braincoder-lowrank and put ahead of the env's libs/braincoder:
#   git clone -b lowrank-inner-weights git@github.com:Gilles86/braincoder.git ~/git/braincoder-lowrank
export PYTHONPATH="${BRAINCODER_PATH:-$HOME/git/braincoder-lowrank}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# stagger array starts (NFS load on the GLMsingle betas)
sleep $(( (${SLURM_ARRAY_TASK_ID:-0} % 12) * 5 ))

echo "decode_lowrankcv: sub-${PARTICIPANT_LABEL} ${QUANTITY} ${ROI} N=${N_VOXELS:-50 250}"
exec $HOME/data/conda/envs/abstract_values/bin/python -u \
    $HOME/git/abstract_values/abstract_values/encoding_models/decode_lowrank_cv.py \
    "$PARTICIPANT_LABEL" --quantity "$QUANTITY" --roi "$ROI" \
    --n-voxels ${N_VOXELS:-50 250} \
    --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue
