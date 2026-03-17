#!/bin/bash
#SBATCH --job-name=benchmark_backends
#SBATCH --output=/home/gdehol/logs/benchmark_backends_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Benchmark Keras backends (TensorFlow / JAX / PyTorch) on real fMRI data.
# Each backend runs in a fresh subprocess so the Keras backend can be switched.
# JAX uses CUDA on the GPU node; TF and Torch default to CPU unless configured.
#
# Results are printed to the SLURM log AND saved as JSON for later inspection.
#
# Usage:
#   sbatch benchmark_backends.sh
#
# Optional overrides via --export key=value:
#   PARTICIPANT_LABEL  subject label (default: pil01)
#   SESSION            session number (default: 1)
#   MASK_DESC          mask label (default: BensonV1)
#   N_VOXELS           number of voxels (default: 200)
#   N_ITERATIONS       gradient descent iterations (default: 500)
#   BACKENDS           space-separated list, e.g. "tensorflow jax" (default: all 3)
#
# Examples:
#   sbatch benchmark_backends.sh
#   sbatch --export=N_VOXELS=500,N_ITERATIONS=1000 benchmark_backends.sh
#   sbatch --export=BACKENDS="tensorflow jax" benchmark_backends.sh

PARTICIPANT_LABEL="${PARTICIPANT_LABEL:-pil01}"
SESSION="${SESSION:-1}"
MASK_DESC="${MASK_DESC:-BensonV1}"
N_VOXELS="${N_VOXELS:-200}"
N_ITERATIONS="${N_ITERATIONS:-500}"
BACKENDS="${BACKENDS:-tensorflow jax torch}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=/shares/zne.uzh/gdehol/git/abstract_values

MASK="${BIDS_FOLDER}/derivatives/masks/sub-${PARTICIPANT_LABEL}/ses-${SESSION}/anat/sub-${PARTICIPANT_LABEL}_ses-${SESSION}_space-T1w_hemi-LR_desc-${MASK_DESC}_mask.nii.gz"

OUT_JSON="${BIDS_FOLDER}/derivatives/benchmarks/sub-${PARTICIPANT_LABEL}_ses-${SESSION}_mask-${MASK_DESC}_nvoxels-${N_VOXELS}_niters-${N_ITERATIONS}_benchmark.json"

# ── load CUDA ─────────────────────────────────────────────────────────────────
source /etc/profile.d/lmod.sh
module load cuda/12.4

echo "============================================================"
echo "  Backend benchmark"
echo "  subject:    sub-${PARTICIPANT_LABEL}"
echo "  session:    ses-${SESSION}"
echo "  mask:       ${MASK_DESC}"
echo "  n_voxels:   ${N_VOXELS}"
echo "  n_iters:    ${N_ITERATIONS}"
echo "  backends:   ${BACKENDS}"
echo "  out_json:   ${OUT_JSON}"
echo "  SLURM job:  ${SLURM_JOB_ID}"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"
echo ""

conda run -n abstract_values python -u \
    "${REPO}/abstract_values/encoding_models/benchmark_backends.py" \
    "${PARTICIPANT_LABEL}" \
    --sessions "${SESSION}" \
    --mask "${MASK}" \
    --mask-desc "${MASK_DESC}" \
    --n-voxels "${N_VOXELS}" \
    --n-iterations "${N_ITERATIONS}" \
    --backends ${BACKENDS} \
    --out-json "${OUT_JSON}" \
    --bids-folder "${BIDS_FOLDER}"

echo ""
echo "Done. Results in: ${OUT_JSON}"
echo "SLURM log:        /home/gdehol/logs/benchmark_backends_${SLURM_JOB_ID}.txt"
