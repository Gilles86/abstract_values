#!/bin/bash
#SBATCH --job-name=benchmark_backends
#SBATCH --output=/home/gdehol/logs/benchmark_backends_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
#   N_ITERATIONS       gradient descent iterations (default: 500)
#   BACKENDS           space-separated list, e.g. "tensorflow jax" (default: all 3)
#
# Examples:
#   sbatch benchmark_backends.sh
#   sbatch --export=N_ITERATIONS=1000 benchmark_backends.sh
#   sbatch --export=BACKENDS="tensorflow jax" benchmark_backends.sh

PARTICIPANT_LABEL="${PARTICIPANT_LABEL:-pil01}"
SESSION="${SESSION:-1}"
N_ITERATIONS="${N_ITERATIONS:-500}"
BACKENDS="${BACKENDS:-tensorflow jax torch}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=/home/gdehol/git/abstract_values

OUT_JSON="${BIDS_FOLDER}/derivatives/benchmarks/sub-${PARTICIPANT_LABEL}_ses-${SESSION}_niters-${N_ITERATIONS}_benchmark.json"

# ── load CUDA ─────────────────────────────────────────────────────────────────
source /etc/profile.d/lmod.sh
module load cuda/12.6.3

echo "============================================================"
echo "  Backend benchmark"
echo "  subject:    sub-${PARTICIPANT_LABEL}"
echo "  session:    ses-${SESSION}"
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
    --n-iterations "${N_ITERATIONS}" \
    --backends ${BACKENDS} \
    --out-json "${OUT_JSON}" \
    --bids-folder "${BIDS_FOLDER}"

echo ""
echo "Done. Results in: ${OUT_JSON}"
echo "SLURM log:        /home/gdehol/logs/benchmark_backends_${SLURM_JOB_ID}.txt"
