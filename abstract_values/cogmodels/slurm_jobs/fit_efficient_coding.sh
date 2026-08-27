#!/bin/bash
#SBATCH --job-name=fit_ec
#SBATCH --output=/home/gdehol/logs/fit_ec_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=zne.uzh

# Hierarchical MCMC fit of the Bedi et al. efficient-coding models.
#
#   sbatch --export=MODEL=sequential fit_efficient_coding.sh
#
# Optional overrides: DRAWS, TUNE, CHAINS, GRID, TARGET_ACCEPT.
# The paradigm TSV is built once beforehand with --write-paradigm (that step
# needs the abstract_values env); this job only needs bauer + pymc, so it runs
# in bauer_cuda and never imports the neuroimaging stack.

MODEL="${MODEL:-sequential}"
DRAWS="${DRAWS:-1500}"
TUNE="${TUNE:-1500}"
CHAINS="${CHAINS:-4}"
GRID="${GRID:-101}"
TARGET_ACCEPT="${TARGET_ACCEPT:-0.9}"
CONDITION="${CONDITION:-}"
CHAIN_METHOD="${CHAIN_METHOD:-sequential}"
LAPSE="${LAPSE:-0.01}"
PRIOR="${PRIOR:-long_term}"
# FREE_PRIOR=1 fits the prior peakedness; NOSEAM=1 closes the 0/180 deg seam.
FREE_PRIOR="${FREE_PRIOR:-}"
NOSEAM="${NOSEAM:-}"
# TRUNC=1 truncates orientation perception at the 0/90/180 cardinals.
TRUNC="${TRUNC:-}"
# FOURIER=K fits the prior as a K-harmonic circular Fourier series.
FOURIER="${FOURIER:-}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values
PARADIGM=$REPO/notes/data/efficient_coding_paradigm.tsv
OUTDIR=$BIDS_FOLDER/derivatives/cogmodels

export TMPDIR=/scratch/gdehol

# XLA compiles the whole fused graph ahead of time, and its CPU backend goes
# through LLVM, which is superlinear in fused-function size: grid 101 costs
# ~22 min before the first sample, grid 51 ~7 min (the CUDA path does the same
# graph in <1 min). JAX can cache the compiled executable across runs, keyed by
# the HLO -- so a resubmit of the same shape skips it. Not /tmp: that is a
# quota'd shared filesystem here and EDQUOT there has killed jobs before.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/scratch/gdehol/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "fit_efficient_coding (GPU): model=$MODEL draws=$DRAWS tune=$TUNE chains=$CHAINS grid=$GRID prior=$PRIOR chain_method=$CHAIN_METHOD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

cd "$REPO" || exit 1
PYTHONUNBUFFERED=1 $HOME/data/conda/envs/bauer_cuda/bin/python -u \
    -m abstract_values.cogmodels.fit_efficient_coding \
    --model "$MODEL" \
    ${CONDITION:+--condition "$CONDITION"} \
    --paradigm-tsv "$PARADIGM" \
    --grid-resolution "$GRID" \
    --draws "$DRAWS" --tune "$TUNE" --chains "$CHAINS" \
    --target-accept "$TARGET_ACCEPT" \
    --nuts-sampler numpyro \
    --chain-method "$CHAIN_METHOD" \
    --lapse-rate "$LAPSE" \
    --perceptual-prior "$PRIOR" \
    ${FREE_PRIOR:+--fit-prior-weight} \
    ${NOSEAM:+--no-seam-crossing} \
    ${TRUNC:+--cardinal-truncation} \
    ${FOURIER:+--prior-fourier-order "$FOURIER"} \
    --out-dir "$OUTDIR"
