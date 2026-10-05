#!/bin/bash
#SBATCH --job-name=fit_ec
#SBATCH --output=/home/gdehol/logs/fit_ec_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
# Jobs 6892838/9 both landed on the same 2-GPU H100 NVL node (u24-chaihm0-633,
# GPUMEM96GB) in the same second and both died in 25 s with "Unable to
# initialize backend 'cuda': no supported devices found" -- either the
# simultaneous-cuInit race or a driver/CUDA mismatch on that node type, which
# no earlier fit had ever run on. Belt and braces: stick to the node types
# every past fit succeeded on (A100 80GB, H100 HBM3, H200), and warm CUDA
# under a per-node lock below, failing fast rather than sampling on CPU.
#SBATCH --constraint=GPUMEM80GB|GPUMEM140GB
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
GROUP_SD="${GROUP_SD:-halfnormal}"
FIND_INIT="${FIND_INIT:-}"
MOTOR="${MOTOR:-}"
# FOURIER=K fits the prior as a K-harmonic circular Fourier series.
FOURIER="${FOURIER:-}"
# PARAM=total-share samples total bid noise + perceptual share instead of
# (kappa_r, sigma_rep); sequential/categorical only. See cogmodels/reparam.py.
PARAM="${PARAM:-kappa-sigma}"

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

echo "fit_efficient_coding (GPU): model=$MODEL draws=$DRAWS tune=$TUNE chains=$CHAINS grid=$GRID prior=$PRIOR param=$PARAM chain_method=$CHAIN_METHOD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# cuInit warm-up under a node-local lock (sciencecluster skill, gpu_jobs.md):
# the first job on a node initialises the driver alone, later ones find it
# warm. Retry a few times; if JAX still sees no GPU, stop -- a silent CPU
# fallback would take ~20x the walltime.
PY=$HOME/data/conda/envs/bauer_cuda/bin/python
ok=0
for attempt in 1 2 3; do
    if ( flock -w 120 -x 200 || exit 0
         $PY -c "import jax; d = jax.devices('gpu'); print(f'cuInit OK: {d}', flush=True)"
       ) 200>"/tmp/cuinit_warm_$(hostname -s).flock"; then
        ok=1; break
    fi
    echo "cuInit attempt $attempt failed; retrying in $((attempt * 20)) s"
    sleep $((attempt * 20))
done
if [ "$ok" != 1 ]; then
    echo "ERROR: JAX cannot see a GPU on $(hostname -s); not falling back to CPU." >&2
    exit 3
fi

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
    --group-sd-dist "$GROUP_SD" \
    ${FIND_INIT:+--find-init "$FIND_INIT"} \
    ${MOTOR:+--fit-motor-noise} \
    ${FOURIER:+--prior-fourier-order "$FOURIER"} \
    --param "$PARAM" \
    --out-dir "$OUTDIR"
