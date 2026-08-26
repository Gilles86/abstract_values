#!/bin/bash
#SBATCH --job-name=fit_ec_cpu
#SBATCH --output=/home/gdehol/logs/fit_ec_cpu_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=zne.uzh

# CPU twin of fit_efficient_coding.sh.  The GPU partition is routinely backed up
# (dozens of pending gpu jobs cluster-wide, no estimated start), and these models
# have only one or two free parameters, so waiting for a GPU costs more wall
# clock than just running on cores.  numpyro runs one chain per core.
#
#   sbatch --export=MODEL=sequential fit_efficient_coding_cpu.sh

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

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values
PARADIGM=$REPO/notes/data/efficient_coding_paradigm.tsv
OUTDIR=$BIDS_FOLDER/derivatives/cogmodels

export TMPDIR=/scratch/gdehol

# numpyro's chain_method="parallel" pmaps over JAX *devices*.  On CPU there is
# exactly one device unless XLA is told otherwise, so without this the run
# silently falls back to one-chain-at-a-time -- which is what made the
# long_term sequential fit miss its wall clock (18.5 h per chain x 4).
if [ "$CHAIN_METHOD" = "parallel" ]; then
    export XLA_FLAGS="--xla_force_host_platform_device_count=$CHAINS"
fi


echo "fit_efficient_coding (CPU): model= draws= tune= chains= grid= prior= chain_method= XLA_FLAGS="

cd "$REPO" || exit 1
PYTHONUNBUFFERED=1 $HOME/data/conda/envs/bauer/bin/python -u \
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
    --out-dir "$OUTDIR"
