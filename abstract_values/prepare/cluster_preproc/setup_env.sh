#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=/home/gdehol/logs/setup_env_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

# Build the abstract_values conda environment on the cluster.
# Requires internet access (pip downloads) — run once per cluster setup.
#
# Usage:
#   sbatch setup_env.sh
#
# After this completes, all analysis jobs can use:
#   conda run -n abstract_values python ...
# or with a specific backend:
#   KERAS_BACKEND=jax conda run -n abstract_values python ...

source /etc/profile.d/lmod.sh
module load cuda/12.6.3        # needed for jax[cuda12] compilation

# ── locate mamba/conda ────────────────────────────────────────────────────────
if command -v mamba &>/dev/null; then
    CONDA_CMD=mamba
elif command -v conda &>/dev/null; then
    CONDA_CMD=conda
else
    echo "ERROR: neither mamba nor conda found. Install miniforge first."
    exit 1
fi

REPO=/home/$USER/git/abstract_values

# ── clone / update braincoder ─────────────────────────────────────────────────
BRAINCODER_DIR=/home/$USER/git/braincoder_keras
if [ ! -d "$BRAINCODER_DIR" ]; then
    git clone git@github.com:Gilles86/braincoder.git "$BRAINCODER_DIR"
fi
git -C "$BRAINCODER_DIR" fetch origin
git -C "$BRAINCODER_DIR" checkout keras-backend
git -C "$BRAINCODER_DIR" pull

# ── create or update environment ──────────────────────────────────────────────
if $CONDA_CMD env list | grep -q "^abstract_values "; then
    echo "Environment abstract_values exists — updating..."
    $CONDA_CMD env update -n abstract_values -f "$REPO/environment_linux.yml" --prune
else
    echo "Creating environment abstract_values..."
    $CONDA_CMD env create -f "$REPO/environment_linux.yml"
fi

# ── install braincoder from local clone ───────────────────────────────────────
# (overrides the git+https install with editable local version)
conda run -n abstract_values pip install -e "$BRAINCODER_DIR"

# ── install abstract_values package itself ────────────────────────────────────
conda run -n abstract_values pip install -e "$REPO"

# ── verify ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
conda run -n abstract_values python -c "
import keras; print('keras:', keras.__version__)
import jax; print('jax:', jax.__version__)
import jax.numpy as jnp
x = jnp.ones((10,10))
print('jax GPU devices:', jax.devices())
from braincoder.models import LogGaussianPRF, VonMisesPRF
from braincoder.optimize import ParameterFitter, ResidualFitter, WeightFitter
print('braincoder OK')
from abstract_values.utils.data import Subject
print('abstract_values OK')
"

echo ""
echo "Setup complete. Use KERAS_BACKEND=jax for GPU-accelerated fits."
