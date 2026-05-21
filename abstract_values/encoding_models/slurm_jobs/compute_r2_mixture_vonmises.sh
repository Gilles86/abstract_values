#!/bin/bash
#SBATCH --job-name=vonmises_mixture
#SBATCH --output=/home/gdehol/logs/vonmises_mixture_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=zne.uzh

# Fit the whole-brain R² mixture for the vonmises encoding model
# across all subjects (auto-discovered). Outputs cached at
#   derivatives/encoding_models/vonmises/sub-XX/sub-XX_desc-p_signal.json
# plus a diagnostic PDF under derivatives/qa/r2_mixture/vonmises/.
#
# Run via:
#   sbatch abstract_values/encoding_models/slurm_jobs/compute_r2_mixture_vonmises.sh

. $HOME/init_conda.sh
conda activate abstract_values

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue

PYTHONUNBUFFERED=1 python -u -m abstract_values.encoding_models.compute_r2_mixture \
    --all --models vonmises --bids-folder "$BIDS_FOLDER"
