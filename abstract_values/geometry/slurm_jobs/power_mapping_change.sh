#!/bin/bash
#SBATCH --job-name=cn_power
#SBATCH --output=/home/gdehol/logs/cn_power_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --account=zne.uzh

# Injection power check for the crossnobis mapping-change test.
#   sbatch --export=ALL,PARTICIPANT_LABEL=03 power_mapping_change.sh
. $HOME/init_conda.sh
conda activate abstract_values
PYTHONUNBUFFERED=1 python -u -m abstract_values.geometry.power_mapping_change "$PARTICIPANT_LABEL" \
    --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue
