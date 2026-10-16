#!/bin/bash
#SBATCH --job-name=crossnobis
#SBATCH --output=/home/gdehol/logs/crossnobis_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --account=zne.uzh

# Crossnobis RDMs (23 orientations x session) for all dorsal ROIs of one subject.
#   sbatch --export=ALL,PARTICIPANT_LABEL=03 crossnobis.sh
# Optional: SMOOTHED=1

SMOOTHED="${SMOOTHED:-0}"
ARGS=("$PARTICIPANT_LABEL" --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue)
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

. $HOME/init_conda.sh
conda activate abstract_values
PYTHONUNBUFFERED=1 python -u -m abstract_values.geometry.crossnobis "${ARGS[@]}"
