#!/bin/bash
#SBATCH --job-name=cf_profiles
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/cf_profiles_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00

# Coupling profiles behind the mapping score, for plotting (profiles.py).
#
# Usage:
#   sbatch --array=3-30 profiles.sh
#   sbatch --export=ALL,PARTICIPANT_LABEL=pil01,PROJECTION=iem profiles.sh
#
# Optional overrides: PROJECTION (bins | iem), REPO (default ~/git/abstract_values)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi
REPO="${REPO:-$HOME/git/abstract_values}"

. $HOME/init_conda.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
conda run --no-capture-output -n abstract_values python -u -W ignore \
    -m abstract_values.connective_fields.profiles "$PARTICIPANT_LABEL" \
    --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue --projection "${PROJECTION:-bins}"
