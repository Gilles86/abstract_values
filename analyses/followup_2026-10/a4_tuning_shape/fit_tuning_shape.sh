#!/bin/bash
#SBATCH --job-name=a4_tuning_shape
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/gdehol/logs/a4_tuning_shape_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=05:00:00
# A4 tuning-shape fits + decoding, one subject per array task.
#   sbatch --array=0-35 --export=ALL fit_tuning_shape.sh       (index into SUBJECTS)
#   sbatch --export=ALL,PARTICIPANT_LABEL=05,EXTRA=--debug fit_tuning_shape.sh
set -eo pipefail
SUBJECTS=(pil01 pil02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
          25 26 27 28 29 30 31 32 33 34 35 36)
PARTICIPANT_LABEL="${PARTICIPANT_LABEL:-${SUBJECTS[$SLURM_ARRAY_TASK_ID]}}"
export KERAS_BACKEND=tensorflow PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
REPO=$HOME/git/abstract_values
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
echo "a4_tuning_shape sub-${PARTICIPANT_LABEL} ${EXTRA:-}"
exec $HOME/data/conda/envs/abstract_values/bin/python -u \
    "$REPO/analyses/followup_2026-10/a4_tuning_shape/fit_tuning_shape.py" \
    "$PARTICIPANT_LABEL" --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue ${EXTRA:-}
