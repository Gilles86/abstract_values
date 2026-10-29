#!/bin/bash
#SBATCH --job-name=a7_m1wrap
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/a7_m1wrap_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --partition=standard
#
# Wrapped value/bid PRF in M1 with parameters kept (fit_m1_wrapped.py).
# Task i -> subject SUBJECTS[i % 36], epoch EPOCHS[i / 36].
#
#   sbatch --array=0-71 --export=ALL fit_m1_wrapped.sh
set -eo pipefail

SUBJECTS=(pil01 pil02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 \
          21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36)
EPOCHS=(response gabor)
i=${SLURM_ARRAY_TASK_ID:-0}
SUBJECT=${SUBJECT:-${SUBJECTS[$((i % 36))]}}
EPOCH=${EPOCH:-${EPOCHS[$((i / 36))]}}

export KERAS_BACKEND=tensorflow
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
REPO="${REPO:-$HOME/git/abstract_values}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

sleep $(( (i % 10) * 3 ))
echo "a7 m1 wrapped: sub-${SUBJECT} epoch=${EPOCH}"
exec $HOME/data/conda/envs/abstract_values/bin/python -u \
    "$REPO/analyses/followup_2026-10/a7_m1_response_code/fit_m1_wrapped.py" "$SUBJECT" \
    --epoch "$EPOCH" --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue \
    ${EXTRA_ARGS}
