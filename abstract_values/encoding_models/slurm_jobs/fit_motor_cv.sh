#!/bin/bash
#SBATCH --job-name=fit_motor_cv
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/fit_motor_cv_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --partition=standard
#
# Leave-one-run-out cvR² of the single-PRF models in bilateral M1
# (fit_motor_cv.py), one subject x epoch per array task.
# Task index i -> subject SUBJECTS[i % 35], epoch DESCS[i / 35].
#
#   sbatch --array=0-69 --export=ALL fit_motor_cv.sh
#   sbatch --array=0-34 --export=ALL,DESC=response fit_motor_cv.sh   # one epoch
#   sbatch --array=0-69 --export=ALL,ROI_SET=preSMA fit_motor_cv.sh  # pre-SMA control
set -eo pipefail

SUBJECTS=(pil01 pil02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 \
          21 22 23 24 25 26 27 28 29 30 31 32 33 34 35)
DESCS=(gabor response)
i=${SLURM_ARRAY_TASK_ID:-0}
SUBJECT=${SUBJECT:-${SUBJECTS[$((i % 35))]}}
DESC=${DESC:-${DESCS[$((i / 35))]}}

export KERAS_BACKEND=tensorflow
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
REPO="${REPO:-$HOME/git/abstract_values}"
SCRIPT="${SCRIPT:-$REPO/abstract_values/encoding_models/fit_motor_cv.py}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

sleep $(( (i % 10) * 3 ))
echo "fit_motor_cv: sub-${SUBJECT} desc=${DESC}"
exec $HOME/data/conda/envs/abstract_values/bin/python -u "$SCRIPT" "$SUBJECT" \
    --desc "$DESC" --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue \
    --skip-existing --roi-set "${ROI_SET:-M1}" ${MODELS:+--models $MODELS}
