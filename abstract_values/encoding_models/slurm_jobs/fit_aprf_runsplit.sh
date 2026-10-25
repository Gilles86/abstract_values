#!/bin/bash
#SBATCH --job-name=aprf_runsplit
#SBATCH --output=/home/gdehol/logs/aprf_runsplit_%A_%a.txt
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Session-shift aPRF on half the runs (odd or even run numbers of every
# session), restricted to the ROI-union mask, for the split-half reliability
# test of the hierarchical lambda. Output: aprf-session-shift.runs-<odd|even>.
#
# One array task per (participant, half): tasks 1..2N, N = participants with a
# full session-shift fit; odd task ids -> odd runs, even -> even runs.
#   sbatch --array=1-70 --exclude=u24-chaihm0-651 fit_aprf_runsplit.sh
# Optional: MODEL (default session-shift), SMOOTHED=1.
set -eo pipefail

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values
MODEL="${MODEL:-session-shift}"
export KERAS_BACKEND=tensorflow
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"

mapfile -t SUBS < <(ls "$BIDS_FOLDER/derivatives/encoding_models/aprf-session-shift" \
                    | grep '^sub-' | sed 's/^sub-//' | sort)
i=$(( (SLURM_ARRAY_TASK_ID - 1) / 2 ))
HALF=$([ $(( SLURM_ARRAY_TASK_ID % 2 )) -eq 1 ] && echo odd || echo even)
SUB=${SUBS[$i]}
MASK=$BIDS_FOLDER/derivatives/masks/sub-$SUB/anat/sub-${SUB}_space-T1w_desc-permnullunion_mask.nii.gz

ARGS=("$SUB" --bids-folder "$BIDS_FOLDER" --model "$MODEL" --runs "$HALF" --mask "$MASK")
[ "${SMOOTHED:-0}" = "1" ] && ARGS+=(--smoothed)
echo "aprf_runsplit: sub-$SUB runs=$HALF model=$MODEL"

exec "$HOME/data/conda/envs/abstract_values/bin/python" -u \
    "$REPO/abstract_values/encoding_models/fit_aprf.py" "${ARGS[@]}"
