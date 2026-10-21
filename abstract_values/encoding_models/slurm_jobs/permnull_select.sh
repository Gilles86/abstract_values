#!/bin/bash
#SBATCH --job-name=permnull
#SBATCH --output=/home/gdehol/logs/permnull_%j.txt
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00

# Label-permutation null voxel selection (permnull_select.py): real fit + K
# within-run label shuffles of one aPRF variant on the ROI-union voxels,
# stacked into one fit; pooled p and BH q per ROI.
#
# Usage:
#   sbatch --export=ALL,PARTICIPANT_LABEL=05 permnull_select.sh
#   sbatch --export=ALL,PARTICIPANT_LABEL=05,MODEL=standard,SMOOTHED=1,N_PERM=20 permnull_select.sh

export KERAS_BACKEND=tensorflow
export TMPDIR=/scratch/gdehol

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi
MODEL="${MODEL:-session-shift}"
SMOOTHED="${SMOOTHED:-0}"
N_PERM="${N_PERM:-20}"
N_ITERATIONS="${N_ITERATIONS:-1000}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO="${REPO:-$HOME/git/abstract_values}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

ARGS=("$PARTICIPANT_LABEL" --bids-folder "$BIDS_FOLDER" --model "$MODEL"
      --n-permutations "$N_PERM" --n-iterations "$N_ITERATIONS")
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

echo "permnull: sub-${PARTICIPANT_LABEL} model=${MODEL} smoothed=${SMOOTHED} K=${N_PERM}"

. $HOME/init_conda.sh
conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/permnull_select.py" "${ARGS[@]}"
