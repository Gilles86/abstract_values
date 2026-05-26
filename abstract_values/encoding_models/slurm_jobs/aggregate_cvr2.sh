#!/bin/bash
#SBATCH --job-name=aggregate_cvr2
#SBATCH --output=/home/gdehol/logs/aggregate_cvr2_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00

# Aggregate per-run leave-one-run-out cvR² NIfTIs into a single mean cvR²
# volume with proper float32 dtype (works around the fmriprep-mask
# quantisation trap; see abstract_values/encoding_models/aggregate_cvr2.py).
#
# Required (via --export):
#   PARTICIPANT_LABEL  subject label without sub-
#   MODEL_DIR          on-disk CV dir name, e.g.
#                        aprf.cv | aprf-shift.cv | aprf-weighted.cv | vonmises.cv
#
# Optional:
#   SMOOTHED           "1" to aggregate the _smoothed variant (default: 0)

set -euo pipefail

if [ -z "${PARTICIPANT_LABEL:-}" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

MODEL_DIR="${MODEL_DIR:?MODEL_DIR must be set (e.g. aprf.cv, vonmises.cv)}"
SMOOTHED="${SMOOTHED:-0}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    --subject     "$PARTICIPANT_LABEL"
    --model-dir   "$MODEL_DIR"
    --bids-folder "$BIDS_FOLDER"
)
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

echo "aggregate_cvr2: sub-${PARTICIPANT_LABEL}  model_dir=${MODEL_DIR}  smoothed=${SMOOTHED}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    -m abstract_values.encoding_models.aggregate_cvr2 \
    "${ARGS[@]}"
