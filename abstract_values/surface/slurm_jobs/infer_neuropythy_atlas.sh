#!/bin/bash
#SBATCH --job-name=neuropythy_atlas
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/neuropythy_atlas_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#
# Benson-14 + Wang-15 anatomical atlases on the subject's own surface, then
# projected to fsaverage. CPU-only and cheap (a few minutes per subject) —
# do NOT request a GPU.
#
# Needs real memory: neuropythy builds the full mesh, and the login node's
# limit is not enough (it dies with a numpy MemoryError on a 327k-face array).
# Always run this through SLURM, never on the login node.
#
# Usage:
#   sbatch --array=0-28 abstract_values/surface/slurm_jobs/infer_neuropythy_atlas.sh
#   sbatch --export=PARTICIPANT_LABEL=29 abstract_values/surface/slurm_jobs/infer_neuropythy_atlas.sh

set -euo pipefail

SUBJECTS=(03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 pil01 pil02)

if [ -z "${PARTICIPANT_LABEL:-}" ]; then
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo "ERROR: set PARTICIPANT_LABEL or submit with --array" >&2
        exit 2
    fi
    PARTICIPANT_LABEL="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
fi

SESSION="${SESSION:-1}"
BIDS_FOLDER="${BIDS_FOLDER:-/shares/zne.uzh/gdehol/ds-abstractvalue}"

export AV_FREESURFER_HOME=/shares/zne.uzh/containers/fmriprep-25.2.3/opt/freesurfer
export AV_FS_LICENSE=/shares/zne.uzh/containers/freesurfer/license.txt
export FREESURFER_HOME=$AV_FREESURFER_HOME
export FS_LICENSE=$AV_FS_LICENSE
export PATH="$FREESURFER_HOME/bin:$PATH"

echo "Host: $(hostname) | sub-${PARTICIPANT_LABEL} ses-${SESSION}"

cd "$HOME/git/abstract_values"
PYTHONUNBUFFERED=1 "$HOME/data/conda/envs/abstract_values/bin/python" -u \
    -m abstract_values.surface.infer_neuropythy_atlas \
    "$PARTICIPANT_LABEL" --bids_folder "$BIDS_FOLDER" --session "$SESSION"
