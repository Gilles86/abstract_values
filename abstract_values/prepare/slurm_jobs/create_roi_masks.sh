#!/bin/bash
#SBATCH --job-name=create_roi_masks
#SBATCH --output=/home/gdehol/logs/create_roi_masks_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=zne.uzh

# Create volumetric ROI masks (exvivo, vpnl, benson) from fmriprep FreeSurfer output.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=03 create_roi_masks.sh
#   sbatch --export=PARTICIPANT_LABEL=pil01 create_roi_masks.sh
#   sbatch --array=1-30 create_roi_masks.sh
#
# Optional exports:
#   SESSION           (default: 1)
#   FMRIPREP_DERIV    (default: fmriprep)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi

SESSION=${SESSION:-1}
FMRIPREP_DERIV=${FMRIPREP_DERIV:-fmriprep}

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

# FreeSurfer from the fmriprep container sandbox (no apptainer needed)
export FREESURFER_HOME=/shares/zne.uzh/containers/fmriprep-25.2.5/opt/freesurfer
export PATH=$FREESURFER_HOME/bin:$PATH
export SUBJECTS_DIR=$BIDS_FOLDER/derivatives/$FMRIPREP_DERIV/sourcedata/freesurfer

export PYTHONUNBUFFERED=1

echo "=== create_roi_masks: sub-${PARTICIPANT_LABEL} ses-${SESSION} ==="
echo "FREESURFER_HOME=$FREESURFER_HOME"
echo "FMRIPREP_DERIV=$FMRIPREP_DERIV"

$HOME/data/conda/envs/abstract_values/bin/python -u \
    $REPO/abstract_values/prepare/create_roi_masks.py \
    "$PARTICIPANT_LABEL" "$SESSION" \
    --bids-folder "$BIDS_FOLDER" \
    --fmriprep-deriv "$FMRIPREP_DERIV"
