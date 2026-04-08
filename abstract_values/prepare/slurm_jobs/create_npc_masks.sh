#!/bin/bash
#SBATCH --job-name=create_npc_masks
#SBATCH --output=/home/gdehol/logs/create_npc_masks_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=zne.uzh

# Project NPC (Numerosity Parietal Cortex) group-level fsaverage surface labels
# to subject-specific T1w volumetric masks via get_surface_roi_mask.py.
#
# Creates NPC, NPC1, NPC2 masks (bilateral + per-hemisphere).
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=03 create_npc_masks.sh
#   sbatch --array=1-30 create_npc_masks.sh
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
SURFACE_MASKS=$BIDS_FOLDER/derivatives/surface_masks

# FreeSurfer from the fmriprep container sandbox
export FREESURFER_HOME=/shares/zne.uzh/containers/fmriprep-25.2.5/opt/freesurfer
export FS_LICENSE=$HOME/freesurfer/license.txt
export PATH=$FREESURFER_HOME/bin:$PATH

export PYTHONUNBUFFERED=1

PYTHON=$HOME/data/conda/envs/abstract_values/bin/python

echo "=== create_npc_masks: sub-${PARTICIPANT_LABEL} ses-${SESSION} ==="

# Project each NPC variant (NPC, NPC1, NPC2) to subject T1w space
for ROI in NPC NPC1 NPC2; do
    LH="${SURFACE_MASKS}/desc-${ROI}_L_space-fsaverage_hemi-lh.label.gii"
    RH="${SURFACE_MASKS}/desc-${ROI}_R_space-fsaverage_hemi-rh.label.gii"

    if [[ ! -f "$LH" ]] || [[ ! -f "$RH" ]]; then
        echo "WARNING: missing surface labels for ${ROI}, skipping"
        continue
    fi

    echo ""
    echo "--- ${ROI} ---"
    $PYTHON -u "$REPO/abstract_values/surface/get_surface_roi_mask.py" \
        "$PARTICIPANT_LABEL" "$SESSION" \
        --lh "$LH" \
        --rh "$RH" \
        --roi "$ROI" \
        --bids-folder "$BIDS_FOLDER" \
        --fmriprep-deriv "$FMRIPREP_DERIV"
done
