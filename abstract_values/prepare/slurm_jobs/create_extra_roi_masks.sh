#!/bin/bash
#SBATCH --job-name=create_extra_roi_masks
#SBATCH --output=/home/gdehol/logs/create_extra_roi_masks_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=zne.uzh
#SBATCH --partition=standard

# Project the *extended* fsaverage ROI set (visual + parietal + frontal +
# number-selective areas) to subject-specific T1w volumetric masks via
# get_surface_roi_mask.py.
#
# Writes, per ROI:  sub-<S>_space-T1w_desc-<ROI>_mask.nii.gz   (bilateral)
#                   sub-<S>_space-T1w_desc-<ROI>l_mask.nii.gz  (left)
#                   sub-<S>_space-T1w_desc-<ROI>r_mask.nii.gz  (right)
# i.e. the hemi=None entity convention used by NPC/NPCl/NPCr — NOT the
# hemi-LR convention used by the Benson/exvivo masks from create_roi_masks.py.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=pil01 create_extra_roi_masks.sh
#   sbatch --array=1-30 create_extra_roi_masks.sh     # whole cohort
#
# Optional exports:
#   SESSION           (default: 1)
#   FMRIPREP_DERIV    (default: fmriprep)
#   ROIS              space-separated override of the ROI list
#   FORCE             set to 1 to rebuild masks that already exist

set -o pipefail

# Cohort, indexed by SLURM_ARRAY_TASK_ID (1-based). sub-01/sub-02 do not exist
# on the MRI side — they are sub-pil01/sub-pil02.
SUBJECTS=(03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
          25 26 27 28 29 30 pil01 pil02)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}
fi

SESSION=${SESSION:-1}
FMRIPREP_DERIV=${FMRIPREP_DERIV:-fmriprep}
FORCE=${FORCE:-0}

# ROIs that do NOT already have a volumetric mask. V1/V2/V3 are covered by
# BensonV1/2/3 (hemi-LR) and NPC/NPC1/NPC2/NPCr already exist, so they are
# deliberately absent here.
ROIS=${ROIS:-"hV4 LO TO1 TO2 V3a V3b IPS0 IPS1 IPS2 IPS3 SPL1 FEF NPC3 NTO NF1 NF2 NINS"}

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values
SURFACE_MASKS=$BIDS_FOLDER/derivatives/surface_masks
MASK_DIR=$BIDS_FOLDER/derivatives/masks/sub-${PARTICIPANT_LABEL}/anat

# FreeSurfer from the fmriprep container sandbox
export FREESURFER_HOME=/shares/zne.uzh/containers/fmriprep-25.2.5/opt/freesurfer
export FS_LICENSE=$HOME/freesurfer/license.txt
export PATH=$FREESURFER_HOME/bin:$PATH

export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"

PYTHON=$HOME/data/conda/envs/abstract_values/bin/python

echo "=== create_extra_roi_masks: sub-${PARTICIPANT_LABEL} ses-${SESSION} ==="
echo "ROIs: ${ROIS}"
date

FAILED=()
for ROI in $ROIS; do
    LH="${SURFACE_MASKS}/desc-${ROI}_L_space-fsaverage_hemi-lh.label.gii"
    RH="${SURFACE_MASKS}/desc-${ROI}_R_space-fsaverage_hemi-rh.label.gii"

    if [[ ! -f "$LH" ]] || [[ ! -f "$RH" ]]; then
        echo "WARNING: missing surface labels for ${ROI}, skipping"
        FAILED+=("${ROI}:no-label")
        continue
    fi

    OUT="${MASK_DIR}/sub-${PARTICIPANT_LABEL}_space-T1w_desc-${ROI}_mask.nii.gz"
    if [[ -f "$OUT" ]] && [[ "$FORCE" != "1" ]]; then
        echo "--- ${ROI}: already exists, skipping ---"
        continue
    fi

    echo ""
    echo "--- ${ROI} ---"
    SECONDS=0
    $PYTHON -u "$REPO/abstract_values/surface/get_surface_roi_mask.py" \
        "$PARTICIPANT_LABEL" "$SESSION" \
        --lh "$LH" \
        --rh "$RH" \
        --roi "$ROI" \
        --bids-folder "$BIDS_FOLDER" \
        --fmriprep-deriv "$FMRIPREP_DERIV" \
        || FAILED+=("${ROI}:error")
    echo "--- ${ROI} took ${SECONDS}s ---"
done

echo ""
date
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED ROIs: ${FAILED[*]}"
    exit 1
fi
echo "All ROIs done for sub-${PARTICIPANT_LABEL}"
