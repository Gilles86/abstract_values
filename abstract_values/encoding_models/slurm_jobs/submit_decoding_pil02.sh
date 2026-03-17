#!/usr/bin/env bash
# Submit full factorial decoding for sub-pil02 ses-1.
# Runs for both fmriprep-noflair and fmriprep-t2w.
#
# Usage:
#   bash submit_decoding_pil02.sh
#
# Full factorial:
#   5 masks × 4 nvoxels × 2 noise × 2 smoothed × 2 lambda
#   = 160 gabor jobs + 160 value jobs = 320 jobs total

set -e

BIDS=/shares/zne.uzh/gdehol/ds-abstractvalue
MASKS_DIR=$BIDS/derivatives/masks/sub-pil02/ses-1/anat
SCRIPT_DIR=$(dirname "$0")

declare -A MASKS=(
    [BensonV1]=$MASKS_DIR/sub-pil02_ses-1_space-T1w_hemi-LR_desc-BensonV1_mask.nii.gz
    [BensonV2]=$MASKS_DIR/sub-pil02_ses-1_space-T1w_hemi-LR_desc-BensonV2_mask.nii.gz
    [BensonV3]=$MASKS_DIR/sub-pil02_ses-1_space-T1w_hemi-LR_desc-BensonV3_mask.nii.gz
    [NPCl]=$MASKS_DIR/sub-pil02_ses-1_space-T1w_desc-NPCl_mask.nii.gz
    [NPCr]=$MASKS_DIR/sub-pil02_ses-1_space-T1w_desc-NPCr_mask.nii.gz
)

N_JOBS=0
for DERIV in fmriprep-t2w; do
    for MASK_DESC in BensonV1 BensonV2 BensonV3 NPCl NPCr; do
        MASK=${MASKS[$MASK_DESC]}
        for NVOX in 50 100 250 500; do
            for SPHERICAL in 0 1; do
                for SMOOTHED in 0 1; do
                    for LAMBD in 0.0 0.1; do

                        EXPORT="PARTICIPANT_LABEL=pil02,SESSION=1"
                        EXPORT+=",FMRIPREP_DERIV=$DERIV"
                        EXPORT+=",MASK=$MASK,MASK_DESC=$MASK_DESC"
                        EXPORT+=",N_VOXELS=$NVOX"
                        EXPORT+=",SPHERICAL=$SPHERICAL"
                        EXPORT+=",SMOOTHED=$SMOOTHED"
                        EXPORT+=",LAMBD=$LAMBD"

                        sbatch --export="$EXPORT" "$SCRIPT_DIR/decode_gabor.sh"
                        sbatch --export="$EXPORT" "$SCRIPT_DIR/decode_value.sh"
                        N_JOBS=$((N_JOBS + 2))

                    done
                done
            done
        done
    done
done

echo "Submitted $N_JOBS jobs."
