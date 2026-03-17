#!/usr/bin/env bash
# Submit full factorial decoding for sub-pil01 ses-1.
# Each combination is an independent SLURM job (runs in parallel).
#
# Usage:
#   bash submit_decoding_pil01.sh
#
# Full factorial:
#   5 masks × 2 nvoxels × 2 noise × 2 smoothed × 2 lambda
#   = 80 gabor jobs + 80 value jobs = 160 jobs total
#
# Currently set to nvoxels=50,500 (100 and 250 submitted separately).

set -e

BIDS=/shares/zne.uzh/gdehol/ds-abstractvalue
MASKS_DIR=$BIDS/derivatives/masks/sub-pil01/ses-1/anat
SCRIPT_DIR=$(dirname "$0")

declare -A MASKS=(
    [BensonV1]=$MASKS_DIR/sub-pil01_ses-1_space-T1w_hemi-LR_desc-BensonV1_mask.nii.gz
    [BensonV2]=$MASKS_DIR/sub-pil01_ses-1_space-T1w_hemi-LR_desc-BensonV2_mask.nii.gz
    [BensonV3]=$MASKS_DIR/sub-pil01_ses-1_space-T1w_hemi-LR_desc-BensonV3_mask.nii.gz
    [NPCl]=$MASKS_DIR/sub-pil01_ses-1_space-T1w_desc-NPCl_mask.nii.gz
    [NPCr]=$MASKS_DIR/sub-pil01_ses-1_space-T1w_desc-NPCr_mask.nii.gz
)

N_JOBS=0
for MASK_DESC in BensonV1 BensonV2 BensonV3 NPCl NPCr; do
    MASK=${MASKS[$MASK_DESC]}
    for NVOX in 50 500; do
        for SPHERICAL in 0 1; do
            for SMOOTHED in 0 1; do
                for LAMBD in 0.0 0.1; do

                    EXPORT="PARTICIPANT_LABEL=pil01,SESSION=1"
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

echo "Submitted $N_JOBS jobs."
