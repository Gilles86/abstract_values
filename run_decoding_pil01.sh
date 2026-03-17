#!/usr/bin/env bash
# Decode gabor orientation + abstract value for sub-pil01 ses-1.
# Full factorial: mask × nvoxels × noise × smoothed × lambda
set -e

REPO=/Users/gdehol/git/abstract_values
CONDA=/Users/gdehol/mambaforge/bin/conda
SCRIPTS=$REPO/abstract_values
LOG=$REPO/logs/decoding_pil01_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"

MASKS_DIR=/data/ds-abstractvalue/derivatives/masks/sub-pil01/ses-1/anat

V1=$MASKS_DIR/sub-pil01_ses-1_space-T1w_hemi-LR_desc-BensonV1_mask.nii.gz
V2=$MASKS_DIR/sub-pil01_ses-1_space-T1w_hemi-LR_desc-BensonV2_mask.nii.gz
V3=$MASKS_DIR/sub-pil01_ses-1_space-T1w_hemi-LR_desc-BensonV3_mask.nii.gz
NPCL=$MASKS_DIR/sub-pil01_ses-1_space-T1w_desc-NPCl_mask.nii.gz
NPCR=$MASKS_DIR/sub-pil01_ses-1_space-T1w_desc-NPCr_mask.nii.gz

for MASK_PATH in $V1 $V2 $V3 $NPCL $NPCR; do
    MASK_DESC=$(basename $MASK_PATH | grep -o 'desc-[^_]*' | sed 's/desc-//')

    for NVOX in 100 250; do
        for NOISE in "" "--spherical-noise"; do
            for SMOOTH in "" "--smoothed"; do
                for LAMBD in 0.0 0.1; do

                    LAMBD_ARG=$([ "$LAMBD" = "0.0" ] && echo "" || echo "--lambd $LAMBD")

                    echo "======================================================"
                    echo " Gabor  mask=$MASK_DESC  nvox=$NVOX  noise=${NOISE:---full}  smooth=${SMOOTH:-raw}  lambda=$LAMBD"
                    echo "======================================================"
                    env PYTHONUNBUFFERED=1 $CONDA run --no-capture-output -n abstract_values python -u \
                        $SCRIPTS/encoding_models/decode_gabor.py pil01 --sessions 1 \
                        --mask $MASK_PATH --mask-desc $MASK_DESC \
                        --n-voxels $NVOX $NOISE $SMOOTH $LAMBD_ARG

                    echo "======================================================"
                    echo " Value  mask=$MASK_DESC  nvox=$NVOX  noise=${NOISE:---full}  smooth=${SMOOTH:-raw}  lambda=$LAMBD"
                    echo "======================================================"
                    env PYTHONUNBUFFERED=1 $CONDA run --no-capture-output -n abstract_values python -u \
                        $SCRIPTS/encoding_models/decode_value.py pil01 --sessions 1 \
                        --mask $MASK_PATH --mask-desc $MASK_DESC \
                        --n-voxels $NVOX $NOISE $SMOOTH $LAMBD_ARG

                done
            done
        done
    done
done

echo "======================================================"
echo " DONE — executing analysis notebook"
echo "======================================================"
$CONDA run -n abstract_values jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1200 \
    $REPO/notebooks/decode_analysis.ipynb
echo "Notebook done."
