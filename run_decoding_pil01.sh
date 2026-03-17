#!/usr/bin/env bash
# Decode gabor orientation + abstract value for sub-pil01 ses-1.
# ROI masks only (V1/V2/V3). Uses new braincoder (abstract_values env).
set -e

REPO=/Users/gdehol/git/abstract_values
CONDA=/Users/gdehol/mambaforge/bin/conda
PYTHON="PYTHONUNBUFFERED=1 $CONDA run -n abstract_values python -u"
SCRIPTS=$REPO/abstract_values
LOG=$REPO/logs/decoding_pil01_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"
V1=/data/ds-abstractvalue/derivatives/masks/sub-pil01/ses-1/anat/sub-pil01_ses-1_space-T1w_hemi-LR_desc-BensonV1_mask.nii.gz
V2=/data/ds-abstractvalue/derivatives/masks/sub-pil01/ses-1/anat/sub-pil01_ses-1_space-T1w_hemi-LR_desc-BensonV2_mask.nii.gz
V3=/data/ds-abstractvalue/derivatives/masks/sub-pil01/ses-1/anat/sub-pil01_ses-1_space-T1w_hemi-LR_desc-BensonV3_mask.nii.gz

for MASK_PATH in $V1 $V2 $V3; do
    MASK_DESC=$(echo $MASK_PATH | grep -o 'desc-[^_]*' | sed 's/desc-//')

    echo "======================================================"
    echo " Gabor decoding — $MASK_DESC"
    echo "======================================================"
    $PYTHON $SCRIPTS/encoding_models/decode_gabor.py pil01 --sessions 1 \
        --mask $MASK_PATH --mask-desc $MASK_DESC
    $PYTHON $SCRIPTS/encoding_models/decode_gabor.py pil01 --sessions 1 \
        --mask $MASK_PATH --mask-desc $MASK_DESC --smoothed
    $PYTHON $SCRIPTS/encoding_models/decode_gabor.py pil01 --sessions 1 \
        --mask $MASK_PATH --mask-desc $MASK_DESC --spherical-noise
    $PYTHON $SCRIPTS/encoding_models/decode_gabor.py pil01 --sessions 1 \
        --mask $MASK_PATH --mask-desc $MASK_DESC --smoothed --spherical-noise

    echo "======================================================"
    echo " Value decoding — $MASK_DESC"
    echo "======================================================"
    $PYTHON $SCRIPTS/encoding_models/decode_value.py pil01 --sessions 1 \
        --mask $MASK_PATH --mask-desc $MASK_DESC
    $PYTHON $SCRIPTS/encoding_models/decode_value.py pil01 --sessions 1 \
        --mask $MASK_PATH --mask-desc $MASK_DESC --smoothed
    $PYTHON $SCRIPTS/encoding_models/decode_value.py pil01 --sessions 1 \
        --mask $MASK_PATH --mask-desc $MASK_DESC --spherical-noise
    $PYTHON $SCRIPTS/encoding_models/decode_value.py pil01 --sessions 1 \
        --mask $MASK_PATH --mask-desc $MASK_DESC --smoothed --spherical-noise
done

echo "======================================================"
echo " DONE — executing analysis notebook"
echo "======================================================"
$CONDA run -n abstract_values jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 \
    $REPO/notebooks/decode_analysis.ipynb
echo "Notebook done."
