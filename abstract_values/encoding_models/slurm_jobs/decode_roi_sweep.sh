#!/bin/bash
#SBATCH --job-name=decode_roi_sweep
#SBATCH --output=/home/gdehol/logs/decode_roi_sweep_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=zne.uzh
#SBATCH --partition=standard

# Bayesian decoding (value and/or gabor orientation) for a whole ROI SET, in
# the canonical setting  --n-voxels 100 --spherical-noise --lambd 0.1,
# unsmoothed.  One array task = one (subject, ROI-chunk); the task loops over
# its ROIs x decoders serially and skips any cell whose _pars.tsv already
# exists, so the array is freely resubmittable.
#
# Outputs (written by decode_{value,gabor}.py):
#   derivatives/decoding/{value,gabor}/sub-<S>/func/
#       sub-<S>_mask-<ROI>_nvoxels-100_noise-spherical_lambda-0.1_pars.tsv
#       ...                                                     _meta.tsv
#
# Usage:
#   # whole cohort, both decoders, 1 chunk per subject
#   sbatch --array=1-30 decode_roi_sweep.sh
#
#   # 30 subjects x 3 ROI chunks = 90 tasks
#   sbatch --array=1-90 --export=N_CHUNKS=3 decode_roi_sweep.sh
#
#   # pilot: one subject, three ROIs, both decoders
#   sbatch --export=PARTICIPANT_LABEL=pil01,ROIS="BensonV1 NPCr IPS1" \
#          --time=04:00:00 decode_roi_sweep.sh
#
# Optional exports:
#   PARTICIPANT_LABEL  explicit subject (bypasses the array index)
#   ROIS               space-separated ROI list (default: the full set below)
#   DECODERS           "value gabor" (default), or just one of them
#   N_CHUNKS           split ROIS into this many chunks (default 1). Array
#                      index maps to (subject, chunk) row-major over chunks.
#   N_VOXELS           default 100
#   LAMBD              default 0.1
#   SMOOTHED           default 0
#   FMRIPREP_DERIV     default fmriprep
#   FORCE              set to 1 to recompute cells that already have a _pars.tsv

set -o pipefail

SUBJECTS=(03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
          25 26 27 28 29 30 pil01 pil02)

# The full ROI table. Naming note:
#   BensonV1/V2/V3      -> hemi-LR entity (created by create_roi_masks.py)
#   everything else     -> no hemi entity (created by
#                          create_npc_masks.sh / create_extra_roi_masks.sh)
# NPCr is right-hemisphere only by construction; all others are bilateral.
DEFAULT_ROIS="BensonV1 BensonV2 BensonV3 hV4 LO TO1 TO2 V3a V3b \
IPS0 IPS1 IPS2 IPS3 SPL1 FEF NPC1 NPC2 NPC3 NPCr NTO NF1 NF2 NINS"

ROIS=${ROIS:-$DEFAULT_ROIS}
DECODERS=${DECODERS:-"value gabor"}
N_CHUNKS=${N_CHUNKS:-1}
N_VOXELS=${N_VOXELS:-100}
LAMBD=${LAMBD:-0.1}
SMOOTHED=${SMOOTHED:-0}
FMRIPREP_DERIV=${FMRIPREP_DERIV:-fmriprep}
FORCE=${FORCE:-0}

read -r -a ROI_ARR <<< "$ROIS"
N_ROIS=${#ROI_ARR[@]}

if [ -z "$PARTICIPANT_LABEL" ]; then
    IDX=$((SLURM_ARRAY_TASK_ID - 1))
    SUB_IDX=$((IDX / N_CHUNKS))
    CHUNK=$((IDX % N_CHUNKS))
    PARTICIPANT_LABEL=${SUBJECTS[$SUB_IDX]}
else
    CHUNK=${CHUNK:-0}
    N_CHUNKS=1
fi

# Select this task's ROIs (stride-based, so chunks are balanced).
MY_ROIS=()
for ((i = CHUNK; i < N_ROIS; i += N_CHUNKS)); do
    MY_ROIS+=("${ROI_ARR[$i]}")
done

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values
MASK_DIR=$BIDS_FOLDER/derivatives/masks/sub-${PARTICIPANT_LABEL}/anat

# Project models are TF-native; cluster keras.json defaults to jax — pin TF.
export KERAS_BACKEND=tensorflow
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"

# The env has activate.d hooks (libglib / libxml2), so activate properly
# rather than calling the env binary directly. NOT `conda run` — that buffers
# all stdout until exit and does not forward SIGTERM.
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate abstract_values
PYTHON=python

SMOOTH_LABEL=""
[ "$SMOOTHED" = "1" ] && SMOOTH_LABEL="_smoothed"
LAMBD_LABEL=""
[ "$LAMBD" != "0" ] && [ "$LAMBD" != "0.0" ] && LAMBD_LABEL="_lambda-${LAMBD}"

echo "=== decode_roi_sweep: sub-${PARTICIPANT_LABEL} chunk ${CHUNK}/${N_CHUNKS} ==="
echo "decoders: ${DECODERS}"
echo "ROIs    : ${MY_ROIS[*]}"
echo "setting : n_voxels=${N_VOXELS} lambd=${LAMBD} spherical smoothed=${SMOOTHED}"
date

FAILED=()
for ROI in "${MY_ROIS[@]}"; do
    # Resolve the mask file: Benson* masks carry a hemi-LR entity, the
    # surface-label-derived ones (NPC*, IPS*, ...) carry none.
    case "$ROI" in
        Benson*|hOc*|V1exvivo*|V2exvivo*)
            MASK="${MASK_DIR}/sub-${PARTICIPANT_LABEL}_space-T1w_hemi-LR_desc-${ROI}_mask.nii.gz" ;;
        *)
            MASK="${MASK_DIR}/sub-${PARTICIPANT_LABEL}_space-T1w_desc-${ROI}_mask.nii.gz" ;;
    esac

    if [[ ! -f "$MASK" ]]; then
        echo "WARNING: no mask for ${ROI} (${MASK}), skipping"
        FAILED+=("${ROI}:no-mask")
        continue
    fi

    for DEC in $DECODERS; do
        OUT="${BIDS_FOLDER}/derivatives/decoding/${DEC}/sub-${PARTICIPANT_LABEL}/func/sub-${PARTICIPANT_LABEL}_mask-${ROI}_nvoxels-${N_VOXELS}_noise-spherical${SMOOTH_LABEL}${LAMBD_LABEL}_pars.tsv"
        if [[ -f "$OUT" ]] && [[ "$FORCE" != "1" ]]; then
            echo "[${ROI} / ${DEC}] already done, skipping"
            continue
        fi

        SCRIPT="$REPO/abstract_values/encoding_models/decode_${DEC}.py"
        ARGS=("$PARTICIPANT_LABEL"
              --mask "$MASK"
              --mask-desc "$ROI"
              --n-voxels "$N_VOXELS"
              --lambd "$LAMBD"
              --spherical-noise
              --bids-folder "$BIDS_FOLDER"
              --fmriprep-deriv "$FMRIPREP_DERIV")
        [ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

        echo ""
        echo "=== [${ROI} / ${DEC}] starting ==="
        SECONDS=0
        $PYTHON -u "$SCRIPT" "${ARGS[@]}" || FAILED+=("${ROI}/${DEC}:error")
        echo "=== [${ROI} / ${DEC}] took ${SECONDS}s ==="
    done
done

echo ""
date
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED cells: ${FAILED[*]}"
    exit 1
fi
echo "decode_roi_sweep done for sub-${PARTICIPANT_LABEL} chunk ${CHUNK}"
