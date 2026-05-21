#!/bin/bash
# Submit the 32-job decode sweep that extends the existing voxel-count
# sweep with the two whole-brain R²-mixture voxel-selection criteria:
#
#   8 subjects × 2 decoders (value, gabor) × 2 criteria (fdr 0.05, psig 0.95)
#
# decode_value uses the cached aprf mixture (no dependency).
# decode_gabor uses the vonmises mixture, which is computed by job
# $VONMISES_JOB — pass it as the first arg; those 16 jobs gate on
# --dependency=afterok:$VONMISES_JOB.
#
# Walltime 6h (nested-CV path); λ=0.1 (matches existing sweep).
#
# Usage:
#   sbatch abstract_values/encoding_models/slurm_jobs/compute_r2_mixture_vonmises.sh
#   # …note the job id; then:
#   bash abstract_values/encoding_models/slurm_jobs/submit_decode_sweep_mixture.sh <vonmises_jobid>

set -e

VONMISES_JOB="${1:-}"
if [ -z "$VONMISES_JOB" ]; then
    echo "Usage: $0 <vonmises_mixture_jobid>"
    exit 1
fi

REPO=$HOME/git/abstract_values
BIDS=/shares/zne.uzh/gdehol/ds-abstractvalue
DECODE_DIR=$REPO/abstract_values/encoding_models/slurm_jobs

SUBJECTS=(03 04 05 06 08 09 pil01 pil02)

SBATCH_OPTS="--time=06:00:00 --account=zne.uzh"

declare -a JOB_IDS=()

for SUBJ in "${SUBJECTS[@]}"; do
    V1_MASK=$BIDS/derivatives/masks/sub-$SUBJ/anat/sub-${SUBJ}_space-T1w_hemi-LR_desc-BensonV1_mask.nii.gz
    NPCR_MASK=$BIDS/derivatives/masks/sub-$SUBJ/anat/sub-${SUBJ}_space-T1w_desc-NPCr_mask.nii.gz

    # ── decode_value: aprf mixture (already cached) ─────────────────────────
    for CRIT in fdr05 psig95; do
        if [ "$CRIT" = "fdr05" ]; then
            EXPORT="PARTICIPANT_LABEL=$SUBJ,MASK=$NPCR_MASK,MASK_DESC=NPCr,LAMBD=0.1,FDR_ALPHA=0.05"
        else
            EXPORT="PARTICIPANT_LABEL=$SUBJ,MASK=$NPCR_MASK,MASK_DESC=NPCr,LAMBD=0.1,P_SIGNAL_THR=0.95"
        fi
        JID=$(sbatch --parsable $SBATCH_OPTS \
            --export="$EXPORT" \
            $DECODE_DIR/decode_value.sh)
        echo "value/$SUBJ/$CRIT -> $JID"
        JOB_IDS+=("$JID")
    done

    # ── decode_gabor: vonmises mixture (depends on $VONMISES_JOB) ──────────
    for CRIT in fdr05 psig95; do
        if [ "$CRIT" = "fdr05" ]; then
            EXPORT="PARTICIPANT_LABEL=$SUBJ,MASK=$V1_MASK,MASK_DESC=BensonV1,LAMBD=0.1,FDR_ALPHA=0.05"
        else
            EXPORT="PARTICIPANT_LABEL=$SUBJ,MASK=$V1_MASK,MASK_DESC=BensonV1,LAMBD=0.1,P_SIGNAL_THR=0.95"
        fi
        JID=$(sbatch --parsable $SBATCH_OPTS \
            --dependency=afterok:$VONMISES_JOB \
            --export="$EXPORT" \
            $DECODE_DIR/decode_gabor.sh)
        echo "gabor/$SUBJ/$CRIT -> $JID  (afterok:$VONMISES_JOB)"
        JOB_IDS+=("$JID")
    done
done

echo
echo "Submitted ${#JOB_IDS[@]} jobs:"
printf '  %s\n' "${JOB_IDS[@]}"
