#!/usr/bin/env bash
# ingest_new_session.sh — end-to-end pipeline for a new MRI session.
#
# Local steps:
#   1. rsync source data  →  local BIDS sourcedata
#   2. BIDS conversion (dry-run preview, then real)
#   3. rsync BIDS session + behavior  →  cluster
#
# Cluster SLURM chain (all chained with --dependency=afterok):
#   4. fmriprep              (full subject, all sessions)
#   5. GLMsingle             (all sessions jointly, after fmriprep)
#
#   After GLMsingle (all parallel):
#   6.  fit_aprf             (standard)
#   7.  fit_aprf_cv
#   8.  fit_aprf session-shift   (only ses≥2)
#   9.  fit_aprf_shift_cv        (only ses≥2)
#  10.  fit_aprf_weighted
#  11.  fit_aprf_weighted_cv
#  12.  fit_vonmises
#  13.  fit_vonmises_cv
#  14.  decode_gabor         (per ROI in DECODE_ROIS)
#  15.  decode_value         (per ROI in DECODE_ROIS)
#  16.  compute_fisher_information  (Von Mises FI, per ROI in FI_ROIS_VONMISES)
#
#   After fit_aprf:
#  17.  compute_fisher_information_aprf  (per ROI in FI_ROIS_APRF)
#
# Prerequisites:
#   ROI masks must exist under derivatives/masks/sub-<subject>/ses-1/anat/
#   before the pipeline is submitted (run get_surface_roi_mask.py after fmriprep).
#
# Usage:
#   ./ingest_new_session.sh --subject pil02 --session 2
#   ./ingest_new_session.sh --subject pil02 --session 2 \
#       --source-dir /custom/path/ses-2
#   ./ingest_new_session.sh --subject pil02 --session 2 --dry-run
#
# Options:
#   --subject LABEL       participant label, e.g. pil02 or 01 (required)
#   --session N           session number (required)
#   --source-dir PATH     source ses-N directory (default: network drive)
#   --glmsingle-deriv L   fmriprep derivative for GLMsingle (default: fmriprep)
#   --dry-run             show BIDS conversion preview; skip all writes and cluster jobs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
BIDS_ROOT="/data/ds-abstractvalue"
NETWORK_BASE='/Volumes/g_econ_department$/projects/2026/dehollander_bedi_ruff_abstract_values/data/sourcedata/mri'
CLUSTER="sciencecluster"
CLUSTER_BIDS="/shares/zne.uzh/gdehol/ds-abstractvalue"
GLMSINGLE_DERIV="fmriprep"
DRY_RUN=0

# ROIs for decode and FI jobs — format: "DESC:HEMI" (HEMI=None omits hemi entity)
DECODE_ROIS="BensonV1:LR NPCr:None"
DECODE_N_VOXELS="100 0"
DECODE_LAMBDAS="0.0 0.1"
FI_ROIS_VONMISES="BensonV1:LR"
FI_ROIS_APRF="NPCr:None"

# ── argument parsing ──────────────────────────────────────────────────────────
SUBJECT=""
SESSION=""
SOURCE_DIR=""

usage() {
    sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject)         SUBJECT="$2";        shift 2 ;;
        --session)         SESSION="$2";        shift 2 ;;
        --source-dir)      SOURCE_DIR="$2";     shift 2 ;;
        --glmsingle-deriv) GLMSINGLE_DERIV="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1;           shift   ;;
        -h|--help)         usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

[[ -z "$SUBJECT" || -z "$SESSION" ]] && { echo "Error: --subject and --session are required." >&2; usage; }

SOURCE_DIR="${SOURCE_DIR:-${NETWORK_BASE}/sub-${SUBJECT}/ses-${SESSION}}"
BIDS_SOURCEDATA="${BIDS_ROOT}/sourcedata/mri/sub-${SUBJECT}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── step 1: rsync source → local BIDS sourcedata ─────────────────────────────
log "Step 1: rsync ${SOURCE_DIR}  →  ${BIDS_SOURCEDATA}/"
if [[ "$DRY_RUN" -eq 1 ]]; then
    rsync -av --dry-run "${SOURCE_DIR}" "${BIDS_SOURCEDATA}/"
else
    rsync -av "${SOURCE_DIR}" "${BIDS_SOURCEDATA}/"
fi

# ── step 2: BIDS conversion ───────────────────────────────────────────────────
log "Step 2: BIDS conversion — dry-run preview"
conda run -n abstract_values python "${SCRIPT_DIR}/fix_and_move_bids.py" \
    --subject "${SUBJECT}" --session "${SESSION}" --dry-run

if [[ "$DRY_RUN" -eq 1 ]]; then
    log "Dry-run mode — stopping before any writes."
    exit 0
fi

log "Step 2: BIDS conversion — writing"
conda run -n abstract_values python "${SCRIPT_DIR}/fix_and_move_bids.py" \
    --subject "${SUBJECT}" --session "${SESSION}"

# ── step 3a: rsync BIDS session → cluster ────────────────────────────────────
log "Step 3a: rsync ${BIDS_ROOT}/sub-${SUBJECT}/ses-${SESSION}/  →  cluster"
rsync -av \
    "${BIDS_ROOT}/sub-${SUBJECT}/ses-${SESSION}/" \
    "${CLUSTER}:${CLUSTER_BIDS}/sub-${SUBJECT}/ses-${SESSION}/"

# ── step 3b: rsync behavior sourcedata → cluster ──────────────────────────────
BEHAVIOR_SRC="${BIDS_ROOT}/sourcedata/behavior/sub-${SUBJECT}/ses-${SESSION}"
BEHAVIOR_DST="${CLUSTER}:${CLUSTER_BIDS}/sourcedata/behavior/sub-${SUBJECT}/ses-${SESSION}/"
if [[ -d "${BEHAVIOR_SRC}" ]]; then
    log "Step 3b: rsync behavior sourcedata  →  cluster"
    rsync -av "${BEHAVIOR_SRC}/" "${BEHAVIOR_DST}"
else
    log "Step 3b: no behavior sourcedata found at ${BEHAVIOR_SRC}, skipping"
fi

# ── steps 4–8: SLURM chain on cluster ────────────────────────────────────────
log "Steps 4–17: submitting SLURM chain on ${CLUSTER}"

SLURM_OUTPUT=$(ssh "${CLUSTER}" bash <<EOF
set -euo pipefail

REPO=\$HOME/git/abstract_values

git -C "\$REPO" pull --ff-only

FMRIPREP_DIR="\$REPO/abstract_values/prepare/cluster_preproc"
PREPARE_DIR="\$REPO/abstract_values/prepare/slurm_jobs"
GLMSINGLE_DIR="\$REPO/abstract_values/glm/slurm_jobs"
APRF_DIR="\$REPO/abstract_values/encoding_models/slurm_jobs"

# 4. fmriprep — full subject (all sessions; reuses nipype cache for prior sessions)
FMRIPREP_JOB=\$(sbatch --parsable \
    --export=PARTICIPANT_LABEL=${SUBJECT} \
    "\$FMRIPREP_DIR/fmriprep.sh")
echo "fmriprep:\$FMRIPREP_JOB"

# 4b. ROI masks — after fmriprep (needed by encoding models & decoding)
MASKS_JOB=\$(sbatch --parsable \
    --dependency=afterok:\$FMRIPREP_JOB \
    --export=PARTICIPANT_LABEL=${SUBJECT},FMRIPREP_DERIV=${GLMSINGLE_DERIV} \
    "\$PREPARE_DIR/create_roi_masks.sh")
echo "create_masks:\$MASKS_JOB"

# 4c. NPC masks — after fmriprep (project NPC group atlas to subject T1w)
NPC_MASKS_JOB=\$(sbatch --parsable \
    --dependency=afterok:\$FMRIPREP_JOB \
    --export=PARTICIPANT_LABEL=${SUBJECT},FMRIPREP_DERIV=${GLMSINGLE_DERIV} \
    "\$PREPARE_DIR/create_npc_masks.sh")
echo "create_npc_masks:\$NPC_MASKS_JOB"

# 5a. GLMsingle — all sessions jointly, unsmoothed
GLMSINGLE_JOB=\$(sbatch --parsable \
    --dependency=afterok:\$FMRIPREP_JOB \
    --export=PARTICIPANT_LABEL=${SUBJECT},FMRIPREP_DERIV=${GLMSINGLE_DERIV} \
    "\$GLMSINGLE_DIR/fit_glmsingle.sh")
echo "glmsingle:\$GLMSINGLE_JOB"

# 5b. GLMsingle — all sessions jointly, smoothed (parallel with 5a)
GLMSINGLE_S_JOB=\$(sbatch --parsable \
    --dependency=afterok:\$FMRIPREP_JOB \
    --export=PARTICIPANT_LABEL=${SUBJECT},FMRIPREP_DERIV=${GLMSINGLE_DERIV},SMOOTHED=1 \
    "\$GLMSINGLE_DIR/fit_glmsingle.sh")
echo "glmsingle_smoothed:\$GLMSINGLE_S_JOB"

# Steps 6–17 run twice: unsmoothed (SMOOTHED=0) then smoothed (SMOOTHED=1)
MASK_BASE="${CLUSTER_BIDS}/derivatives/masks/sub-${SUBJECT}/anat"
for smoothed in 0 1; do
    smooth_export=""
    smooth_label=""
    glmsingle_dep="\$GLMSINGLE_JOB"
    if [[ "\$smoothed" = "1" ]]; then
        smooth_export=",SMOOTHED=1"
        smooth_label="_smoothed"
        glmsingle_dep="\$GLMSINGLE_S_JOB"
    fi

    # 6. fit_aprf standard — all sessions
    APRF_JOB=\$(sbatch --parsable \
        --dependency=afterok:\${glmsingle_dep} \
        --export=PARTICIPANT_LABEL=${SUBJECT}\${smooth_export} \
        "\$APRF_DIR/fit_aprf.sh")
    echo "fit_aprf\${smooth_label}:\$APRF_JOB"

    # 7. fit_aprf_cv
    APRF_CV_JOB=\$(sbatch --parsable \
        --dependency=afterok:\${glmsingle_dep} \
        --export=PARTICIPANT_LABEL=${SUBJECT}\${smooth_export} \
        "\$APRF_DIR/fit_aprf_cv.sh")
    echo "fit_aprf_cv\${smooth_label}:\$APRF_CV_JOB"

    # 8+9. session-shift model (requires ≥2 sessions)
    if [[ ${SESSION} -ge 2 ]]; then
        APRF_SHIFT_JOB=\$(sbatch --parsable \
            --dependency=afterok:\${glmsingle_dep} \
            --export=PARTICIPANT_LABEL=${SUBJECT},MODEL=session-shift\${smooth_export} \
            "\$APRF_DIR/fit_aprf.sh")
        echo "fit_aprf_shift\${smooth_label}:\$APRF_SHIFT_JOB"

        APRF_SHIFT_CV_JOB=\$(sbatch --parsable \
            --dependency=afterok:\${glmsingle_dep} \
            --export=PARTICIPANT_LABEL=${SUBJECT}\${smooth_export} \
            "\$APRF_DIR/fit_aprf_shift_cv.sh")
        echo "fit_aprf_shift_cv\${smooth_label}:\$APRF_SHIFT_CV_JOB"
    fi

    # 10. fit_aprf_weighted
    APRF_W_JOB=\$(sbatch --parsable \
        --dependency=afterok:\${glmsingle_dep} \
        --export=PARTICIPANT_LABEL=${SUBJECT}\${smooth_export} \
        "\$APRF_DIR/fit_aprf_weighted.sh")
    echo "fit_aprf_weighted\${smooth_label}:\$APRF_W_JOB"

    # 11. fit_aprf_weighted_cv
    APRF_W_CV_JOB=\$(sbatch --parsable \
        --dependency=afterok:\${glmsingle_dep} \
        --export=PARTICIPANT_LABEL=${SUBJECT}\${smooth_export} \
        "\$APRF_DIR/fit_aprf_weighted_cv.sh")
    echo "fit_aprf_weighted_cv\${smooth_label}:\$APRF_W_CV_JOB"

    # 12. fit_vonmises
    VONMISES_JOB=\$(sbatch --parsable \
        --dependency=afterok:\${glmsingle_dep} \
        --export=PARTICIPANT_LABEL=${SUBJECT}\${smooth_export} \
        "\$APRF_DIR/fit_vonmises.sh")
    echo "fit_vonmises\${smooth_label}:\$VONMISES_JOB"

    # 13. fit_vonmises_cv
    VONMISES_CV_JOB=\$(sbatch --parsable \
        --dependency=afterok:\${glmsingle_dep} \
        --export=PARTICIPANT_LABEL=${SUBJECT}\${smooth_export} \
        "\$APRF_DIR/fit_vonmises_cv.sh")
    echo "fit_vonmises_cv\${smooth_label}:\$VONMISES_CV_JOB"

    # 14+15. decode_gabor + decode_value — per ROI × n_voxels × lambda
    # Depend on both GLMsingle AND mask creation
    for roi_hemi in ${DECODE_ROIS}; do
        desc=\${roi_hemi%%:*}
        hemi=\${roi_hemi##*:}
        if [[ "\$hemi" = "None" ]]; then
            mask_file="\${MASK_BASE}/sub-${SUBJECT}_space-T1w_desc-\${desc}_mask.nii.gz"
            mask_dep="\$NPC_MASKS_JOB"
        else
            mask_file="\${MASK_BASE}/sub-${SUBJECT}_space-T1w_hemi-\${hemi}_desc-\${desc}_mask.nii.gz"
            mask_dep="\$MASKS_JOB"
        fi

        for nv in ${DECODE_N_VOXELS}; do
            for lam in ${DECODE_LAMBDAS}; do
                nv_label=""
                [[ "\$nv" != "100" ]] && nv_label="_nv\${nv}"
                lam_label=""
                [[ "\$lam" != "0.0" ]] && lam_label="_l\${lam}"

                # n_voxels=0 value decode needs more time (nested CV)
                time_gabor=""
                time_value=""
                if [[ "\$nv" = "0" ]]; then
                    time_value="--time=04:00:00"
                fi

                DECODE_GABOR_JOB=\$(sbatch --parsable \$time_gabor \
                    --dependency=afterok:\${glmsingle_dep}:\${mask_dep} \
                    --export=PARTICIPANT_LABEL=${SUBJECT},MASK=\$mask_file,MASK_DESC=\${desc},N_VOXELS=\${nv},LAMBD=\${lam}\${smooth_export} \
                    "\$APRF_DIR/decode_gabor.sh")
                echo "decode_gabor_\${desc}\${nv_label}\${lam_label}\${smooth_label}:\$DECODE_GABOR_JOB"

                DECODE_VALUE_JOB=\$(sbatch --parsable \$time_value \
                    --dependency=afterok:\${glmsingle_dep}:\${mask_dep} \
                    --export=PARTICIPANT_LABEL=${SUBJECT},MASK=\$mask_file,MASK_DESC=\${desc},N_VOXELS=\${nv},LAMBD=\${lam}\${smooth_export} \
                    "\$APRF_DIR/decode_value.sh")
                echo "decode_value_\${desc}\${nv_label}\${lam_label}\${smooth_label}:\$DECODE_VALUE_JOB"
            done
        done
    done

    # 16. compute_fisher_information (Von Mises) — per ROI in FI_ROIS_VONMISES
    # Depend on both GLMsingle AND mask creation
    for roi_hemi in ${FI_ROIS_VONMISES}; do
        desc=\${roi_hemi%%:*}
        hemi=\${roi_hemi##*:}
        if [[ "\$hemi" = "None" ]]; then
            mask_dep="\$NPC_MASKS_JOB"
        else
            mask_dep="\$MASKS_JOB"
        fi

        FI_VONMISES_JOB=\$(sbatch --parsable \
            --dependency=afterok:\${glmsingle_dep}:\${mask_dep} \
            --export=PARTICIPANT_LABEL=${SUBJECT},ROI=\${desc},HEMI=\${hemi}\${smooth_export} \
            "\$APRF_DIR/compute_fisher_information.sh")
        echo "fi_vonmises_\${desc}\${smooth_label}:\$FI_VONMISES_JOB"
    done

    # 17. compute_fisher_information_aprf — per ROI in FI_ROIS_APRF, after fit_aprf + masks
    for roi_hemi in ${FI_ROIS_APRF}; do
        desc=\${roi_hemi%%:*}
        hemi=\${roi_hemi##*:}
        if [[ "\$hemi" = "None" ]]; then
            mask_dep="\$NPC_MASKS_JOB"
        else
            mask_dep="\$MASKS_JOB"
        fi

        FI_APRF_JOB=\$(sbatch --parsable \
            --dependency=afterok:\$APRF_JOB:\${mask_dep} \
            --export=PARTICIPANT_LABEL=${SUBJECT},ROI=\${desc},HEMI=\${hemi}\${smooth_export} \
            "\$APRF_DIR/compute_fisher_information_aprf.sh")
        echo "fi_aprf_\${desc}\${smooth_label}:\$FI_APRF_JOB"
    done

done  # smoothed loop
EOF
)

# ── summary ───────────────────────────────────────────────────────────────────
log "All done. SLURM chain:"
echo "$SLURM_OUTPUT" | while IFS=: read -r name jobid; do
    printf "  %-24s  job %s\n" "$name" "$jobid"
done
