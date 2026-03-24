#!/usr/bin/env bash
# ingest_new_session.sh — end-to-end pipeline for a new MRI session.
#
# Local steps:
#   1. rsync source data  →  local BIDS sourcedata
#   2. BIDS conversion (dry-run preview, then real)
#   3. rsync BIDS session  →  cluster
#
# Cluster SLURM chain (all chained with --dependency=afterok):
#   4. fmriprep              (full subject, all sessions)
#   5. GLMsingle             (all sessions jointly, after fmriprep)
#   6.  fit_aprf             (standard)            ─┐
#   7.  fit_aprf_cv          (standard CV)          │
#   8.  fit_aprf session-shift   (only ses≥2)       │
#   9.  fit_aprf_shift_cv        (only ses≥2)       ├─ all parallel after GLMsingle
#  10.  fit_aprf_weighted                           │
#  11.  fit_aprf_weighted_cv                        │
#  12.  fit_vonmises                                │
#  13.  fit_vonmises_cv                            ─┘
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
#   --glmsingle-deriv L   fmriprep derivative for GLMsingle (default: fmriprep-flair)
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
log "Steps 4–13: submitting SLURM chain on ${CLUSTER}"

SLURM_OUTPUT=$(ssh "${CLUSTER}" bash <<EOF
set -euo pipefail

REPO=\$HOME/git/abstract_values

git -C "\$REPO" pull --ff-only

FMRIPREP_DIR="\$REPO/abstract_values/prepare/cluster_preproc"
GLMSINGLE_DIR="\$REPO/abstract_values/glm/slurm_jobs"
APRF_DIR="\$REPO/abstract_values/encoding_models/slurm_jobs"

# 4. fmriprep — full subject (all sessions; reuses nipype cache for prior sessions)
FMRIPREP_JOB=\$(sbatch --parsable \
    --export=PARTICIPANT_LABEL=${SUBJECT} \
    "\$FMRIPREP_DIR/fmriprep.sh")
echo "fmriprep:\$FMRIPREP_JOB"

# 5. GLMsingle — all sessions jointly (better noise estimates)
GLMSINGLE_JOB=\$(sbatch --parsable \
    --dependency=afterok:\$FMRIPREP_JOB \
    --export=PARTICIPANT_LABEL=${SUBJECT},FMRIPREP_DERIV=${GLMSINGLE_DERIV} \
    "\$GLMSINGLE_DIR/fit_glmsingle.sh")
echo "glmsingle:\$GLMSINGLE_JOB"

# 6. fit_aprf standard — all sessions
APRF_JOB=\$(sbatch --parsable \
    --dependency=afterok:\$GLMSINGLE_JOB \
    --export=PARTICIPANT_LABEL=${SUBJECT} \
    "\$APRF_DIR/fit_aprf.sh")
echo "fit_aprf:\$APRF_JOB"

# 7. fit_aprf_cv — all sessions (parallel with 6)
APRF_CV_JOB=\$(sbatch --parsable \
    --dependency=afterok:\$GLMSINGLE_JOB \
    --export=PARTICIPANT_LABEL=${SUBJECT} \
    "\$APRF_DIR/fit_aprf_cv.sh")
echo "fit_aprf_cv:\$APRF_CV_JOB"

# 8+9. session-shift model (requires ≥2 sessions)
if [[ ${SESSION} -ge 2 ]]; then
    APRF_SHIFT_JOB=\$(sbatch --parsable \
        --dependency=afterok:\$GLMSINGLE_JOB \
        --export=PARTICIPANT_LABEL=${SUBJECT},MODEL=session-shift \
        "\$APRF_DIR/fit_aprf.sh")
    echo "fit_aprf_shift:\$APRF_SHIFT_JOB"

    APRF_SHIFT_CV_JOB=\$(sbatch --parsable \
        --dependency=afterok:\$GLMSINGLE_JOB \
        --export=PARTICIPANT_LABEL=${SUBJECT} \
        "\$APRF_DIR/fit_aprf_shift_cv.sh")
    echo "fit_aprf_shift_cv:\$APRF_SHIFT_CV_JOB"
fi

# 10. fit_aprf_weighted
APRF_W_JOB=\$(sbatch --parsable \
    --dependency=afterok:\$GLMSINGLE_JOB \
    --export=PARTICIPANT_LABEL=${SUBJECT} \
    "\$APRF_DIR/fit_aprf_weighted.sh")
echo "fit_aprf_weighted:\$APRF_W_JOB"

# 11. fit_aprf_weighted_cv
APRF_W_CV_JOB=\$(sbatch --parsable \
    --dependency=afterok:\$GLMSINGLE_JOB \
    --export=PARTICIPANT_LABEL=${SUBJECT} \
    "\$APRF_DIR/fit_aprf_weighted_cv.sh")
echo "fit_aprf_weighted_cv:\$APRF_W_CV_JOB"

# 12. fit_vonmises
VONMISES_JOB=\$(sbatch --parsable \
    --dependency=afterok:\$GLMSINGLE_JOB \
    --export=PARTICIPANT_LABEL=${SUBJECT} \
    "\$APRF_DIR/fit_vonmises.sh")
echo "fit_vonmises:\$VONMISES_JOB"

# 13. fit_vonmises_cv
VONMISES_CV_JOB=\$(sbatch --parsable \
    --dependency=afterok:\$GLMSINGLE_JOB \
    --export=PARTICIPANT_LABEL=${SUBJECT} \
    "\$APRF_DIR/fit_vonmises_cv.sh")
echo "fit_vonmises_cv:\$VONMISES_CV_JOB"
EOF
)

# ── summary ───────────────────────────────────────────────────────────────────
log "All done. SLURM chain:"
echo "$SLURM_OUTPUT" | while IFS=: read -r name jobid; do
    printf "  %-24s  job %s\n" "$name" "$jobid"
done
