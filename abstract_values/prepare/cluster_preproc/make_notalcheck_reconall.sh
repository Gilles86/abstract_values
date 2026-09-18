#!/bin/bash
# Build a patched copy of FreeSurfer's recon-all with the Talairach AFD
# check disabled, for bind-mounting into the fmriprep container.
#
# Why
# ---
# `talairach_afd` compares the computed talairach.xfm against a reference
# distribution and hard-fails recon-all if the subject looks atypical. It
# false-positives on subjects whose head was rotated substantially in the
# scanner, even when the registration itself is perfectly good.
#
# sub-27 (2026-08-19) tripped this: pval=0.0034 vs threshold=0.0050, on a
# talairach transform with a 34 deg yaw but near-uniform singular values
# (1.081 / 1.037 / 1.034) -- i.e. a sane, near-rigid fit of a turned head.
# Compare sub-25/24/23, all of which sit near identity with ~15 deg pitch.
#
# FreeSurfer's documented remedy is the `-notal-check` flag, but fmriprep
# does not expose recon-all directives, and both expert-options hooks
# (per-subject scripts/expert-options and SUBJECTS_DIR/global-expert-
# options.txt) hard-require `-xopts-use` on the command line, which
# smriprep does not pass -- supplying them makes recon-all error out.
#
# So we patch the two `set DoTalCheck = 1;` assignments (the `-all` and
# `-autorecon1` case blocks; smriprep uses `-autorecon1`) to 0. This is
# exactly equivalent to `-notal-check` and changes nothing else: the
# talairach transform is still computed and used, only the goodness-of-fit
# gate is skipped.
#
# Usage
# -----
#   bash make_notalcheck_reconall.sh
#   # then, per subject:
#   NOTAL_CHECK=1 PARTICIPANT_LABEL=27 bash fmriprep.sh
#
# NB: source /etc/profile BEFORE `set -e`, and never with `set -u` — it
# references unbound vars (XDG_DATA_DIRS) and returns non-zero, either of
# which aborts the script silently.
source /etc/profile || true
module load apptainer/1.4.1

set -eo pipefail

CONTAINER="${CONTAINER:-/shares/zne.uzh/containers/fmriprep-25.2.5}"
PATCHDIR="${PATCHDIR:-/shares/zne.uzh/gdehol/container_patches}"
OUT="$PATCHDIR/recon-all-notalcheck"

mkdir -p "$PATCHDIR"

apptainer exec "$CONTAINER" cat /opt/freesurfer/bin/recon-all > "$OUT"
chmod +x "$OUT"

BEFORE=$(grep -c 'set DoTalCheck       = 1;' "$OUT" || true)
if [[ "$BEFORE" -ne 2 ]]; then
    echo "ERROR: expected 2 'DoTalCheck = 1' assignments, found $BEFORE."
    echo "The container's recon-all differs from what this patch assumes."
    exit 1
fi

sed -i 's/set DoTalCheck       = 1;/set DoTalCheck       = 0;  # PATCHED: talairach_afd disabled/' "$OUT"

AFTER=$(grep -c 'set DoTalCheck       = 1;' "$OUT" || true)
PATCHED=$(grep -c 'PATCHED: talairach_afd disabled' "$OUT" || true)

echo "recon-all patched: $OUT"
echo "  'DoTalCheck = 1' remaining: $AFTER  (expected 0)"
echo "  patched assignments:        $PATCHED  (expected 2)"

if [[ "$AFTER" -ne 0 || "$PATCHED" -ne 2 ]]; then
    echo "ERROR: patch did not apply cleanly."
    exit 1
fi

echo "OK — bind-mount with: -B $OUT:/opt/freesurfer/bin/recon-all"
