#!/bin/bash
# Sync fmriprep derivatives from sciencecluster to local.
#
# Keeps:
#   - HTML reports + figures/ SVGs
#   - anat/ (preproc T1w, segs, surfaces, transforms, brain masks)
#   - *_space-T1w_boldref.nii.gz    (T1w-space single-volume reference — coreg QA)
#   - *_space-T1w_desc-brain_mask*  (T1w-space BOLD brain mask)
#   - *_*xfm*                       (coreg / motion-correction transforms)
#
# Excludes (large or rarely-used):
#   - 4D preproc BOLD timeseries (*_desc-preproc_bold.nii.gz) — too big
#   - native-BOLD-space boldrefs (*_desc-hmc_boldref*, *_desc-coreg_boldref*) —
#     not useful locally, we work in T1w space
#   - fsnative-space outputs and surface .gii — only meaningful on cluster
#     with the matching freesurfer dir

EXCLUDES=(
    --exclude '*_space-fsnative_*'
    --exclude 'func/*_hemi-*'
    --exclude '*_desc-preproc_bold.nii.gz'
    --exclude '*_desc-hmc_boldref*'
    --exclude '*_desc-coreg_boldref*'
)
CLUSTER=sciencecluster:/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives
LOCAL=/data/ds-abstractvalue/derivatives

for DERIV in fmriprep; do
    echo "=== syncing $DERIV ==="
    # Anti-stall flags. Symptom we're fighting: rsync hangs for many
    # seconds (sometimes minutes) on a single file, but if you Ctrl+C
    # and rerun it resumes very fast. Multiple root causes; this set
    # covers all the common ones.
    #
    # --whole-file: skip rsync's rolling-checksum delta algorithm. For
    #     fmriprep files (gzip NIfTIs, either absent or bit-identical
    #     locally), the delta computation is pure server-side overhead.
    # --partial: keep partial transfers on Ctrl+C so a retry resumes
    #     rather than restarts the big file from byte 0.
    # ssh -T:  disable PTY allocation, which avoids occasional remote-
    #     side stalls in non-interactive sessions.
    # ServerAliveInterval=30 / ServerAliveCountMax=3: ssh sends keepalive
    #     every 30s and disconnects after 3 missed responses. Without
    #     this, a transient NAT / network drop kills the connection
    #     silently and rsync sits forever waiting for the next byte
    #     (the most common cause of "stuck on a random file").
    # ServerAliveInterval=30 / 3 → max-hang 90s before clean death.
    # (macOS rsync is 2.6.9 — no --append-verify available.)
    rsync -av --progress --whole-file --partial \
      -e "ssh -T -o ServerAliveInterval=30 -o ServerAliveCountMax=3" \
      "${EXCLUDES[@]}" \
      "${CLUSTER}/${DERIV}/" \
      "${LOCAL}/${DERIV}/"
done
