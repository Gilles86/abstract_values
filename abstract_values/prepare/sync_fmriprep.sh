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
    # --whole-file: skip the rolling-checksum delta algorithm. rsync's
    #     default delta-mode stalls for many seconds per multi-GB file
    #     while computing deltas server-side; for fmriprep where files
    #     either don't exist locally yet or are bit-identical, the delta
    #     pass is pure overhead. Symptom of NOT using this: rsync
    #     "freezes" on one file for ~minute, then if you Ctrl+C + retry,
    #     it resumes very fast (because size+mtime check skips the file).
    # --partial: keep partial transfers on Ctrl+C so a retry resumes
    #     rather than restarts the big file from byte 0.
    # (macOS rsync is 2.6.9 — no --append-verify; --whole-file +
    #  --partial is enough for our case.)
    rsync -av --progress --whole-file --partial \
      "${EXCLUDES[@]}" \
      "${CLUSTER}/${DERIV}/" \
      "${LOCAL}/${DERIV}/"
done
