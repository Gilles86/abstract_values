#!/bin/bash
# Sync fmriprep derivatives from sciencecluster to local, keeping only
# T1w-space functional outputs (excludes fsnative BOLD and surface .gii files).

EXCLUDES=(--exclude '*_space-fsnative_*' --exclude 'func/*_hemi-*' --exclude '*_desc-preproc_bold.nii.gz')
CLUSTER=sciencecluster:/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives
LOCAL=/data/ds-abstractvalue/derivatives

for DERIV in fmriprep fmriprep-flair fmriprep-noflair; do
    echo "=== syncing $DERIV ==="
    rsync -av --progress "${EXCLUDES[@]}" \
      "${CLUSTER}/${DERIV}/" \
      "${LOCAL}/${DERIV}/"
done
