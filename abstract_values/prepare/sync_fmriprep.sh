#!/bin/bash
# Sync fmriprep derivatives from sciencecluster to local, keeping only
# T1w-space functional outputs (excludes fsnative BOLD and surface .gii files).

rsync -av --progress \
  --exclude '*_space-fsnative_*' \
  --exclude '*_hemi-*' \
  sciencecluster:/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/fmriprep/ \
  /data/ds-abstractvalue/derivatives/fmriprep/
