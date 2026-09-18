#!/bin/bash
# Sync analysis derivatives from sciencecluster to local.
# Skips GLMsingle .npy intermediates (DESIGNINFO, RUNWISEFIR, TYPED_FITHRF_GLMDENOISE_RR)
# — those are huge and only needed on the cluster for downstream fits.

CLUSTER=sciencecluster:/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives
LOCAL=/data/ds-abstractvalue/derivatives

# Lightweight: masks, encoding_models (incl. fisher info), decoding
for DERIV in masks encoding_models decoding; do
    echo "=== syncing $DERIV ==="
    rsync -av --progress \
      "${CLUSTER}/${DERIV}/" \
      "${LOCAL}/${DERIV}/"
done

# GLMsingle: keep .nii.gz betas + figures, skip .npy intermediates
for DERIV in glmsingle glmsingle.smoothed; do
    echo "=== syncing $DERIV (excluding .npy) ==="
    rsync -av --progress \
      --exclude '*.npy' \
      "${CLUSTER}/${DERIV}/" \
      "${LOCAL}/${DERIV}/"
done
