#!/bin/bash
# Backfill the V1 orientation expected-uncertainty branch.
#
# This branch is not in the Snakefile -- the subjects that already have it were
# run out of band -- so this script reproduces the full chain for the subjects
# that are missing it, matching the file set the existing subjects have:
#
#   vonmises-session-shift fit   x {unsmoothed, smoothed}
#     -> expected-decoded-orientation  x {nvoxels-100, fdr05} x {spherical, residual}
#
# Each EU invocation loops over the subject's sessions internally and writes one
# TSV per session, so 8 EU jobs per subject cover both sessions.  EU jobs chain
# off the matching-smoothing fit with afterok.
#
# Prerequisites (verified present for sub-03..28 as of 2026-08-25): the JOINT
# vonmises fit, both smoothed and unsmoothed -- the FDR voxel selection is
# computed in-process from its whole-brain R2 mixture, so no separate mixture
# job is needed.
#
# Usage:
#   bash submit_v1_eu_backfill.sh 11 12 15 16 ...
#   DRY_RUN=1 bash submit_v1_eu_backfill.sh 11        # print, don't submit

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN="${DRY_RUN:-0}"

submit() {   # echoes the job id
    if [ "$DRY_RUN" = "1" ]; then echo "DRY:$*" >&2; echo "000000"; return; fi
    sbatch --parsable "$@"
}

for S in "$@"; do
    for SM in 0 1; do
        smtag=$([ "$SM" = "1" ] && echo smoothed || echo unsmoothed)

        fit=$(submit --export="PARTICIPANT_LABEL=${S},SESSION_SHIFT=1,SMOOTHED=${SM}" \
                     "$HERE/fit_vonmises.sh")
        echo "sub-${S} ${smtag}: fit=${fit}"

        for SEL in "N_VOXELS=100" "FDR_ALPHA=0.05"; do
            for SPH in 1 0; do
                eu=$(submit --dependency="afterok:${fit}" \
                     --export="PARTICIPANT_LABEL=${S},SESSION_SHIFT_WEIGHTS=1,SMOOTHED=${SM},SPHERICAL=${SPH},${SEL}" \
                     "$HERE/compute_expected_decoded_orientation_vonmises.sh")
                echo "    eu=${eu}  ${SEL}  spherical=${SPH}"
            done
        done
    done
done
