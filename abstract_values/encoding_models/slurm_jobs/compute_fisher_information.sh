#!/bin/bash
#SBATCH --job-name=compute_fi_vonmises
#SBATCH --output=/home/gdehol/logs/compute_fi_vonmises_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Compute Fisher information for gabor orientation from the Von Mises model.
# Note: re-fits Von Mises weights internally; does not depend on fit_vonmises output.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=pil02,ROI=BensonV1,HEMI=LR compute_fisher_information.sh
#
# Required overrides (--export key=value):
#   ROI           ROI label, e.g. BensonV1, NPC (default: BensonV1)
#   HEMI          hemisphere: L, R, LR, or None (default: LR)
#
# Optional overrides:
#   SESSION       space-separated session numbers (default: all sessions)
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep)
#   N_VOXELS      top voxels by R² to use (default: 250)
#   SMOOTHED      set to "1" to use smoothed betas (default: off)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

SESSION="${SESSION:-}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
ROI="${ROI:-BensonV1}"
HEMI="${HEMI:-LR}"
N_VOXELS="${N_VOXELS:-250}"
SMOOTHED="${SMOOTHED:-0}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --roi "$ROI"
    --hemi "$HEMI"
    --n-voxels "$N_VOXELS"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
)

[ -n "$SESSION" ] && ARGS+=(--sessions $SESSION)
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

echo "compute_fi_vonmises: sub-${PARTICIPANT_LABEL}  roi=${ROI}  hemi=${HEMI}  deriv=${FMRIPREP_DERIV}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/compute_fisher_information.py" \
    "${ARGS[@]}"
