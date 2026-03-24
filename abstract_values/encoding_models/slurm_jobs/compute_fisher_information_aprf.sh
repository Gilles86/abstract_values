#!/bin/bash
#SBATCH --job-name=compute_fi_aprf
#SBATCH --output=/home/gdehol/logs/compute_fi_aprf_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Compute Fisher information for abstract value from the aPRF model.
# Requires fit_aprf (or fit_aprf with MODEL=session-shift) to have completed.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=pil02 compute_fisher_information_aprf.sh
#   sbatch --export=PARTICIPANT_LABEL=pil02,MODEL=session-shift compute_fisher_information_aprf.sh
#
# Optional overrides (--export key=value):
#   SESSION       space-separated session numbers (default: all sessions)
#   MODEL         standard or session-shift (default: standard)
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep)
#   ROI           ROI label (default: NPCr)
#   HEMI          hemisphere: L, R, LR, or None (default: None)
#   N_VOXELS      top voxels by R² to use (default: 250)
#   SMOOTHED      set to "1" to use smoothed betas (default: off)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

SESSION="${SESSION:-}"
MODEL="${MODEL:-standard}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
ROI="${ROI:-NPCr}"
HEMI="${HEMI:-None}"
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
[ "$MODEL" != "standard" ] && ARGS+=(--model "$MODEL")
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

echo "compute_fi_aprf: sub-${PARTICIPANT_LABEL}  model=${MODEL}  roi=${ROI}  hemi=${HEMI}  deriv=${FMRIPREP_DERIV}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/compute_fisher_information_aprf.py" \
    "${ARGS[@]}"
