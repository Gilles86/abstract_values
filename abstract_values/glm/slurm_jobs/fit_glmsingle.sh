#!/bin/bash
#SBATCH --job-name=fit_glmsingle
#SBATCH --output=/home/gdehol/logs/fit_glmsingle_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Fit GLMsingle single-trial betas for the abstract values fMRI task.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=pil02 fit_glmsingle.sh
#   sbatch --export=PARTICIPANT_LABEL=pil02,FMRIPREP_DERIV=fmriprep-t2w fit_glmsingle.sh
#
# Optional overrides (--export key=value):
#   SESSION         session number(s), space-separated (default: all sessions)
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep-flair)
#   SMOOTHED        set to "1" to smooth BOLD before fitting (default: off)
#   DEBUG           set to "1" to write all 4 model steps + figures (default: off)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

SESSION="${SESSION:-}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep-flair}"
SMOOTHED="${SMOOTHED:-0}"
DEBUG="${DEBUG:-0}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
)

[ -n "$SESSION" ] && ARGS+=(--sessions $SESSION)
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)
[ "$DEBUG"    = "1" ] && ARGS+=(--debug)

echo "fit_glmsingle: sub-${PARTICIPANT_LABEL}  deriv=${FMRIPREP_DERIV}  smoothed=${SMOOTHED}  debug=${DEBUG}"
echo "Args: ${ARGS[*]}"

# Load environment
. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/glm/fit_glmsingle.py" \
    "${ARGS[@]}"
