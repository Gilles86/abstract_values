#!/bin/bash
#SBATCH --job-name=fit_glmsingle
#SBATCH --output=/home/gdehol/logs/fit_glmsingle_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
# 128G covers 2-session subjects with TR_UP=0.5 (2x upsample). 4x upsample
# (TR_UP=0.25) hit OOM at ~131G and TIMEOUT at 4h on sub-07/08 — now reduced
# in fit_glmsingle.py. 6h hit TIMEOUT for sub-07 unsmoothed + sub-10 (both
# variants) — bumped to 12h so the slow subjects can finish.

# Fit GLMsingle single-trial betas for the abstract values fMRI task.
# Fits all sessions jointly by default — single-session fitting is a corner
# case and should be avoided unless there is a specific reason.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=pil02 fit_glmsingle.sh
#
# Optional overrides (--export key=value):
#   SESSION         space-separated session numbers; omit to fit all sessions
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep)
#   SMOOTHED        set to "1" to smooth BOLD before fitting (default: off)
#   DEBUG           set to "1" to write all 4 model steps + figures (default: off)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

SESSION="${SESSION:-}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
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

# Use the env's python binary directly: `conda run` buffers all subprocess
# stdout until exit, which makes monitoring progress via `tail -f` impossible.
export PYTHONUNBUFFERED=1
"$HOME/data/conda/envs/abstract_values/bin/python" -u \
    "$REPO/abstract_values/glm/fit_glmsingle.py" \
    "${ARGS[@]}"
