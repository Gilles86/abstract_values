#!/bin/bash
#SBATCH --job-name=fit_aprf_weighted
#SBATCH --output=/home/gdehol/logs/fit_aprf_weighted_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Fit weighted abstract pRF (log-Gaussian basis set) to all data.
# Analogous to the Von Mises model but for the abstract value dimension.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=pil02 fit_aprf_weighted.sh
#
# Optional overrides (--export key=value):
#   SESSION         space-separated session numbers (default: all sessions)
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep)
#   N_BASIS         number of log-Gaussian basis pRFs (default: 8)
#   SMOOTHED        set to "1" to use smoothed betas (default: off)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

SESSION="${SESSION:-}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
N_BASIS="${N_BASIS:-8}"
SMOOTHED="${SMOOTHED:-0}"
BASIS="${BASIS:-loggauss}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
    --n-basis "$N_BASIS"
)

[ -n "$SESSION" ] && ARGS+=(--sessions $SESSION)
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)
[ "$BASIS" != "loggauss" ] && ARGS+=(--basis "$BASIS")

echo "fit_aprf_weighted: sub-${PARTICIPANT_LABEL}  deriv=${FMRIPREP_DERIV}  n_basis=${N_BASIS}  smoothed=${SMOOTHED}  basis=${BASIS}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/fit_aprf_weighted.py" \
    "${ARGS[@]}"
