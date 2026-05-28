#!/bin/bash
#SBATCH --job-name=v1_basis_sweep
#SBATCH --output=/home/gdehol/logs/v1_basis_sweep_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=zne.uzh

# V1 basis-count sweep: per-session vonmises fit + EU at multiple n_basis,
# fdr05 selection, spherical noise.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=04 v1_basis_sweep.sh
#
# Optional overrides:
#   SMOOTHED        set to "1" for smoothed betas (default: off)
#   N_BASIS         space-separated basis counts (default: "8 16 32")
#   FMRIPREP_DERIV  (unused — passed through, default: fmriprep)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

SMOOTHED="${SMOOTHED:-0}"
N_BASIS="${N_BASIS:-8 16 32}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --n-basis $N_BASIS
)
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

echo "v1_basis_sweep: sub-${PARTICIPANT_LABEL}  n_basis=[$N_BASIS]  smoothed=${SMOOTHED}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh

PYTHONUNBUFFERED=1 conda run -n abstract_values python -u \
    -m abstract_values.experiments.v1_basis_sweep \
    "${ARGS[@]}"
