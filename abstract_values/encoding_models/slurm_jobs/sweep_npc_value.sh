#!/bin/bash
#SBATCH --job-name=sweep_npc_value
#SBATCH --output=/home/gdehol/logs/sweep_npc_value_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --account=zne.uzh

# NPC value model comparison: single non-linear pRF vs weighted basis,
# crossed with condition handling (joint/shift/separate). CPU-only; the
# single-pRF grid+Adam fits are the cost (NPCr is small, so manageable).
#
# Usage:
#   sbatch --array=3-14 sweep_npc_value.sh
#   sbatch --export=PARTICIPANT_LABEL=03 sweep_npc_value.sh
#
# Optional overrides:
#   N_BASIS  basis counts (default "4 6 8 12 16 20")
#   FWHM     basis fwhm in CHF (default "2 4 6 10")
#   N_ITER   Adam iters for single-pRF fits (default 500)
#   SMOOTHED set to "1" for smoothed betas

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi

N_BASIS="${N_BASIS:-4 6 8 12 16 20}"
FWHM="${FWHM:-2 4 6 10}"
N_ITER="${N_ITER:-500}"
SMOOTHED="${SMOOTHED:-0}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values
ENV=$HOME/data/conda/envs/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --n-basis $N_BASIS
    --fwhm $FWHM
    --n-iter "$N_ITER"
)
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

echo "sweep_npc_value: sub-${PARTICIPANT_LABEL}  n_basis=[${N_BASIS}]  fwhm=[${FWHM}]  n_iter=${N_ITER}"
echo "Args: ${ARGS[*]}"

# exec env-binary directly: streams logs live + forwards SIGTERM (conda run
# buffers and orphans the child). See sciencecluster skill.
exec ${ENV}/bin/python -u \
    "$REPO/abstract_values/encoding_models/sweep_npc_value.py" \
    "${ARGS[@]}"
