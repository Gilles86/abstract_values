#!/bin/bash
#SBATCH --job-name=sweep_npc_value
#SBATCH --output=/home/gdehol/logs/sweep_npc_value_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --account=zne.uzh

# NPC value model comparison: single non-linear pRF vs weighted basis,
# crossed with condition handling (joint/shift/separate). CPU-only; the
# single-pRF grid+Adam fits are the cost (NPCr is small, so manageable).
#
# Resources are measured, not guessed: a 243-cell grid (6 n_basis x 4 fwhm x
# 5 alpha x 2 cond, plus 3 single-pRF fits) on sub-29 / NPCr ran in 24m21s at
# 4.6 GB peak RSS (job 5475839). The ratio grid has 5 widths rather than 4, so
# 303 cells -- and the ~1900 extra ridge solves are minutes, not hours, since
# the 24 single-pRF grid+Adam fits are what actually costs. 1h30 / 12G stands.
# Over-requesting costs queue priority for nothing, so re-measure rather than
# padding if the grid grows again.
#
# Usage:
#   sbatch --array=3-14 sweep_npc_value.sh
#   sbatch --export=PARTICIPANT_LABEL=03 sweep_npc_value.sh
#
# Optional overrides:
#   N_BASIS  basis counts (default "4 6 8 12 16 20")
#   FWHM_RATIO basis fwhm as a multiple of inter-basis spacing
#            (default "0.75 1 1.5 2 3"; 2 is what fit_aprf_weighted deploys)
#   FWHM     absolute basis fwhm in CHF; overrides FWHM_RATIO
#   N_ITER   Adam iters for single-pRF fits (default 500)
#   ALPHA    ridge penalties for the weighted basis (default "0.01 0.1 1 10 100")
#   ROI      ROI desc (default NPCr; e.g. BensonV1ecc075-375)
#   ROI_HEMI hemi entity, or "none" (default) for descs that encode it already
#   SMOOTHED set to "1" for smoothed betas

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi

N_BASIS="${N_BASIS:-4 6 8 12 16 20}"
FWHM_RATIO="${FWHM_RATIO:-0.75 1 1.5 2 3}"
FWHM="${FWHM:-}"
N_ITER="${N_ITER:-500}"
ALPHA="${ALPHA:-0.01 0.1 1 10 100}"
ROI="${ROI:-NPCr}"
ROI_HEMI="${ROI_HEMI:-none}"
SMOOTHED="${SMOOTHED:-0}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values
ENV=$HOME/data/conda/envs/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --n-basis $N_BASIS
    --n-iter "$N_ITER"
    --alpha $ALPHA
    --roi "$ROI"
    --roi-hemi "$ROI_HEMI"
)
if [ -n "$FWHM" ]; then
    ARGS+=(--fwhm $FWHM)
else
    ARGS+=(--fwhm-ratio $FWHM_RATIO)
fi
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

WIDTH_DESC="${FWHM:+fwhm=[${FWHM}] CHF}"
WIDTH_DESC="${WIDTH_DESC:-fwhm=[${FWHM_RATIO}] x spacing}"
echo "sweep_npc_value: sub-${PARTICIPANT_LABEL}  roi=${ROI}  n_basis=[${N_BASIS}]  ${WIDTH_DESC}  alpha=[${ALPHA}]  n_iter=${N_ITER}"
echo "Args: ${ARGS[*]}"

# exec env-binary directly: streams logs live + forwards SIGTERM (conda run
# buffers and orphans the child). See sciencecluster skill.
exec ${ENV}/bin/python -u \
    "$REPO/abstract_values/encoding_models/sweep_npc_value.py" \
    "${ARGS[@]}"
