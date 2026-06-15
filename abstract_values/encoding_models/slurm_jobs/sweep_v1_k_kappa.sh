#!/bin/bash
#SBATCH --job-name=sweep_v1_k_kappa
#SBATCH --output=/home/gdehol/logs/sweep_v1_k_kappa_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=zne.uzh

# V1 Von Mises model comparison: sweep n_basis (k) x kappa (dispersion).
# CPU-only -- encoding cvR2 is closed-form OLS; the optional --decode pass
# adds a (CPU) noise-model + posterior fit per fold. No GPU requested.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=04 sweep_v1_k_kappa.sh
#   sbatch --array=3-16 sweep_v1_k_kappa.sh                  # study subjects
#   sbatch --export=PARTICIPANT_LABEL=04,DECODE=1 sweep_v1_k_kappa.sh
#
# Optional overrides (--export key=value):
#   N_BASIS    space-separated basis counts (default: "4 6 8 12 16 20 24")
#   KAPPA      space-separated kappa values (default: "1 2 4 8")
#   DECODE     set to "1" to add out-of-sample FDR decoding (slower)
#   FDR_ALPHA  FDR alpha for decoding voxel selection (default: 0.05)
#   SMOOTHED   set to "1" to use smoothed betas (default: off)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi

N_BASIS="${N_BASIS:-4 6 8 12 16 20 24}"
KAPPA="${KAPPA:-1 2 4 8}"
DECODE="${DECODE:-0}"
FDR_ALPHA="${FDR_ALPHA:-0.05}"
SMOOTHED="${SMOOTHED:-0}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --n-basis $N_BASIS
    --kappa $KAPPA
    --fdr-alpha "$FDR_ALPHA"
)
[ "$DECODE" = "1" ]   && ARGS+=(--decode)
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

echo "sweep_v1_k_kappa: sub-${PARTICIPANT_LABEL}  n_basis=[${N_BASIS}]  kappa=[${KAPPA}]  decode=${DECODE}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/encoding_models/sweep_v1_k_kappa.py" \
    "${ARGS[@]}"
