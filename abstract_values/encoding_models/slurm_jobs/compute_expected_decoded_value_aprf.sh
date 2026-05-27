#!/bin/bash
#SBATCH --job-name=compute_eu_aprf
#SBATCH --output=/home/gdehol/logs/compute_eu_aprf_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=zne.uzh

# Simulate from aPRF session-shift encoding model + Student-t noise, decode,
# compute per-stimulus expected decoded value, bias, variance, abs error.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=03 compute_expected_decoded_value_aprf.sh
#
# Optional overrides (--export key=value):
#   ROI            ROI label (default: NPCr)
#   HEMI           hemisphere: LR, L, R, None (default: None)
#   N_VOXELS       top voxels by R² (default: 100)
#   N_SIMULATIONS  noisy repeats per stimulus (default: 1000)
#   N_VALUES       stimulus grid points (default: 200)
#   BATCH_STIMULI  how many stimuli to simulate together (memory knob, default 25)
#   SMOOTHED       set to "1" for smoothed betas (default: off)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

ROI="${ROI:-NPCr}"
HEMI="${HEMI:-None}"
N_VOXELS="${N_VOXELS:-100}"
FDR_ALPHA="${FDR_ALPHA:-}"
P_SIGNAL_THR="${P_SIGNAL_THR:-}"
N_SIMULATIONS="${N_SIMULATIONS:-1000}"
N_VALUES="${N_VALUES:-200}"
BATCH_STIMULI="${BATCH_STIMULI:-25}"
SMOOTHED="${SMOOTHED:-0}"
SPHERICAL="${SPHERICAL:-0}"

if [ -n "$FDR_ALPHA" ] && [ -n "$P_SIGNAL_THR" ]; then
    echo "ERROR: FDR_ALPHA and P_SIGNAL_THR are mutually exclusive."
    exit 1
fi

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

ARGS=(
    "$PARTICIPANT_LABEL"
    --roi "$ROI"
    --hemi "$HEMI"
    --n-voxels "$N_VOXELS"
    --n-simulations "$N_SIMULATIONS"
    --n-values "$N_VALUES"
    --batch-stimuli "$BATCH_STIMULI"
    --bids-folder "$BIDS_FOLDER"
)
[ -n "$FDR_ALPHA" ] && ARGS+=(--fdr-alpha "$FDR_ALPHA")
[ -n "$P_SIGNAL_THR" ] && ARGS+=(--p-signal-thr "$P_SIGNAL_THR")
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)
[ "$SPHERICAL" = "1" ] && ARGS+=(--spherical-noise)

echo "compute_eu_aprf: sub-${PARTICIPANT_LABEL}  roi=${ROI}  n_voxels=${N_VOXELS}  n_simulations=${N_SIMULATIONS}  spherical=${SPHERICAL}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh
conda activate abstract_values

PYTHONUNBUFFERED=1 python -u \
    "$REPO/abstract_values/encoding_models/compute_expected_decoded_value_aprf.py" \
    "${ARGS[@]}"
