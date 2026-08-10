#!/bin/bash
#SBATCH --job-name=compute_eu_vonmises
#SBATCH --output=/home/gdehol/logs/compute_eu_vonmises_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=zne.uzh

# Project models are TF-native; cluster keras.json defaults to jax — pin TF.
export KERAS_BACKEND=tensorflow

# V1 expected-decoded-orientation simulation via vonmises basis weights.
#
# Optional overrides (--export key=value):
#   ROI            ROI label (default: BensonV1)
#   HEMI           hemisphere: LR, L, R, None (default: LR)
#   N_VOXELS       top voxels by R² (default: 100)
#   FDR_ALPHA      FDR α on vonmises whole-brain mixture
#   P_SIGNAL_THR   P(signal|r²) threshold on same (mutually exclusive)
#   N_SIMULATIONS  noisy repeats per orientation (default: 1000)
#   N_ORIENTATIONS stimulus grid points (default: 180)
#   BATCH_STIMULI  stimuli per simulation batch (memory knob, default 25)
#   SMOOTHED       set to "1" for smoothed betas (default: off)
#   SPHERICAL      set to "1" for iid Gaussian noise (default: full Omega)
#   FULL_GRID      set to "1" for full [0°, 180°) grid (default: 23 trained)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

ROI="${ROI:-BensonV1}"
HEMI="${HEMI:-LR}"
N_VOXELS="${N_VOXELS:-100}"
FDR_ALPHA="${FDR_ALPHA:-}"
P_SIGNAL_THR="${P_SIGNAL_THR:-}"
N_SIMULATIONS="${N_SIMULATIONS:-1000}"
N_ORIENTATIONS="${N_ORIENTATIONS:-180}"
BATCH_STIMULI="${BATCH_STIMULI:-25}"
SMOOTHED="${SMOOTHED:-0}"
SPHERICAL="${SPHERICAL:-1}"  # default: iid Gaussian (see Python script)
FULL_GRID="${FULL_GRID:-0}"
SESSION_SHIFT_WEIGHTS="${SESSION_SHIFT_WEIGHTS:-0}"

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
    --n-orientations "$N_ORIENTATIONS"
    --batch-stimuli "$BATCH_STIMULI"
    --bids-folder "$BIDS_FOLDER"
)
[ -n "$FDR_ALPHA" ] && ARGS+=(--fdr-alpha "$FDR_ALPHA")
[ -n "$P_SIGNAL_THR" ] && ARGS+=(--p-signal-thr "$P_SIGNAL_THR")
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)
[ "$SPHERICAL" = "1" ] && ARGS+=(--spherical-noise) || ARGS+=(--no-spherical-noise)
[ "$FULL_GRID" = "1" ] && ARGS+=(--full-grid)
[ "$SESSION_SHIFT_WEIGHTS" = "1" ] && ARGS+=(--session-shift-weights)

echo "compute_eu_vonmises: sub-${PARTICIPANT_LABEL}  roi=${ROI}  n_voxels=${N_VOXELS}  fdr=${FDR_ALPHA}  psig=${P_SIGNAL_THR}  smoothed=${SMOOTHED}  spherical=${SPHERICAL}  full_grid=${FULL_GRID}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh
conda activate abstract_values

PYTHONUNBUFFERED=1 python -u \
    "$REPO/abstract_values/encoding_models/compute_expected_decoded_orientation_vonmises.py" \
    "${ARGS[@]}"
