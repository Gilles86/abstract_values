#!/bin/bash
#SBATCH --job-name=a5_coupling90
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/a5_coupling90_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

# A5: V1 -> IPS residual coupling by V1 preferred orientation (see a5_coupling_90.py).
# Usage (array index -> subject label via SUBJECTS list below):
#   sbatch --array=0-34 a5_coupling_90.sh
#   sbatch --export=ALL,PARTICIPANT_LABEL=12 a5_coupling_90.sh
set -eo pipefail
SUBJECTS=(pil01 pil02 $(seq -f "%02g" 3 35))
if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
fi
REPO="${REPO:-$HOME/git/abstract_values}"
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
cd "$REPO"
exec $HOME/data/conda/envs/abstract_values/bin/python -u -W ignore \
    analyses/followup_2026-10/a5_coupling_90/a5_coupling_90.py "$PARTICIPANT_LABEL" \
    --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue \
    --n-perm "${N_PERM:-1000}" --n-sub "${N_SUB:-200}" --lags "${LAGS:-1}"
