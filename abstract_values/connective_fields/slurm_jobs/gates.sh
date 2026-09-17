#!/bin/bash
#SBATCH --job-name=cf_gates
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/cf_gates_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Step-0 gates for the orientation-space connective-field analysis.
#
# Usage:
#   sbatch --array=3-30 gates.sh                        # sub-03 .. sub-30
#   sbatch --export=ALL,PARTICIPANT_LABEL=pil01 gates.sh
#
# Optional overrides (--export ALL,KEY=value):
#   V1_VOXELS   selected (default) | all  -- V1 voxels feeding the channels
#   N_PERM      injection permutations (default 50)
#   AMPLITUDES  space-separated injected couplings (default "0 0.05 0.1 0.2")
#   REPO        checkout to run from (default ~/git/abstract_values); put first on
#               PYTHONPATH so a clean clone wins over the env's editable install

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi
V1_VOXELS="${V1_VOXELS:-selected}"
N_PERM="${N_PERM:-50}"
AMPLITUDES="${AMPLITUDES:-0 0.05 0.1 0.2}"

. $HOME/init_conda.sh
export PYTHONUNBUFFERED=1
REPO="${REPO:-$HOME/git/abstract_values}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
conda run --no-capture-output -n abstract_values python -u -W ignore \
    -m abstract_values.connective_fields.gates "$PARTICIPANT_LABEL" \
    --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue \
    --v1-voxels "$V1_VOXELS" --n-perm "$N_PERM" --amplitudes $AMPLITUDES
