#!/bin/bash
#SBATCH --job-name=simdec_mix
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/simdec_mix_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

# Simulated decoding precision of the mix-<lambda> codes (simulate_decoding_mix.py).
# Needs the mix-<lambda> full fits and CV fits of this participant.
#   sbatch --export=ALL,PARTICIPANT_LABEL=05,SMOOTHED=1 simulate_decoding_mix.sh

set -eo pipefail
export KERAS_BACKEND=tensorflow PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
REPO="${REPO:-$HOME/git/abstract_values}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
ARGS=("$PARTICIPANT_LABEL" --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue)
[ "${SMOOTHED:-0}" = "1" ] && ARGS+=(--smoothed)
exec $HOME/data/conda/envs/abstract_values/bin/python -u -m \
    abstract_values.encoding_models.simulate_decoding_mix "${ARGS[@]}"
