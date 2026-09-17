#!/bin/bash
#SBATCH --job-name=cf_coupling
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/cf_coupling_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Primary connective-field test: does residual NPC <-> V1 coupling follow the
# session's orientation->value mapping? See test_coupling.py.
#
# Usage:
#   sbatch --array=3-30 test_coupling.sh
#   sbatch --export=ALL,PARTICIPANT_LABEL=pil01 test_coupling.sh
#
# Optional overrides (--export ALL,KEY=value):
#   N_SHUFFLE   label permutations (default 200)
#   LAGS        also remove neighbouring trials' orientations, +-LAGS (default 0)
#   PROJECTION  bins (default) | iem
#   N_CHANNELS  bins: V1 orientation channels (default 8)
#   N_BASIS, KAPPA  iem: basis functions (default 8) and concentration (default 2)
#   GAIN        1 = remove per-trial gain in both regions
#   DRIVE       low | high: target trials by predicted drive (npc_from_v1 only)
#   SHAM_OUTER  1 = outermost lag uses permuted orientations (df-matched control)
#   REPO        checkout to run from (default ~/git/abstract_values)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi
N_SHUFFLE="${N_SHUFFLE:-200}"
LAGS="${LAGS:-0}"
EXTRA=(--projection "${PROJECTION:-bins}" --n-channels "${N_CHANNELS:-8}"
       --n-basis "${N_BASIS:-8}" --kappa "${KAPPA:-2}")
[ "${SHAM_OUTER:-0}" = "1" ] && EXTRA+=(--sham-outer)
[ "${GAIN:-0}" = "1" ] && EXTRA+=(--gain)
[ -n "${DRIVE:-}" ] && EXTRA+=(--drive "$DRIVE" --directions npc_from_v1)
REPO="${REPO:-$HOME/git/abstract_values}"

. $HOME/init_conda.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
conda run --no-capture-output -n abstract_values python -u -W ignore \
    -m abstract_values.connective_fields.test_coupling "$PARTICIPANT_LABEL" \
    --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue --n-shuffle "$N_SHUFFLE" --lags "$LAGS" "${EXTRA[@]}"
