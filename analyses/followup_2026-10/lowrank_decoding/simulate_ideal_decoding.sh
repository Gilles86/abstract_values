#!/bin/bash
#SBATCH --job-name=simideal
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/home/gdehol/logs/simideal_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
# CPU only. Array index -> participant (1..36 = pil01 pil02 03..36).
#   MODE=ideal:    all ROI x code combinations for the participant (minutes)
#   MODE=pipeline: one ROI x CODE through the real CV-N low-rank decoder (hours)
#   sbatch --array=1-36 --export=ALL,MODE=ideal simulate_ideal_decoding.sh
#   sbatch --array=1-36 --export=ALL,MODE=pipeline,ROI=IPS12,CODE=lam1 simulate_ideal_decoding.sh
set -eo pipefail
SUBJECTS=(pil01 pil02 $(seq -f "%02g" 3 36))
S=${PARTICIPANT_LABEL:-${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}}
export KERAS_BACKEND=tensorflow PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/git/braincoder-lowrank:$HOME/git/abstract_values${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
PY=$HOME/data/conda/envs/abstract_values/bin/python
SCRIPT=$HOME/git/abstract_values/analyses/followup_2026-10/lowrank_decoding/simulate_ideal_decoding.py
BIDS=/shares/zne.uzh/gdehol/ds-abstractvalue
sleep $(( (${SLURM_ARRAY_TASK_ID:-0} % 12) * 5 ))
if [ "$MODE" = "ideal" ]; then
    for r in BensonV1 BensonV2 BensonV3; do
        $PY -u $SCRIPT "$S" --roi $r --code vonmises --mode ideal --bids-folder $BIDS --noise ${NOISE:-cvmatched}
    done
    for r in BensonV1 IPS0 IPS12 IPS345 NPCr; do
        for c in lam0 lam1; do
            $PY -u $SCRIPT "$S" --roi $r --code $c --mode ideal --bids-folder $BIDS --noise ${NOISE:-cvmatched}
        done
    done
else
    exec $PY -u $SCRIPT "$S" --roi "$ROI" --code "$CODE" --mode pipeline --bids-folder $BIDS --noise ${NOISE:-cvmatched}
fi
