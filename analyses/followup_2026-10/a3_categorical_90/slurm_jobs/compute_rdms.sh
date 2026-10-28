#!/bin/bash
#SBATCH --job-name=a3_rdms
#SBATCH --output=/home/gdehol/logs/a3_rdms_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --account=zne.uzh
#SBATCH --partition=standard

# A3: crossnobis (all/odd/even runs) + model-implied RDMs per subject.
#   sbatch --array=0-35 compute_rdms.sh      (index into SUBJECTS)
#   sbatch --export=ALL,PARTICIPANT_LABEL=03 --array=0 compute_rdms.sh
#   sbatch --export=ALL,SCRIPT=compute_controls.py --array=0-35 compute_rdms.sh   (univariate control)
set -eo pipefail
SUBJECTS=(pil01 pil02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36)
SUBJ="${PARTICIPANT_LABEL:-${SUBJECTS[$SLURM_ARRAY_TASK_ID]}}"
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TF_CPP_MIN_LOG_LEVEL=2
cd "$HOME/git/abstract_values"
exec "$HOME/data/conda/envs/abstract_values/bin/python" -u \
    analyses/followup_2026-10/a3_categorical_90/${SCRIPT:-compute_rdms.py} "$SUBJ" \
    --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue
