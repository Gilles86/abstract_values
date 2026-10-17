#!/bin/bash
#SBATCH --job-name=dropout_combine
#SBATCH --output=/home/gdehol/logs/dropout_combine_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --account=zne.uzh

# Acquisition parameters (acquisition.py, a few subjects per dataset), then the
# summary TSVs / NPZ (combine.py) into the cluster clone's notes/data, to be
# rsynced back for plotting.
#   sbatch combine.sh
#   sbatch --export=SKIP_ACQ=1 combine.sh
#   sbatch --export=SKIP_ACQ=1,GLM_ONLY=1 combine.sh     # only the GLM detectability tables
OUT=/scratch/gdehol/dropout_survey
export TMPDIR=/scratch/gdehol/tmp
mkdir -p $TMPDIR
. $HOME/init_conda.sh
conda activate abstract_values
cd $HOME/git/abstract_values
if [ -z "$SKIP_ACQ" ]; then
    PYTHONUNBUFFERED=1 python -u -m abstract_values.dropout_survey.acquisition \
        --max-subjects 6 --out $OUT/acquisition.tsv || exit 1
fi
PYTHONUNBUFFERED=1 python -u -m abstract_values.dropout_survey.combine \
    --out-dir $HOME/git/abstract_values/notes/data --scratch $OUT ${GLM_ONLY:+--glm-only}
