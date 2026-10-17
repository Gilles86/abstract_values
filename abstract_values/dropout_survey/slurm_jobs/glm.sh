#!/bin/bash
#SBATCH --job-name=dropout_glm
#SBATCH --output=/home/gdehol/logs/dropout_glm_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --account=zne.uzh

# GLMsingle R2 / single-trial-beta detectability per survey ROI for one
# (dataset, subject) (dropout_survey/glm.py).  Subjects = those already in the
# dropout survey (/scratch/gdehol/dropout_survey/<DATASET>/sub-*), sorted.
#   sbatch --array=1-30%10 --export=DATASET=numrisk glm.sh
#   sbatch --export=DATASET=numrisk,PARTICIPANT_LABEL=10 glm.sh
OUT=/scratch/gdehol/dropout_survey
if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(ls -d $OUT/$DATASET/sub-* | sed 's|.*/sub-||' | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
fi
export TMPDIR=/scratch/gdehol/tmp
mkdir -p $TMPDIR
sleep $(( RANDOM % 20 ))
. $HOME/init_conda.sh
conda activate abstract_values
cd $HOME/git/abstract_values
PYTHONUNBUFFERED=1 python -u -m abstract_values.dropout_survey.glm "$DATASET" "$PARTICIPANT_LABEL" \
    --out-dir $OUT
