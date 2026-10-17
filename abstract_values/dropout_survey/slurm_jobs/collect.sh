#!/bin/bash
#SBATCH --job-name=dropout_survey
#SBATCH --output=/home/gdehol/logs/dropout_survey_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --account=zne.uzh

# tSNR / mean-EPI / FOV maps and ROI dropout metrics for one (dataset, subject)
# (dropout_survey/collect.py), then warps the maps to MNI152NLin2009cAsym 2 mm
# with ANTs from the fmriprep container.  Subjects are read from
#   /scratch/gdehol/dropout_survey/subjects_<DATASET>.txt   (written by inventory)
#   sbatch --array=1-30%15 --export=DATASET=numrisk collect.sh
#   sbatch --export=DATASET=numrisk,PARTICIPANT_LABEL=10 collect.sh
# Read-only on all dataset folders; writes only to /scratch/gdehol/dropout_survey.

OUT=/scratch/gdehol/dropout_survey
if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $OUT/subjects_${DATASET}.txt)
fi
S=$PARTICIPANT_LABEL
export TMPDIR=/scratch/gdehol/tmp
mkdir -p $TMPDIR
sleep $(( RANDOM % 20 ))

. $HOME/init_conda.sh
conda activate abstract_values
cd $HOME/git/abstract_values
PYTHONUNBUFFERED=1 python -u -m abstract_values.dropout_survey.collect "$DATASET" "$S" \
    --out-dir $OUT || exit 1

D=$OUT/$DATASET/sub-$S
XFM=$(python -c "import json; print(json.load(open('$D/sub-${S}_meta.json'))['xfm'] or '')")
if [ -z "$XFM" ]; then
    echo "no T1w->MNI transform: maps stay in T1w space"; exit 0
fi
source /etc/profile
module load apptainer/1.4.1
TPL=/shares/zne.uzh/containers/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz
for d in tsnrnorm meanepi fov; do
    apptainer exec -B /shares/zne.uzh -B /scratch/gdehol /shares/zne.uzh/containers/fmriprep-25.2.3 \
        antsApplyTransforms -d 3 -i $D/sub-${S}_space-T1w_desc-${d}_map.nii.gz -r $TPL -t $XFM \
        -n Linear -o $D/sub-${S}_space-MNI152NLin2009cAsym_res-2_desc-${d}_map.nii.gz --float 1
done
ls -la $D
