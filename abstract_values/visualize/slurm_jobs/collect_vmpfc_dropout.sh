#!/bin/bash
#SBATCH --job-name=vmpfc_dropout
#SBATCH --output=/home/gdehol/logs/vmpfc_dropout_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --account=zne.uzh

# tSNR / mean-EPI maps, ROI summaries, peri-event epochs and display slices
# for one participant (collect_vmpfc_dropout.py), then warps the maps and the
# mPFC label volume to MNI152NLin2009cAsym (2 mm) with ANTs from the fmriprep
# container, for the group dropout map.
#   sbatch --array=1-35 collect_vmpfc_dropout.sh          # index into glmsingle subjects
#   sbatch --export=PARTICIPANT_LABEL=03 collect_vmpfc_dropout.sh

BIDS=/shares/zne.uzh/gdehol/ds-abstractvalue
OUT=/scratch/gdehol/vmpfc_dropout
if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(ls -d $BIDS/derivatives/glmsingle/sub-* | sed 's|.*/sub-||' | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
fi
S=$PARTICIPANT_LABEL
export TMPDIR=/scratch/gdehol/tmp
mkdir -p $TMPDIR $OUT

. $HOME/init_conda.sh
conda activate abstract_values
cd $HOME/git/abstract_values
PYTHONUNBUFFERED=1 python -u -m abstract_values.visualize.collect_vmpfc_dropout "$S" \
    --bids-folder $BIDS --out-dir $OUT || exit 1

source /etc/profile
module load apptainer/1.4.1
QA=$BIDS/derivatives/qa/tsnr/sub-$S
XFM=$(ls $BIDS/derivatives/fmriprep/sub-$S/*/anat/sub-${S}_*from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5 \
          $BIDS/derivatives/fmriprep/sub-$S/anat/sub-${S}_*from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5 2>/dev/null | head -1)
TPL=/shares/zne.uzh/containers/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz
run_ants() {
    apptainer exec -B /shares/zne.uzh /shares/zne.uzh/containers/fmriprep-25.2.3 \
        antsApplyTransforms -d 3 -i "$1" -r $TPL -t $XFM -n "$3" -o "$2" --float 1
}
for d in tsnr meanepi; do
    run_ants $QA/sub-${S}_task-abstractvalue_space-T1w_desc-${d}_map.nii.gz \
             $QA/sub-${S}_task-abstractvalue_space-MNI152NLin2009cAsym_res-2_desc-${d}_map.nii.gz Linear
done
run_ants $QA/sub-${S}_space-T1w_desc-mpfclabels_dseg.nii.gz \
         $QA/sub-${S}_space-MNI152NLin2009cAsym_res-2_desc-mpfclabels_dseg.nii.gz GenericLabel
ls -la $QA
