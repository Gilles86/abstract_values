#!/bin/bash
#SBATCH --job-name=fmriprep_noflair
#SBATCH --output=/home/gdehol/logs/abstractvalue_fmriprep_noflair_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# FLAIR comparison — WITHOUT FLAIR
# Uses bids-filter-file that restricts anatomical input to T1w only.
# Output: derivatives/fmriprep-noflair
#
# Two ways to run:
#   Array job:  sbatch --array=1-30 fmriprep_noflair.sh
#   By name:    sbatch --export=PARTICIPANT_LABEL=pil01 fmriprep_noflair.sh
if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

source /etc/profile.d/lmod.sh
module load apptainer/1.4.1

export APPTAINERENV_FS_LICENSE=$HOME/freesurfer/license.txt

# Write the bids filter inline so no external path is needed
FILTER_FILE=$(mktemp /tmp/bids_filter_noflair_XXXXXX.json)
cat > "$FILTER_FILE" << 'EOF'
{
    "fmap": {"datatype": "fmap"},
    "bold": {"datatype": "func", "suffix": "bold"},
    "t1w":  {"datatype": "anat", "suffix": "T1w"}
}
EOF

apptainer run \
  -B /shares/zne.uzh/containers/templateflow:/opt/templateflow \
  -B /shares/zne.uzh/gdehol/ds-abstractvalue:/data \
  -B /scratch/gdehol:/workflow \
  -B ${FILTER_FILE}:/bids_filter_noflair.json \
  --cleanenv /shares/zne.uzh/containers/fmriprep-25.2.5 \
    /data /data/derivatives/fmriprep-noflair participant \
  --participant-label $PARTICIPANT_LABEL \
  --bids-filter-file /bids_filter_noflair.json \
  --output-spaces T1w fsnative \
  --skip_bids_validation \
  -w /workflow/fmriprep-noflair \
  --nthreads 16 \
  --omp-nthreads 16 \
  --low-mem \
  --no-submm-recon
