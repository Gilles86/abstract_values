#!/bin/bash
#SBATCH --job-name=fmriprep_t2w
#SBATCH --output=/home/gdehol/logs/abstractvalue_fmriprep_t2w_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# T2w-assisted FreeSurfer — T1w + T2w, no FLAIR
# Passes T2w to FreeSurfer for T2-pial surface correction.
# Output: derivatives/fmriprep-t2w
#
# Two ways to run:
#   By name (pilot):  sbatch --export=PARTICIPANT_LABEL=pil02 fmriprep_t2w.sh
#   Array job:        sbatch --array=1-30 fmriprep_t2w.sh
if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

source /etc/profile.d/lmod.sh
module load apptainer/1.4.1

export APPTAINERENV_FS_LICENSE=$HOME/freesurfer/license.txt

FILTER_FILE=$(mktemp /tmp/bids_filter_t2w_XXXXXX.json)
cat > "$FILTER_FILE" << 'EOF'
{
    "fmap": {"datatype": "fmap"},
    "bold": {"datatype": "func", "suffix": "bold"},
    "t1w":  {"datatype": "anat", "suffix": "T1w"},
    "t2w":  {"datatype": "anat", "suffix": "T2w"}
}
EOF

apptainer run \
  -B /shares/zne.uzh/containers/templateflow:/opt/templateflow \
  -B /shares/zne.uzh/gdehol/ds-abstractvalue:/data \
  -B /scratch/gdehol:/workflow \
  -B ${FILTER_FILE}:/bids_filter_t2w.json \
  --cleanenv /shares/zne.uzh/containers/fmriprep-25.2.5 \
    /data /data/derivatives/fmriprep-t2w participant \
  --participant-label $PARTICIPANT_LABEL \
  --bids-filter-file /bids_filter_t2w.json \
  --output-spaces T1w fsnative \
  --skip_bids_validation \
  -w /workflow/fmriprep-t2w \
  --nthreads 16 \
  --omp-nthreads 16 \
  --low-mem \
  --no-submm-recon
