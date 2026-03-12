#!/bin/bash
#SBATCH --job-name=fmriprep_flair
#SBATCH --output=/home/gdehol/logs/abstractvalue_fmriprep_flair_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# FLAIR comparison — WITH FLAIR
# Uses no bids-filter-file so fmriprep picks up both T1w and FLAIR.
# Output: derivatives/fmriprep-flair
#
# Two ways to run:
#   Array job:  sbatch --array=1-30 fmriprep_flair.sh
#   By name:    sbatch --export=PARTICIPANT_LABEL=pil01 fmriprep_flair.sh
if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

source /etc/profile.d/lmod.sh
module load apptainer/1.4.1

export APPTAINERENV_FS_LICENSE=$HOME/freesurfer/license.txt

apptainer run \
  -B /shares/zne.uzh/containers/templateflow:/opt/templateflow \
  -B /shares/zne.uzh/gdehol/ds-abstractvalue:/data \
  -B /scratch/gdehol:/workflow \
  --cleanenv /shares/zne.uzh/containers/fmriprep-25.2.3 \
    /data /data/derivatives/fmriprep-flair participant \
  --participant-label $PARTICIPANT_LABEL \
  --output-spaces T1w fsnative \
  --skip_bids_validation \
  -w /workflow/fmriprep-flair \
  --nthreads 16 \
  --omp-nthreads 16 \
  --low-mem \
  --no-submm-recon
