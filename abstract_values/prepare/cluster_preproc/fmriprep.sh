#!/bin/bash
#SBATCH --job-name=fmriprep_abstractvalue
#SBATCH --output=/home/gdehol/logs/abstractvalue_fmriprep_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ── participant label ──────────────────────────────────────────────────────────
# Two ways to run:
#
#   Numeric subjects (array job):
#     sbatch --array=1-30 fmriprep.sh
#     -> labels 001, 002, ..., 030
#
#   Any subject by name (single job, overrides array):
#     sbatch --export=PARTICIPANT_LABEL=pil001 fmriprep.sh
#
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
  --cleanenv /shares/zne.uzh/containers/fmriprep-25.2.5 \
    /data /data/derivatives/fmriprep participant \
  --participant-label $PARTICIPANT_LABEL \
  --output-spaces T1w fsnative \
  --skip_bids_validation \
  -w /workflow \
  --nthreads 16 \
  --omp-nthreads 16 \
  --low-mem \
  --no-submm-recon
