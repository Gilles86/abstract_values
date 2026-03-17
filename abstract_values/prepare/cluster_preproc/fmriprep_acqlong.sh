#!/bin/bash
#SBATCH --job-name=fmriprep_acqlong
#SBATCH --output=/home/gdehol/logs/abstractvalue_fmriprep_acqlong_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# FLAIR comparison — acq-long FLAIR only (+ T1w + T2w)
# Uses a BIDS filter to restrict FLAIR input to acq-long sequences.
# Output: derivatives/fmriprep-acqlong
#
# Two ways to run:
#   By name (pilot):  sbatch --export=PARTICIPANT_LABEL=pil01 fmriprep_acqlong.sh
#   Array job:        sbatch --array=1-30 fmriprep_acqlong.sh
if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

source /etc/profile.d/lmod.sh
module load apptainer/1.4.1

export APPTAINERENV_FS_LICENSE=$HOME/freesurfer/license.txt

FILTER_FILE=$(mktemp /tmp/bids_filter_acqlong_XXXXXX.json)
cat > "$FILTER_FILE" << 'EOF'
{
    "fmap":  {"datatype": "fmap"},
    "bold":  {"datatype": "func", "suffix": "bold"},
    "t1w":   {"datatype": "anat", "suffix": "T1w"},
    "flair": {"datatype": "anat", "suffix": "FLAIR", "acquisition": "long"}
}
EOF

apptainer run \
  -B /shares/zne.uzh/containers/templateflow:/opt/templateflow \
  -B /shares/zne.uzh/gdehol/ds-abstractvalue:/data \
  -B /scratch/gdehol:/workflow \
  -B ${FILTER_FILE}:/bids_filter_acqlong.json \
  --cleanenv /shares/zne.uzh/containers/fmriprep-25.2.5 \
    /data /data/derivatives/fmriprep-acqlong participant \
  --participant-label $PARTICIPANT_LABEL \
  --bids-filter-file /bids_filter_acqlong.json \
  --output-spaces T1w fsnative \
  --skip_bids_validation \
  -w /workflow/fmriprep-acqlong \
  --nthreads 16 \
  --omp-nthreads 16 \
  --low-mem \
  --no-submm-recon
