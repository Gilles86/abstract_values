#!/bin/bash
#SBATCH --job-name=roi_ts
#SBATCH --output=/home/gdehol/logs/roi_ts_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --account=zne.uzh

# ROI-mean time courses + confounds + events for nideconv (collect_roi_timeseries.py).
#   sbatch --array=1-35 collect_roi_timeseries.sh
BIDS=/shares/zne.uzh/gdehol/ds-abstractvalue
if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(ls -d $BIDS/derivatives/glmsingle/sub-* | sed 's|.*/sub-||' | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
fi
export TMPDIR=/scratch/gdehol/tmp
sleep $(( RANDOM % 20 ))
cd $HOME/git/abstract_values
PYTHONUNBUFFERED=1 exec $HOME/data/conda/envs/abstract_values/bin/python -u \
    -m abstract_values.visualize.collect_roi_timeseries "$PARTICIPANT_LABEL" \
    --bids-folder $BIDS --out-dir /scratch/gdehol/vmpfc_roi_ts
