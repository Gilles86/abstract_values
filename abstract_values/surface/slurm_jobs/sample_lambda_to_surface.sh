#!/bin/bash
#SBATCH --job-name=lamsurf
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/lamsurf_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --exclude=u24-chaihm0-651

# Whole-brain lambda cvR^2 profile -> fsaverage, both smoothing variants
# (surface/sample_lambda_to_surface.py).
#   sbatch --export=ALL,PARTICIPANT_LABEL=05 sample_lambda_to_surface.sh

set -eo pipefail
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
for candidate in /shares/zne.uzh/containers/fmriprep-25.2.5/opt/freesurfer \
                 /shares/zne.uzh/containers/fmriprep-25.2.3/opt/freesurfer; do
    [ -d "$candidate" ] && { export FREESURFER_HOME="$candidate"; break; }
done
export PATH="$FREESURFER_HOME/bin:$PATH"
export FS_LICENSE=/shares/zne.uzh/containers/freesurfer/license.txt
REPO="${REPO:-$HOME/git/abstract_values}"
cd "$REPO"
exec $HOME/data/conda/envs/abstract_values/bin/python -u -W ignore -m \
    abstract_values.surface.sample_lambda_to_surface "$PARTICIPANT_LABEL" \
    --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue
