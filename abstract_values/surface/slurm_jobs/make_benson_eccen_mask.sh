#!/bin/bash
#SBATCH --job-name=benson_eccen_mask
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/benson_eccen_mask_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#
# Eccentricity-restricted Benson V1 mask (0.75-3.75 deg, the annulus the gabor
# drives) in T1w space. Needs infer_neuropythy_atlas.sh to have run first.
#
# Usage:
#   sbatch --export=ALL,PARTICIPANT_LABEL=30 abstract_values/surface/slurm_jobs/make_benson_eccen_mask.sh
#
# Optional exports: SESSION (default 1), REPO (default ~/git/abstract_values)

set -euo pipefail

if [ -z "${PARTICIPANT_LABEL:-}" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
fi
SESSION="${SESSION:-1}"
BIDS_FOLDER="${BIDS_FOLDER:-/shares/zne.uzh/gdehol/ds-abstractvalue}"
REPO="${REPO:-$HOME/git/abstract_values}"

export FREESURFER_HOME=/shares/zne.uzh/containers/fmriprep-25.2.3/opt/freesurfer
export FS_LICENSE=/shares/zne.uzh/containers/freesurfer/license.txt
export PATH="$FREESURFER_HOME/bin:$PATH"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "Host: $(hostname) | sub-${PARTICIPANT_LABEL} ses-${SESSION}"
cd "$REPO"
PYTHONUNBUFFERED=1 "$HOME/data/conda/envs/abstract_values/bin/python" -u \
    -m abstract_values.surface.make_benson_eccen_mask \
    "$PARTICIPANT_LABEL" --session "$SESSION" --bids-folder "$BIDS_FOLDER"
