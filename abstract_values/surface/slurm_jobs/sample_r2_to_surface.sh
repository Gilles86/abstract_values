#!/bin/bash
#SBATCH --job-name=sample_r2_to_surface
#SBATCH --output=/home/gdehol/logs/sample_r2_to_surface_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# Sample one encoding-model (cv)R² volume (T1w space) to fsnative + fsaverage
# surfaces for one subject. Wraps abstract_values.surface.sample_r2_to_surface.
#
# Required environment (set via `--export key=value` from sbatch, or by the
# Snakemake rule that invokes us):
#   PARTICIPANT_LABEL  subject label without sub- prefix (e.g. 08)
#   MODEL_DIR          on-disk encoder dir, e.g. aprf | vonmises | aprf.cv |
#                      aprf-weighted | aprf-shift.cv | aprf-weighted.cv
#   DESC               BIDS desc entity (default: r2; pass `cvr2` for *.cv dirs)
#
# Optional:
#   SMOOTHED           "1" to sample the _smoothed variant (default: 0)
#   FMRIPREP_DERIV     fmriprep derivative label (default: fmriprep)
#   SESSION            session number for the fmriprep anat surfaces (default: 1)
#   NO_FSAVERAGE       "1" to skip the fsnative->fsaverage transform (default: 0)
#
# FreeSurfer note
# ---------------
# The cluster's `abstract_values` conda env intentionally doesn't carry
# FreeSurfer. We reuse the binaries shipped inside the fmriprep Apptainer
# image (mounted on the host filesystem at /shares/zne.uzh/containers/...).
# That's enough to satisfy nipype's SurfaceTransform call.

set -euo pipefail

if [ -z "${PARTICIPANT_LABEL:-}" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

MODEL_DIR="${MODEL_DIR:?MODEL_DIR must be set (e.g. aprf, aprf.cv, vonmises)}"
DESC="${DESC:-r2}"
SMOOTHED="${SMOOTHED:-0}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
SESSION="${SESSION:-1}"
NO_FSAVERAGE="${NO_FSAVERAGE:-0}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

# FreeSurfer from the fmriprep container's mounted opt/freesurfer tree.
# Falls back to the older container path if the newer one is unavailable.
for candidate in \
    /shares/zne.uzh/containers/fmriprep-25.2.5/opt/freesurfer \
    /shares/zne.uzh/containers/fmriprep-25.2.3/opt/freesurfer
do
    if [ -d "$candidate" ]; then
        export FREESURFER_HOME="$candidate"
        break
    fi
done
if [ -z "${FREESURFER_HOME:-}" ]; then
    echo "ERROR: no FreeSurfer dir found under /shares/zne.uzh/containers/fmriprep-*"
    exit 1
fi
export PATH="$FREESURFER_HOME/bin:$PATH"
export FS_LICENSE=/shares/zne.uzh/containers/freesurfer/license.txt

ARGS=(
    --subjects "$PARTICIPANT_LABEL"
    --models   "$MODEL_DIR"
    --session  "$SESSION"
    --bids-folder    "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
    --desc     "$DESC"
)
[ "$SMOOTHED"     = "1" ] && ARGS+=(--smoothed)
[ "$NO_FSAVERAGE" = "1" ] && ARGS+=(--no-fsaverage)

echo "sample_r2_to_surface: sub-${PARTICIPANT_LABEL}  model_dir=${MODEL_DIR}  desc=${DESC}  smoothed=${SMOOTHED}"
echo "FREESURFER_HOME=${FREESURFER_HOME}"
echo "Args: ${ARGS[*]}"

# Use the env's python binary directly — `conda run` buffers output and
# has historically failed silently for this script (16/16 sub-07/10 jobs
# failed in <4 s with no useful stderr).
PYTHONUNBUFFERED=1 $HOME/data/conda/envs/abstract_values/bin/python -u \
    -m abstract_values.surface.sample_r2_to_surface \
    "${ARGS[@]}"
