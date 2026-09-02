#!/bin/bash
#SBATCH --job-name=sample_aprf_to_surface
#SBATCH --output=/home/gdehol/logs/sample_aprf_to_surface_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Sample the aPRF *parameter* volumes (T1w space) to fsnative + fsaverage
# surfaces for one subject. Wraps abstract_values.surface.sample_aprf_to_surface.
#
# This is the companion to sample_r2_to_surface.sh, which only ever samples
# (cv)R². Everything that describes the tuning itself — mode (preferred value
# in CHF), fwhm, amplitude, baseline, plus the von Mises R² as `gabor-r2` —
# comes from here.
#
# Required environment (set via `--export key=value`, or by the Snakemake rule):
#   PARTICIPANT_LABEL  subject label without sub- prefix (e.g. 08)
#
# Optional:
#   SMOOTHED           "1" to sample the _smoothed variant (default: 0)
#   FMRIPREP_DERIV     fmriprep derivative label (default: fmriprep)
#   SESSION            session number for the fmriprep anat surfaces (default: 1)
#
# FreeSurfer note
# ---------------
# Same as sample_r2_to_surface.sh: the cluster's `abstract_values` env carries
# no FreeSurfer, so we reuse the binaries inside the fmriprep Apptainer image
# (extracted on the host filesystem, so they run directly). nipype's
# SurfaceTransform needs `mri_surf2surf` on PATH and a valid FS_LICENSE.

set -euo pipefail

if [ -z "${PARTICIPANT_LABEL:-}" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

SMOOTHED="${SMOOTHED:-0}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
SESSION="${SESSION:-1}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

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

ARGS=("$PARTICIPANT_LABEL" --session "$SESSION"
      --bids-folder "$BIDS_FOLDER" --fmriprep-deriv "$FMRIPREP_DERIV")
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)

echo "sample_aprf_to_surface: sub-${PARTICIPANT_LABEL}  smoothed=${SMOOTHED}"
echo "FREESURFER_HOME=${FREESURFER_HOME}"
echo "Args: ${ARGS[*]}"

# Direct env binary, not `conda run` — the latter buffers output and has
# historically failed silently for these jobs.
PYTHONUNBUFFERED=1 $HOME/data/conda/envs/abstract_values/bin/python -u \
    -m abstract_values.surface.sample_aprf_to_surface \
    "${ARGS[@]}"
