#!/bin/bash
#SBATCH --job-name=fmriprep_abstractvalue
#SBATCH --output=/home/gdehol/logs/abstractvalue_fmriprep_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Main fmriprep pipeline — T1w + T2w (T2-pial surface correction).
# Output: derivatives/fmriprep
#
# Two ways to run:
#
#   Numeric subjects (array job):
#     sbatch --array=1-30 fmriprep.sh
#     -> labels 001, 002, ..., 030
#
#   Any subject by name (single job, overrides array):
#     sbatch --export=PARTICIPANT_LABEL=pil02 fmriprep.sh
#
if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

EXTRA_ARGS=""
if [ -n "$BOLD2ANAT_INIT" ]; then
    EXTRA_ARGS="--bold2anat-init $BOLD2ANAT_INIT"
fi

# `source /etc/profile.d/lmod.sh` alone defines `module` but leaves
# MODULEPATH empty — module load works for interactive sbatch (which
# inherits MODULEPATH from the submitting login shell), but fails when
# the job is submitted from a cleaner env (e.g. snakemake-executor-plugin-
# slurm). `source /etc/profile` sources the whole chain incl. MODULEPATH.
# See: sciencecluster skill, section "module in SLURM scripts".
source /etc/profile
module load apptainer/1.4.1

export APPTAINERENV_FS_LICENSE=$HOME/freesurfer/license.txt

FILTER_FILE=$(mktemp /tmp/bids_filter_XXXXXX.json)
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
  -B ${FILTER_FILE}:/bids_filter.json \
  --cleanenv /shares/zne.uzh/containers/fmriprep-25.2.5 \
    /data /data/derivatives/fmriprep participant \
  --participant-label $PARTICIPANT_LABEL \
  --bids-filter-file /bids_filter.json \
  --output-spaces T1w fsnative \
  --skip_bids_validation \
  -w /workflow \
  --nthreads 16 \
  --omp-nthreads 16 \
  --low-mem \
  --no-submm-recon \
  $EXTRA_ARGS
APPTAINER_RC=$?

# Trust apptainer's exit code as the source of truth. The previous
# "if exit-nonzero but html exists, treat as clean" fallback was misleading
# us — fmriprep writes a *failure-report* HTML when its workflow raises
# (e.g. sub-12 / 3573299 on 2026-05-29: ZRAN_READ_FAIL on a workflow .nii.gz,
# `fMRIPrep failed: 15 raised`, apptainer exit 1; the html-tolerance branch
# silently exited 0 and Snakemake marked the rule as done despite missing
# preproc_T2w in ses-2 anat). Genuine spurious exit-1 from apptainer is rare
# enough that a manual rerun is the right fix.
if [[ $APPTAINER_RC -ne 0 ]]; then
    echo "apptainer exited $APPTAINER_RC — fmriprep failed."
    exit $APPTAINER_RC
fi

# Completion sentinel — touched ONLY on apptainer exit 0, so its existence
# is a hard guarantee fmriprep ran to its end. See the **fmriprep** skill,
# section "HTML-report vs NIfTI ground truth".
DONE="/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/fmriprep/sub-${PARTICIPANT_LABEL}/.fmriprep_done"
mkdir -p "$(dirname "$DONE")"
touch "$DONE"
echo "fmriprep done sentinel: $DONE"
