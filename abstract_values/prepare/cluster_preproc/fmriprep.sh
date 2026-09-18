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

# NOTAL_CHECK=1 bind-mounts a recon-all patched to skip FreeSurfer's
# talairach_afd goodness-of-fit gate (equivalent to `-notal-check`, which
# fmriprep does not expose). Needed for subjects whose head was rotated far
# enough in the scanner to false-positive the check — sub-27 is the first.
# Build the patched copy with make_notalcheck_reconall.sh. See that script
# for the full rationale and the evidence that it's a false positive.
NOTAL_BIND=""
if [ "${NOTAL_CHECK:-0}" = "1" ]; then
    NOTAL_RECONALL=/shares/zne.uzh/gdehol/container_patches/recon-all-notalcheck
    if [ ! -x "$NOTAL_RECONALL" ]; then
        echo "ERROR: NOTAL_CHECK=1 but $NOTAL_RECONALL is missing."
        echo "Run make_notalcheck_reconall.sh first."
        exit 1
    fi
    NOTAL_BIND="-B ${NOTAL_RECONALL}:/opt/freesurfer/bin/recon-all"
    echo "NOTAL_CHECK=1 — binding patched recon-all (talairach_afd disabled)."
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
  $NOTAL_BIND \
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

# Treat the apptainer exit code as the primary signal, but tolerate
# spurious exit-1 from apptainer ≥ 1.4 on otherwise-clean runs. The naive
# "if exit-nonzero and html exists, treat as clean" version got bitten by
# fmriprep writing a *failure-report* HTML on real failures (sub-12 /
# 3573299 on 2026-05-29: ZRAN_READ_FAIL, `fMRIPrep failed: 15 raised`,
# apptainer exit 1, *html written anyway*). The discriminator below
# adds a late-stage output check that's only written after the freesurfer
# autorecon completes — present on truly-finished runs, absent on early
# failures like ZRAN.
APARCASEG="/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/fmriprep/sub-${PARTICIPANT_LABEL}/ses-1/anat/sub-${PARTICIPANT_LABEL}_ses-1_desc-aparcaseg_dseg.nii.gz"
if [[ $APPTAINER_RC -ne 0 ]]; then
    if [[ -f "$APARCASEG" ]]; then
        echo "apptainer exited $APPTAINER_RC but late-stage anat output exists ($APARCASEG) — treating as spurious exit, clean run."
    else
        echo "apptainer exited $APPTAINER_RC and late-stage anat output missing ($APARCASEG) — fmriprep failed."
        exit $APPTAINER_RC
    fi
fi

# Completion sentinel — touched ONLY after the discriminator above passes,
# so its existence is a hard guarantee fmriprep ran to its end. See the
# **fmriprep** skill, section "HTML-report vs NIfTI ground truth".
DONE="/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/fmriprep/sub-${PARTICIPANT_LABEL}/.fmriprep_done"
mkdir -p "$(dirname "$DONE")"
touch "$DONE"
echo "fmriprep done sentinel: $DONE"
