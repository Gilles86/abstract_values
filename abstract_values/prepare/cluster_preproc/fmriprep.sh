#!/bin/bash -l
#SBATCH --job-name=fmriprep_abstractvalue
#SBATCH --output=/home/gdehol/logs/abstractvalue_fmriprep_%A-%a.txt
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# 8 CPUs: measured ~1.07x slower than 16 at about half the CPU cost; memory sized from real
# (anon) use, not sacct MaxRSS. See the fmriprep skill, references/resources.md.

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
#     PARTICIPANT_LABEL=pil02 sbatch fmriprep.sh
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

# Tell fMRIPrep its allocation (otherwise nipype plans with the whole node's RAM and cores).
NCPU=${SLURM_CPUS_PER_TASK:-8}
MEM_MB=$(( ${SLURM_MEM_PER_NODE:-65536} - 8000 ))
STDOUT_COPY="/scratch/$USER/fmriprep_stdout_sub-${PARTICIPANT_LABEL}_${SLURM_JOB_ID:-manual}.log"

set -o pipefail
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
  --nthreads "$NCPU" \
  --omp-nthreads "$NCPU" \
  --mem-mb "$MEM_MB" \
  --low-mem \
  --no-submm-recon \
  $EXTRA_ARGS 2>&1 | tee "$STDOUT_COPY"
RC=${PIPESTATUS[0]}

# apptainer >= 1.4 can exit 1 after a clean run (value_capture, 2026-11-14: all 12 subjects),
# and fMRIPrep writes an html on failures too, so neither is proof of success. fMRIPrep's own
# last line, printed only after a successful run, is. See the fmriprep skill.
if [[ $RC -ne 0 ]]; then
    if grep -q "fMRIPrep finished successfully" "$STDOUT_COPY"; then
        echo "apptainer exited $RC but fMRIPrep reported success: treating as a clean run."
    else
        echo "apptainer exited $RC and fMRIPrep did not report success: failed."
        exit $RC
    fi
fi

# Completion sentinel — touched ONLY after the discriminator above passes,
# so its existence is a hard guarantee fmriprep ran to its end. See the
# **fmriprep** skill, section "HTML-report vs NIfTI ground truth".
DONE="/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/fmriprep/sub-${PARTICIPANT_LABEL}/.fmriprep_done"
mkdir -p "$(dirname "$DONE")"
touch "$DONE"
echo "fmriprep done sentinel: $DONE"
