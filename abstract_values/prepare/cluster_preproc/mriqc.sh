#!/bin/bash
#SBATCH --job-name=mriqc_abstractvalue
#SBATCH --output=/home/gdehol/logs/abstractvalue_mriqc_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# MRIQC participant-level QC pipeline — per-scan IQMs + HTML reports.
# Output: derivatives/mriqc
#
# Two ways to run:
#
#   Numeric subjects (array job):
#     sbatch --array=1-30 mriqc.sh
#     -> labels 001, 002, ..., 030
#
#   Any subject by name (single job, overrides array):
#     sbatch --export=PARTICIPANT_LABEL=pil02 mriqc.sh
#
# Lessons reapplied from the fmriprep wrapper:
# - source /etc/profile so MODULEPATH is populated when snakemake's slurm
#   plugin submits with a cleaner env than a login shell.
# - Trust apptainer's exit code, but discriminate spurious exit-1 from a
#   real failure with an output-existence check (the anat-report HTML).
# - Touch a .mriqc_done sentinel ONLY after the discriminator passes, so
#   Snakemake's "is this rule done" check is based on a true completion
#   marker not a possibly-half-written file. (See **fmriprep** skill,
#   section "The html exists ≠ fmriprep succeeded".)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
fi

source /etc/profile
module load apptainer/1.4.1

apptainer run \
  -B /shares/zne.uzh/gdehol/ds-abstractvalue:/data \
  -B /scratch/gdehol:/workflow \
  --cleanenv /shares/zne.uzh/containers/mriqc-24.0.0 \
    /data /data/derivatives/mriqc participant \
  --participant-label $PARTICIPANT_LABEL \
  -w /workflow/mriqc_24_wf \
  --nprocs 8 \
  --omp-nthreads 8 \
  --mem 30
APPTAINER_RC=$?

# Discriminator: mriqc writes ses-1 T1w report HTML at the very end of
# the anat workflow. Present on truly-finished runs; absent on early
# failures. Mirrors the fmriprep wrapper's aparcaseg-discriminator
# pattern — see the **fmriprep** skill section "The html exists ≠
# fmriprep succeeded".
ANAT_REPORT="/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/mriqc/sub-${PARTICIPANT_LABEL}_ses-1_T1w.html"
if [[ $APPTAINER_RC -ne 0 ]]; then
    if [[ -f "$ANAT_REPORT" ]]; then
        echo "apptainer exit $APPTAINER_RC tolerated — anat report exists ($ANAT_REPORT)."
    else
        echo "apptainer exit $APPTAINER_RC and anat report missing ($ANAT_REPORT) — mriqc failed."
        exit $APPTAINER_RC
    fi
fi

DONE="/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/mriqc/sub-${PARTICIPANT_LABEL}/.mriqc_done"
mkdir -p "$(dirname "$DONE")"
touch "$DONE"
echo "mriqc done sentinel: $DONE"
