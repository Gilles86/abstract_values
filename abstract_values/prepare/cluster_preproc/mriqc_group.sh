#!/bin/bash
#SBATCH --job-name=mriqc_group_abstractvalue
#SBATCH --output=/home/gdehol/logs/abstractvalue_mriqc_group_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=30:00

# MRIQC group-level aggregation — re-reads per-participant IQM JSONs
# under derivatives/mriqc/ and produces:
#   group_bold.tsv / group_bold.html   (BOLD IQMs aggregated)
#   group_T1w.tsv  / group_T1w.html    (T1w IQMs aggregated)
#   group_T2w.tsv  / group_T2w.html    (T2w IQMs)
#
# Doesn't need PARTICIPANT_LABEL — picks up all per-participant IQMs
# already present in the derivatives dir.
#
# Snakemake-driver-friendly: the group_*.tsv files are the rule outputs,
# so the rule re-runs whenever the input set (per-subject .mriqc_done
# sentinels) changes — which is exactly when a new subject finishes
# participant-level mriqc.

source /etc/profile
module load apptainer/1.4.1

apptainer run \
  -B /shares/zne.uzh/gdehol/ds-abstractvalue:/data \
  -B /scratch/gdehol:/workflow \
  --cleanenv /shares/zne.uzh/containers/mriqc-24.0.0 \
    /data /data/derivatives/mriqc group \
  -w /workflow/mriqc_24_group_wf

# Group mode is cheap and deterministic — no spurious-exit-1 quirk to
# handle. Trust apptainer's exit code directly.
