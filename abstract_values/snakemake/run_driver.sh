#!/bin/bash
#SBATCH --job-name=abstract_values_snake_driver
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gdehol/logs/snake_driver_%j.log
#
# Snakemake driver for the abstract_values cluster pipeline.
# Runs as its own SLURM job so the long-running driver process isn't subject
# to login-node ulimits (see snakemake skill: "driver placement").
#
# Usage:
#   ssh sciencecluster 'cd ~/git/abstract_values && \
#       sbatch abstract_values/snakemake/run_driver.sh'
#
# Re-submitting the same script resumes where the previous driver left off
# (Snakemake persistence under .snakemake/).

set -eo pipefail

cd "$HOME/git/abstract_values"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate abstract_values     # snakemake + plugin pip-installed here

exec snakemake \
    --snakefile abstract_values/snakemake/Snakefile \
    --workflow-profile abstract_values/snakemake/profile \
    --configfile abstract_values/snakemake/config.yaml \
    --rerun-incomplete
