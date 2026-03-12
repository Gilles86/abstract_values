#!/bin/bash
#SBATCH --job-name=build_fmriprep
#SBATCH --output=/home/gdehol/logs/build_fmriprep_%A.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source /etc/profile.d/lmod.sh
module load apptainer/1.4.1
apptainer build --sandbox /shares/zne.uzh/containers/fmriprep-25.2.5 docker://nipreps/fmriprep:25.2.5
