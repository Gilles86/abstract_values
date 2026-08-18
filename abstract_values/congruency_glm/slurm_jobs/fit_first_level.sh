#!/bin/bash
#SBATCH --job-name=congruency_glm
#SBATCH --output=/home/gdehol/logs/congruency_glm_%j.txt
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Congruent/incongruent value-mapping first-level GLM (nilearn).
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=06,MODEL=congruent fit_first_level.sh
#   sbatch --array=3-26 --export=MODEL=all fit_first_level.sh   # PARTICIPANT_LABEL from array index
#
# MODEL: congruent | incongruent | both | all  (default: all)
# SMOOTHING_FWHM: spatial smoothing kernel in mm (default: 6)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi
MODEL="${MODEL:-all}"
SMOOTHING_FWHM="${SMOOTHING_FWHM:-6}"
BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
REPO=$HOME/git/abstract_values

echo "congruency_glm: sub-${PARTICIPANT_LABEL} model=${MODEL} smoothing_fwhm=${SMOOTHING_FWHM}mm"

. $HOME/init_conda.sh

conda run -n abstract_values python -u \
    "$REPO/abstract_values/congruency_glm/fit_first_level.py" \
    "$PARTICIPANT_LABEL" --model "$MODEL" --bids-folder "$BIDS_FOLDER" \
    --smoothing-fwhm "$SMOOTHING_FWHM"
