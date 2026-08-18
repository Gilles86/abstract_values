#!/bin/bash
#SBATCH --job-name=congruency_glm_group
#SBATCH --output=/home/gdehol/logs/congruency_glm_group_%j.txt
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# MNI-normalize every subject's congruency_glm first-level effect maps, then
# fit the group (second-level) one-sample tests. Meant to be submitted with
# --dependency=afterok:<first-level job ids> so it only runs once all
# first-levels are done.
#
# Usage:
#   sbatch --dependency=afterok:JID1:JID2:... normalize_and_second_level.sh

SUBJECTS=(03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 pil01 pil02)
REPO=$HOME/git/abstract_values

. $HOME/init_conda.sh

for SUB in "${SUBJECTS[@]}"; do
    echo "=== normalizing sub-${SUB} ==="
    conda run -n abstract_values python -u \
        "$REPO/abstract_values/congruency_glm/normalize_to_mni.py" \
        "$SUB" --model all
done

echo "=== fitting second-level (group) models ==="
conda run -n abstract_values python -u \
    "$REPO/abstract_values/congruency_glm/fit_second_level.py" \
    --subjects "${SUBJECTS[@]}"
