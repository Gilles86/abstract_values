#!/bin/bash
#SBATCH --job-name=score_heldout
#SBATCH --output=/home/gdehol/logs/score_heldout_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --account=zne.uzh

# Split-half validation (score_heldout.py): fits on the odd trials of each
# subject x mapping x orientation cell, scored on the even trials.
#   sbatch --dependency=afterok:<fit jobs> score_heldout.sh
D=/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/cogmodels_n35/split-odd
P=$HOME/git/abstract_values/notes/data
B=efficient_coding_categorical_freeprior_grid161_trunc_hn_total-share_erf
OUT=${OUT:-/scratch/gdehol/heldout}
export TMPDIR=/scratch/gdehol/tmp/${SLURM_JOB_ID:-manual}
export PYTENSOR_FLAGS="base_compiledir=/scratch/gdehol/pytensor/${SLURM_JOB_ID:-manual}"
mkdir -p $TMPDIR $OUT
cd $HOME/git/abstract_values
PYTHONUNBUFFERED=1 exec $HOME/data/conda/envs/bauer/bin/python -u -m abstract_values.cogmodels.score_heldout \
    --train-tsv $P/efficient_coding_paradigm_n35_split-odd.tsv \
    --test-tsv $P/efficient_coding_paradigm_n35_split-even.tsv \
    --fit "No offsets=$D/${B}_shrink_freelapse_trace.nc:trunc,prior_weight,shrink,freelapse" \
    --fit "Per-cell offsets=$D/${B}_shrink_offsets_freelapse_trace.nc:trunc,prior_weight,shrink,freelapse,offsets" \
    --fit "Spline offsets=$D/${B}_shrink_splineoffsets_freelapse_trace.nc:trunc,prior_weight,shrink,freelapse,splineoffsets" \
    --fit "Spline offsets, group mean, no shrinkage=$D/${B}_splineoffsets_offsetmean_freelapse_trace.nc:trunc,prior_weight,freelapse,splineoffsets,offsetmean" \
    --baseline "No offsets" --out-dir $OUT
