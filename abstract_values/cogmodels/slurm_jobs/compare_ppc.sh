#!/bin/bash
#SBATCH --job-name=compare_ppc
#SBATCH --output=/home/gdehol/logs/compare_ppc_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --account=zne.uzh

# PPC + LOO for the n=35 categorical fits: fixed long-term prior vs free prior
# steepness, mapping shrinkage, free lapse, mapping offsets (compare_ppc.py).
# Outputs to $OUT (default /scratch/gdehol/compare_ppc).
#   sbatch compare_ppc.sh
D=/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/cogmodels_n35
OUT=${OUT:-/scratch/gdehol/compare_ppc}
export TMPDIR=/scratch/gdehol/tmp/${SLURM_JOB_ID:-manual}
export PYTENSOR_FLAGS="base_compiledir=/scratch/gdehol/pytensor/${SLURM_JOB_ID:-manual}"
mkdir -p $TMPDIR $OUT
cd $HOME/git/abstract_values
PYTHONUNBUFFERED=1 exec $HOME/data/conda/envs/bauer/bin/python -u -m abstract_values.cogmodels.compare_ppc \
    --paradigm-tsv notes/data/efficient_coding_paradigm_n35.tsv \
    --fit "Fixed prior=$D/efficient_coding_categorical_grid161_trunc_hn_total-share_erf_trace.nc:trunc" \
    --fit "Free prior steepness=$D/efficient_coding_categorical_freeprior_grid161_trunc_hn_total-share_erf_trace.nc:trunc,prior_weight" \
    --fit "Free prior + shrinkage=$D/efficient_coding_categorical_freeprior_grid161_trunc_hn_total-share_erf_shrink_trace.nc:trunc,prior_weight,shrink" \
    --fit "+ free lapse=$D/efficient_coding_categorical_freeprior_grid161_trunc_hn_total-share_erf_shrink_freelapse_trace.nc:trunc,prior_weight,shrink,freelapse" \
    --fit "+ free lapse + offsets=$D/efficient_coding_categorical_freeprior_grid161_trunc_hn_total-share_erf_shrink_offsets_freelapse_trace.nc:trunc,prior_weight,shrink,freelapse,offsets" \
    --fit "+ free lapse + spline offsets=$D/efficient_coding_categorical_freeprior_grid161_trunc_hn_total-share_erf_shrink_splineoffsets_freelapse_trace.nc:trunc,prior_weight,shrink,freelapse,splineoffsets" \
    --fit "Free prior + free lapse + spline offsets with group mean (no shrinkage)=$D/efficient_coding_categorical_freeprior_grid161_trunc_hn_total-share_erf_splineoffsets_offsetmean_freelapse_trace.nc:trunc,prior_weight,freelapse,splineoffsets,offsetmean" \
    --out-dir $OUT
