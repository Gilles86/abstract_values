#!/bin/bash
#SBATCH --job-name=cf_variants
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/cf_variants_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Several test_coupling.sh variants for one participant in one job (each run is
# ~1 min, mostly loading betas; one job per variant floods the queue).
#
# VARIANTS: ';'-separated list of ','-separated KEY=VALUE settings, each passed
# to test_coupling.sh on top of the job's environment, e.g.
#   sbatch --array=3-35 --export=ALL,TARGET_ROI=IPS345,PROJECTION=iem,N_BASIS=24,KAPPA=16,\
#VARIANTS="TEMPLATE=basis LAGS=0;TEMPLATE=basis LAGS=1" test_coupling_variants.sh
# (inside one variant, separate settings with spaces: --export splits on commas)

if [ -z "$PARTICIPANT_LABEL" ]; then
    export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi
REPO="${REPO:-$HOME/git/abstract_values}"
fail=0
IFS=';' read -ra VS <<< "$VARIANTS"
for v in "${VS[@]}"; do
    echo "=== sub-$PARTICIPANT_LABEL: $v ==="
    env $v bash "$REPO/abstract_values/connective_fields/slurm_jobs/test_coupling.sh" || fail=1
done
exit $fail
