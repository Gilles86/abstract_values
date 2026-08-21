#!/bin/bash
#SBATCH --job-name=abstract_values_snake_driver
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00          # walltime is handled by the self-resubmit chain below
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
# The driver resubmits itself (see "self-resubmit chain"), so a single sbatch
# keeps the pipeline advancing unattended across walltime kills and failed
# rules. Re-submitting by hand also works — Snakemake resumes from
# .snakemake/ persistence either way.
#
# ONLY EVER ONE DRIVER AT A TIME per repo workdir: the --unlock below rips the
# lock from a live driver. The chain is safe (each link waits on the previous
# via --dependency=afterany), but do not hand-sbatch a second driver while a
# chain is alive. Check first:
#     squeue -u gdehol -h -o "%j %T" | grep -i snake

set -eo pipefail

cd "$HOME/git/abstract_values"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate abstract_values     # snakemake + plugin pip-installed here

# Keep temporary files off /tmp. On this cluster /tmp is the shared
# `cluster_tmp` network filesystem with a per-user quota, and a pipeline with
# fmriprep + GLMsingle running concurrently blows through it: on 2026-08-20
# three driver generations died with `OSError: [Errno 122] Disk quota exceeded`
# writing the PuLP scheduler's MPS file to /tmp. /scratch has a 20 TB quota.
# Exported so the workers Snakemake sbatches inherit it too — they are the ones
# actually filling /tmp.
export TMPDIR="/scratch/gdehol/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

SNAKE_ARGS=(
    --snakefile  abstract_values/snakemake/Snakefile
    --workflow-profile abstract_values/snakemake/profile
    --configfile abstract_values/snakemake/config.yaml
)

# ── self-resubmit chain ──────────────────────────────────────────────────────
# A driver dies for three reasons: it finished, it hit the 1-day walltime, or a
# rule failed (keep-going means it still drains the rest of the DAG first). Only
# the first is a real stopping point, so queue the successor NOW — before
# running Snakemake — with afterany on this job. A walltime SIGKILL skips any
# cleanup we could put at the end of the script, so "submit first" is the only
# placement that survives it.
#
# The chain stops itself on: nothing left to do, no progress since the previous
# link, or MAX_GENERATIONS reached. That last guard matters because a
# permanently-failing rule (e.g. a subject with broken fmriprep derivatives)
# never lets the remaining-job count reach zero.
GENERATION=${SNAKE_DRIVER_GENERATION:-1}
MAX_GENERATIONS=${SNAKE_DRIVER_MAX_GENERATIONS:-8}
PREV_REMAINING=${SNAKE_DRIVER_PREV_REMAINING:-}

# `|| true`: under `set -e` a failing command substitution would kill the driver
# here, before it has chained a successor or run anything.
REMAINING=$(snakemake "${SNAKE_ARGS[@]}" --dry-run --quiet rules 2>/dev/null \
                | awk '$1 == "total" { print $2 }' | tail -1) || true
REMAINING=${REMAINING:-unknown}
echo "[driver] generation $GENERATION/$MAX_GENERATIONS, ${REMAINING} job(s) remaining"

chain_next () {
    if [[ "$REMAINING" == "0" ]]; then
        echo "[driver] DAG complete — not chaining a successor."
        return
    fi
    if [[ -n "$PREV_REMAINING" && "$REMAINING" == "$PREV_REMAINING" ]]; then
        echo "[driver] stalled at ${REMAINING} remaining job(s) since the previous" \
             "generation — not chaining. Inspect the failing rules by hand."
        return
    fi
    if (( GENERATION >= MAX_GENERATIONS )); then
        echo "[driver] generation cap ($MAX_GENERATIONS) reached — not chaining." \
             "Re-run by hand once the blocking rules are fixed."
        return
    fi
    sbatch --dependency="afterany:${SLURM_JOB_ID}" \
           --export="ALL,SNAKE_DRIVER_GENERATION=$((GENERATION + 1)),SNAKE_DRIVER_MAX_GENERATIONS=${MAX_GENERATIONS},SNAKE_DRIVER_PREV_REMAINING=${REMAINING}" \
           abstract_values/snakemake/run_driver.sh \
        || echo "[driver] WARNING: could not queue a successor — the chain ends here."
}
chain_next

# Clear leftover lock from a prior scancel'd / walltime-killed driver
# (idempotent). Without this, the next driver dies in <5s with LockException.
# Safe to run even when no lock exists — and safe in the chain, because
# afterany guarantees the previous link is already dead.
snakemake "${SNAKE_ARGS[@]}" --unlock || true

exec snakemake "${SNAKE_ARGS[@]}" --rerun-incomplete
