#!/usr/bin/env bash
# Human-readable status for jobs in flight on the cluster.
#
# Works for both:
#   - Snakemake-submitted jobs: name is a UUID, comment carries
#     "rule_<rule_name>_wildcards_<key=value,...>"
#   - Traditional sbatch jobs (ingest_new_session.sh chain): name carries
#     the script identity (fmriprep_abstractvalue, fit_glmsingle, fit_aprf, etc.)
#
# Usage:
#   bash abstract_values/snakemake/status.sh            # local: ssh into cluster
#   bash abstract_values/snakemake/status.sh --remote   # already on cluster

set -eo pipefail

ON_CLUSTER=0
[[ "${1:-}" == "--remote" ]] && ON_CLUSTER=1

run() {
    if [[ "$ON_CLUSTER" -eq 1 ]]; then
        bash -c "$1"
    else
        ssh sciencecluster "$1"
    fi
}

# 1) Compact overview: counts per state
run 'squeue -u $USER -h -O "State:12" | sort | uniq -c | sort -rn'
echo

# 2) Per-job table with meaningful identity field (comment for Snakemake, name otherwise)
fmt='JobID:14,State:10,TimeUsed:10,Partition:10,Name:25,Comment:80'
run "squeue -u \$USER -h -O '$fmt'" | \
    awk 'BEGIN { printf "%-13s %-9s %-9s %-9s %-24s %s\n",
                       "JOBID","STATE","TIME","PART","NAME","DETAIL" }
         { jobid=$1; state=$2; t=$3; part=$4; name=$5;
           # build "detail" — comment if present, else name
           detail=""
           for (i=6; i<=NF; i++) detail = (detail ? detail " " : "") $i
           if (detail == "") detail = name
           printf "%-13s %-9s %-9s %-9s %-24s %s\n", jobid, state, t, part, name, detail
         }'
