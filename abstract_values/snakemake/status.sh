#!/usr/bin/env bash
# Human-readable status for jobs in flight on the cluster.
#
# Modes:
#   (default)        all of $USER's jobs
#   --snake          only this project's Snakemake jobs (auto-detected via
#                    most-recent driver log mentioning "abstract_values")
#   --remote         already on cluster — skip the ssh hop
#
# For Snakemake jobs the SLURM executor plugin sets:
#   --job-name <run-uuid>       (per driver invocation; unique)
#   --comment  rule_<rule>_wildcards_<key=value,...>
# We filter on the Name == driver's run UUID.
#
# Usage:
#   bash abstract_values/snakemake/status.sh
#   bash abstract_values/snakemake/status.sh --snake
#   bash abstract_values/snakemake/status.sh --snake --remote

set -eo pipefail

SNAKE_ONLY=0
ON_CLUSTER=0
for arg in "$@"; do
    case "$arg" in
        --snake|--snakemake) SNAKE_ONLY=1 ;;
        --remote)            ON_CLUSTER=1 ;;
    esac
done

# `ssh -T` disables PTY allocation (no "Shared connection closed" stderr noise);
# `tr -d '\r'` strips CR that sometimes sneaks in over ssh and mangles strings.
run() {
    if [[ "$ON_CLUSTER" -eq 1 ]]; then
        bash -c "$1" | tr -d '\r'
    else
        ssh -T sciencecluster "$1" 2>/dev/null | tr -d '\r'
    fi
}

fmt='JobID:14,State:10,TimeUsed:10,Partition:10,Name:38,Comment:90'
raw=$(run "squeue -u \$USER -h -O '$fmt'")

if [[ "$SNAKE_ONLY" -eq 1 ]]; then
    log_file=$(run "ls -t ~/logs/snake_driver_*.log 2>/dev/null | \
                    xargs -r grep -l 'abstract_values' 2>/dev/null | head -1")
    if [[ -z "$log_file" ]]; then
        echo "No abstract_values snakemake driver log found." >&2
        exit 1
    fi

    uuid=$(run "grep -m1 'SLURM run ID:' '$log_file' 2>/dev/null | awk '{print \$NF}'")
    driver_jid=$(basename "$log_file" .log | sed 's/^snake_driver_//')
    driver_state=$(run "squeue -j $driver_jid -h -O 'State:12,TimeUsed:12,NodeList:18' 2>/dev/null | tr -s ' '")

    if [[ -z "$uuid" ]]; then
        echo "Driver: job $driver_jid  (log: $log_file)" >&2
        echo "Could not extract run UUID from log — driver may have failed before printing it." >&2
        exit 1
    fi

    if [[ -n "$driver_state" ]]; then
        echo "Driver: job $driver_jid  ($driver_state)"
    else
        echo "Driver: job $driver_jid  (not in queue — finished or failed)"
    fi
    echo "Run UUID: $uuid"
    echo

    # Filter on the UUID appearing in the Name column (column 5 after splitting)
    raw=$(echo "$raw" | grep -F "$uuid" || true)
    if [[ -z "$raw" ]]; then
        echo "No child jobs in the queue right now. Driver's recent activity:"
        echo "---"
        run "tail -15 '$log_file'"
        exit 0
    fi
fi

echo "=== counts ==="
echo "$raw" | awk '{print $2}' | sort | uniq -c | sort -rn
echo

echo "$raw" | awk '
    BEGIN { printf "%-12s %-9s %-9s %-9s %-22s %s\n",
                   "JOBID","STATE","TIME","PART","NAME","DETAIL" }
    {
        jobid=$1; state=$2; t=$3; part=$4;
        name = $5
        detail = ""
        for (i=6; i<=NF; i++) detail = (detail ? detail " " : "") $i
        if (detail == "") detail = "(no comment)"
        if (length(name) > 22) name = substr(name, 1, 19) "…"
        printf "%-12s %-9s %-9s %-9s %-22s %s\n", jobid, state, t, part, name, detail
    }'
