#!/usr/bin/env bash
# Human-readable status for jobs in flight on the cluster.
#
# Modes:
#   (default)        all of $USER's jobs
#   --snake          only this project's Snakemake jobs (auto-detected via
#                    most-recent driver log containing "abstract_values")
#   --remote         already on cluster — skip the ssh hop
#
# How --snake filters: the SLURM executor plugin tags every Snakemake-
# submitted job with `--job-name <run-uuid>`, where the UUID is unique per
# driver invocation. We pick the most recent `snake_driver_*.log` that
# mentions our project, read its `SLURM run ID:`, and filter squeue's NAME
# column to that UUID. So --snake here means "this project's current DAG".
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

run() {
    if [[ "$ON_CLUSTER" -eq 1 ]]; then bash -c "$1"; else ssh sciencecluster "$1"; fi
}

fmt='JobID:14,State:10,TimeUsed:10,Partition:10,Name:38,Comment:90'
raw=$(run "squeue -u \$USER -h -O '$fmt'")

if [[ "$SNAKE_ONLY" -eq 1 ]]; then
    # Find this project's current driver's log file + extract run UUID
    log_file=$(run "ls -t ~/logs/snake_driver_*.log 2>/dev/null | \
                    xargs grep -l 'abstract_values' 2>/dev/null | head -1")
    if [[ -z "$log_file" ]]; then
        echo "No abstract_values snakemake driver log found." >&2
        exit 1
    fi
    uuid=$(run "grep -m1 'SLURM run ID:' $log_file 2>/dev/null | awk '{print \$NF}'")
    driver_jid=$(basename "$log_file" .log | sed 's/^snake_driver_//')

    # Driver status (state, time-used) — empty if driver has exited
    driver_status=$(run "squeue -j $driver_jid -h -O 'State:12,TimeUsed:10,NodeList:18' 2>/dev/null" \
                    | sed 's/[[:space:]]\+/ /g')
    if [[ -n "$driver_status" ]]; then
        echo "Driver: job $driver_jid  (RUNNING — $driver_status)"
    else
        echo "Driver: job $driver_jid  (not in queue — finished or failed)"
    fi
    echo "Run UUID: $uuid"
    echo

    raw=$(echo "$raw" | grep -F "$uuid" || true)
    if [[ -z "$raw" ]]; then
        echo "No child jobs in the queue right now. Driver's recent activity:"
        echo "---"
        run "tail -15 $log_file"
        exit 0
    fi
fi

echo "=== counts ==="
echo "$raw" | awk '{print $2}' | sort | uniq -c | sort -rn
echo

# Per-job table: surface Comment (rule + wildcards) since the Name is a UUID
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
