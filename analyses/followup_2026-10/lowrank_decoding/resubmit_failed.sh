#!/bin/bash
# Resubmit failed/timed-out decode_lowrank_cv array tasks one by one (cluster side).
#   bash resubmit_failed.sh <jobid> [<jobid> ...]
SUBJECTS=(pil01 pil02 $(seq -f "%02g" 3 36))
cd $HOME/git/abstract_values/abstract_values/encoding_models/slurm_jobs
for j in "$@"; do
    name=$(sacct -j $j -X -n --format=JobName%30 | head -1 | tr -d ' ')
    q=$([[ $name == dlr_ori_* ]] && echo orientation || echo value)
    roi=${name#dlr_*_}
    for t in $(sacct -j $j -X -n --format=JobID%20,State | awk '$2 ~ /FAILED|TIMEOUT|OUT_OF_ME|NODE_FAIL/ {split($1,a,"_"); print a[2]}'); do
        s=${SUBJECTS[$((t - 1))]}
        echo "$name task $t -> sub-$s: $(sbatch --parsable --partition=lowprio --job-name=$name --export=ALL,PARTICIPANT_LABEL=$s,QUANTITY=$q,ROI=$roi decode_lowrank_cv.sh)"
    done
done
