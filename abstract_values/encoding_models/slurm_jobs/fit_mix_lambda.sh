#!/bin/bash
#SBATCH --job-name=mixlam
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/mixlam_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00

# Shared-lambda mix-axis aPRF, cross-validated (fit_mix_lambda.py).
# Array: 7 conditions x N participants; task t -> participant (t-1) % N,
# condition (t-1) / N: observed; simulated lambda {0, 1} x noise {1, 2, 3}.
#   sbatch --array=1-245 --exclude=u24-chaihm0-651 fit_mix_lambda.sh
# MODEL=width: separate position / width mixing (lam_mu, lam_sigma).
set -eo pipefail
export PYTHONUNBUFFERED=1
export KERAS_BACKEND=tensorflow
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue
cd "$HOME/git/abstract_values"
mapfile -t SUBS < <(ls "$BIDS_FOLDER/derivatives/encoding_models/aprf-session-shift" \
                    | grep '^sub-' | sed 's/^sub-//' | sort)
N=${#SUBS[@]}
t=$(( SLURM_ARRAY_TASK_ID - 1 ))
ROIS=(V1 IPS0 IPS12 IPS345 V3ab hV4)
if [ "${PER_ROI:-0}" = "1" ]; then   # one task per ROI: index = (cond * N + subject) * 6 + roi
    ROI=${ROIS[$(( t % 6 ))]}
    t=$(( t / 6 ))
fi
SUB=${SUBS[$(( t % N ))]}
c=$(( t / N ))
ARGS=("$SUB" --bids-folder "$BIDS_FOLDER")
[ -n "${ROI:-}" ] && ARGS+=(--targets "$ROI")
[ -n "${MODEL:-}" ] && ARGS+=(--model "$MODEL")
if [ "${MODEL:-mix}" = "width" ]; then
    # observed; corners (lam_mu, lam_sigma) (0,0) (1,0) (0,1) (1,1) x noise {1, 2}:
    #   sbatch --array=1-315 --export=ALL,MODEL=width ...
    #   per ROI (6x the tasks, ~30-60 min each):
    #   sbatch -p lowprio --array=1-1890 --export=ALL,MODEL=width,PER_ROI=1 ...
    if [ $c -gt 0 ]; then
        LM=(x 0 1 0 1 0 1 0 1); LS=(x 0 0 1 1 0 0 1 1); NOISE=(x 1 1 1 1 2 2 2 2)
        ARGS+=(--sim-corner "${LM[$c]}" "${LS[$c]}" --noise-scale "${NOISE[$c]}")
    fi
elif [ $c -gt 0 ]; then
    LAMS=(x 0 0 0 1 1 1); NOISE=(x 1 2 3 1 2 3)
    ARGS+=(--sim-lam "${LAMS[$c]}" --noise-scale "${NOISE[$c]}")
fi
exec $HOME/data/conda/envs/abstract_values/bin/python -u -W ignore -m \
    abstract_values.encoding_models.fit_mix_lambda "${ARGS[@]}"
