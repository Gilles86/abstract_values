#!/bin/bash
#SBATCH --job-name=dropout_inventory
#SBATCH --output=/home/gdehol/logs/dropout_inventory_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --account=zne.uzh

# File inventory of every dataset (dropout_survey/datasets.py) and one subject
# list per dataset with usable inputs.
#   sbatch inventory.sh                       # all datasets
#   sbatch --export=DATASETS="risk3t risk7t" inventory.sh
OUT=/scratch/gdehol/dropout_survey
export TMPDIR=/scratch/gdehol/tmp
mkdir -p $TMPDIR $OUT
. $HOME/init_conda.sh
conda activate abstract_values
cd $HOME/git/abstract_values
TAG=$(echo ${DATASETS:-all} | tr ' ' '_')
PYTHONUNBUFFERED=1 python -u -m abstract_values.dropout_survey.datasets \
    ${DATASETS:+--datasets $DATASETS} --out $OUT/inventory_${TAG}.tsv || exit 1
python - <<PY
import pandas as pd
df = pd.read_csv("$OUT/inventory_${TAG}.tsv", sep="\t", dtype={"subject": str})
ok = df[(df.n_runs > 0) & df.aparcaseg.notna() & df.t1w]
for ds, g in ok.groupby("dataset"):
    open(f"$OUT/subjects_{ds}.txt", "w").write("\n".join(g.subject) + "\n")
    print(ds, len(g))
PY
