#!/bin/bash
# Regenerate every model-comparison figure after an encoding-model refit.
#
# Written for the alpha=0 -> alpha=10 ridge refit (see plot_v1_sweep.py), but
# it applies after any refit that changes cvR2: the comparison figures are all
# downstream of the same fsaverage surfaces and volumetric cvR2 maps, and
# regenerating them one at a time by hand is how they drift out of sync with
# each other.
#
# Pulls from the cluster first, then plots locally -- the split the project
# uses everywhere: aggregation on the cluster, "read summary + matplotlib"
# locally, where iteration is fast and the PDF opens immediately.
#
# Usage:  bash abstract_values/visualize/regenerate_model_comparison.sh [--no-sync]
set -euo pipefail

CLUSTER=sciencecluster:/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives
LOCAL=/data/ds-abstractvalue/derivatives
PY=$HOME/mambaforge/envs/pycortex2/bin/python
FIG=notes/figures

cd "$(dirname "$0")/../.."

if [ "${1:-}" != "--no-sync" ]; then
    echo "=== syncing encoding_models from the cluster ==="
    rsync -a --stats "${CLUSTER}/encoding_models/" "${LOCAL}/encoding_models/" \
        | grep -E "Number of files transferred|Total transferred file size" || true
fi

mkdir -p "$FIG"

echo "=== 1/5 factorial: architecture x space x flexibility ==="
$PY -u -m abstract_values.visualize.factorial_model_comparison \
    --out "$FIG/factorial_model_comparison.pdf" \
    --tsv notes/data/factorial_model_comparison.tsv

echo "=== 2/5 per-vertex model winner (all four cv models) ==="
$PY -u -m abstract_values.visualize.model_winner_maps \
    --summary "$FIG/model_winner_summary.pdf"

echo "=== 3/5 per-vertex model winner (value models only) ==="
$PY -u -m abstract_values.visualize.model_winner_maps \
    --models aprf.cv aprf-shift.cv aprf-fully-shifted.cv aprf-linear.cv \
    --summary "$FIG/model_winner_summary_value_only.pdf"

echo "=== 4/5 group surface contact sheet ==="
$PY -u -m abstract_values.visualize.group_surface_maps \
    --contact-sheet "$FIG/group_aprf_vs_null.pdf"

echo "=== 5/5 ROI-level nested-model ladder ==="
$PY -u -m abstract_values.visualize.cvr2_model_comparison \
    --out "$FIG/cvr2_model_comparison.pdf"

echo
echo "Wrote:"
ls -lt "$FIG"/*.pdf | head -8
