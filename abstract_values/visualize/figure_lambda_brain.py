"""Whole-brain group map of the mix-lambda tuning axis (fixed value vs quantile).

Left: lambda-hat, the argmax of the group-mean cvR^2 profile per fsaverage
vertex (0 = tuned to objective value, 1 = tuned to the value's quantile in its
session = orientation rank). Right: participant t of cvR^2(lambda = 1) -
cvR^2(lambda = 0), blue = quantile wins. Opacity, both maps: where value is
coded, the group-mean aPRF cvR^2 over the null model (as in the gradient
figure), ramped from 0.2% to 1%. (The lambda model's own cvR^2 is a poor gate:
picking 1 of 410 tuning shapes per voxel without a null leaves it negative in
most of cortex.) Rows: unsmoothed,
smoothed. Input: ``surface/sample_lambda_to_surface.py --collect``.

    python -m abstract_values.visualize.figure_lambda_brain   # pycortex2 env
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize import qc_flatmaps as qf

REPO = Path(__file__).resolve().parents[2]
VAL_COL, ORI_COL = "#C44E52", "#3B5BA5"     # as figure_tuning_connectivity


def value_gate(smoothed, cache):
    """Group-mean aPRF cvR^2 over the null per fsaverage vertex (cached)."""
    cache = Path(cache)
    key = "smoothed" if smoothed else "unsmoothed"
    if cache.exists() and key in np.load(cache):
        return np.load(cache)[key]
    from abstract_values.visualize.group_surface_maps import discover_subjects
    from abstract_values.visualize.qc_tuning_maps import collect
    deriv = Path(BIDS_FOLDER) / "derivatives"
    g = np.nanmean(np.stack(collect(deriv, discover_subjects(deriv), smoothed)["val_gain"]), 0)
    old = dict(np.load(cache)) if cache.exists() else {}
    np.savez_compressed(cache, **old, **{key: g.astype(np.float32)})
    return g


def main(npz, out, lo=.002, hi=.012):
    d = np.load(npz)
    cache = Path(npz).with_name("value_gain_fsaverage.npz")
    cmap = LinearSegmentedColormap.from_list(
        "lam", ["#7a1f2b", VAL_COL, "#e8dcd0", ORI_COL, "#1b2c57"])
    fig = plt.figure(figsize=(7.25, 4.6))
    w, h = .47, .39
    for r, key in enumerate(("unsmoothed", "smoothed")):
        gate = value_gate(key == "smoothed", cache)
        alpha = np.clip((gate - lo) / (hi - lo), 0, 1)
        n = int(np.median(d[f"{key}_n"]))
        y0 = .55 - r * .45
        ax = fig.add_axes([.01, y0, w, h])
        qf.draw_flat(ax, qf.blended(d[f"{key}_group_hat"], alpha, -.5, 1.5, cmap),
                     f"{key.capitalize()}: group λ̂", labels=r == 0)
        ax = fig.add_axes([.51, y0, w, h])
        qf.draw_flat(ax, qf.blended(d[f"{key}_quantile_vs_value_t"], alpha, -4, 4, cmap),
                     f"{key.capitalize()}: quantile vs fixed value (t, n = {n})",
                     labels=r == 0)
    qf.colorbar(fig, [.12, .07, .25, .018], cmap, -.5, 1.5, "λ̂",
                ticks=[-.5, 0, .5, 1, 1.5],
                ticklabels=["−0.5", "0\nFixed value", "0.5", "1\nQuantile", "1.5"])
    qf.colorbar(fig, [.62, .07, .25, .018], cmap, -4, 4, "t (participants)",
                ticks=[-4, -2, 0, 2, 4],
                ticklabels=["−4\nValue", "−2", "0", "2", "4\nQuantile"])
    fig.savefig(out, dpi=300)
    fig.savefig(Path(out).with_suffix(".png"), dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=str(REPO / "notes/data/lambda_brain_fsaverage.npz"))
    ap.add_argument("--out", default=str(REPO / "notes/figures/fig_lambda_brain.pdf"))
    a = ap.parse_args()
    main(a.npz, a.out)
