"""Whole-brain group map of the mix-lambda tuning axis (fixed value vs quantile).

Left: lambda-hat, the argmax of the group-mean cvR^2 profile per fsaverage
vertex (0 = tuned to objective value, 1 = tuned to the value's quantile in its
session = orientation rank). Opacity ramps with the participant t of the
profile's gain over lambda = 0 (t 2 -> 4), so only vertices where the
best-lambda code beats a fixed-value code show. Right: that t map.
Rows: unsmoothed, smoothed. Input: ``surface/sample_lambda_to_surface.py
--collect``.

    python -m abstract_values.visualize.figure_lambda_brain   # pycortex2 env
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from abstract_values.visualize import qc_flatmaps as qf

REPO = Path(__file__).resolve().parents[2]
VAL_COL, ORI_COL = "#C44E52", "#3B5BA5"     # as figure_tuning_connectivity


def main(npz, out, t_lo=2., t_hi=4.):
    d = np.load(npz)
    cmap = LinearSegmentedColormap.from_list(
        "lam", ["#7a1f2b", VAL_COL, "#e8dcd0", ORI_COL, "#1b2c57"])
    fig = plt.figure(figsize=(7.25, 4.6))
    w, h = .47, .39
    for r, key in enumerate(("unsmoothed", "smoothed")):
        t = d[f"{key}_gain_t"]
        alpha = np.clip((t - t_lo) / (t_hi - t_lo), 0, 1)
        y0 = .55 - r * .45
        ax = fig.add_axes([.01, y0, w, h])
        qf.draw_flat(ax, qf.blended(d[f"{key}_group_hat"], alpha, -.5, 1.5, cmap),
                     f"{key.capitalize()}: group λ̂", labels=r == 0)
        ax = fig.add_axes([.51, y0, w, h])
        qf.draw_flat(ax, qf.blended(t, np.clip(t / t_lo, 0, 1), 0, 8, "magma"),
                     f"{key.capitalize()}: gain over fixed value (t, n = "
                     f"{int(np.median(d[f'{key}_n']))})", labels=r == 0)
    qf.colorbar(fig, [.12, .07, .25, .018], cmap, -.5, 1.5, "λ̂",
                ticks=[-.5, 0, .5, 1, 1.5],
                ticklabels=["−0.5", "0\nFixed value", "0.5", "1\nQuantile", "1.5"])
    qf.colorbar(fig, [.62, .07, .25, .018], "magma", 0, 8, "t (participants)")
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
