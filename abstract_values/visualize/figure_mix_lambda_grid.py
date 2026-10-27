"""Estimating (lam_mu, lam_sigma) per participant and ROI from a cvR^2 surface.

a: group cvR^2 surface over the 0.1-step (lam_mu, lam_sigma) grid, relative to the value
   code (0, 0), per ROI (mean over participants); cross = group peak.
b: each participant's surface peak (jittered); dot = median.
c: held-out cvR^2 of the gradient-descent refinement (started per fold from the best
   training grid point) relative to the fixed quantile code (1, 1).

Input: notes/data/mix_lambda_grid.tsv (fit_mix_lambda.py --grid, per ROI, collected).

    python -m abstract_values.visualize.figure_mix_lambda_grid
    -> notes/figures/fig_mix_lambda_grid.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from abstract_values.visualize import qc_value_shift as q   # noqa: F401 house rcParams
from abstract_values.visualize.figure_mix_lambda import CORNERS, NAME

REPO = Path(__file__).resolve().parents[2]
ROIS = ["V1", "hV4", "V3ab", "IPS0", "IPS12", "IPS345"]


def load():
    d = pd.read_csv(REPO / "notes/data/mix_lambda_grid.tsv", sep="\t", dtype={"subject": str})
    d["lam_mu"], d["lam_sigma"] = d.lam_mu.round(2), d.lam_sigma.round(2)
    corner = lambda a, b: d[(d.lam_mu == a) & (d.lam_sigma == b)].set_index(  # noqa: E731
        ["subject", "roi"]).cvr2
    s = d.drop_duplicates(["subject", "roi"]).set_index(["subject", "roi"])
    s["c00"], s["c11"] = corner(0, 0), corner(1, 1)
    s = s.reset_index()
    d = d.merge(s[["subject", "roi", "c00"]], on=["subject", "roi"])
    d["rel"] = 1e3 * (d.cvr2 - d.c00)
    return d, s


def main(out=REPO / "notes/figures/fig_mix_lambda_grid.pdf"):
    d, s = load()
    g = d.groupby(["roi", "lam_mu", "lam_sigma"]).rel.mean()
    grid = np.sort(d.lam_mu.unique())
    vmax = max(abs(g.min()), g.max())
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(7.25, 4.4))
    top = fig.add_gridspec(1, 6, left=.07, right=.9, top=.9, bottom=.66, wspace=.12)
    for i, r in enumerate(ROIS):
        ax = fig.add_subplot(top[i])
        M = g.loc[r].unstack().reindex(index=grid, columns=grid).to_numpy()  # rows lam_mu
        im = ax.imshow(M.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[-.05, 1.05, -.05, 1.05])
        k = np.unravel_index(np.nanargmax(M), M.shape)
        ax.plot(grid[k[0]], grid[k[1]], "x", color="k", ms=5, mew=1.2)
        ax.set_title(NAME[r], fontsize=7)
        ax.set_xticks([0, .5, 1])
        ax.set_yticks([0, .5, 1] if i == 0 else [])
        if i == 0:
            ax.set_ylabel("λσ (width follows quantile)")
    fig.text(.485, .59, "λμ (position follows quantile)", ha="center", fontsize=8)
    cax = fig.add_axes([.915, .69, .01, .18])
    fig.colorbar(im, cax=cax).set_label("cvR² vs value code (×10⁻³)", fontsize=6.5)
    fig.text(.01, .95, "a  Group cvR² surface: the peak is the quantile code (1, 1) in every ROI",
             fontweight="bold", fontsize=8)

    bot = fig.add_gridspec(1, 2, left=.08, right=.99, top=.44, bottom=.1, wspace=.3,
                           width_ratios=[1.5, 1])
    ax = fig.add_subplot(bot[0])
    for i, r in enumerate(ROIS):
        x = s[s.roi == r]
        for j, (col, lab) in enumerate((("peak_lam_mu", "λμ"), ("peak_lam_sigma", "λσ"))):
            c = CORNERS["1_0"][1] if j == 0 else "0.5"
            xp = i + (j - .5) * .32
            ax.scatter(xp + rng.uniform(-.09, .09, len(x)),
                       x[col] + rng.uniform(-.025, .025, len(x)), s=3, color=c, alpha=.45, lw=0)
            ax.plot([xp - .13, xp + .13], [x[col].median()] * 2, color=c, lw=2,
                    solid_capstyle="butt")
    ax.text(.01, 1.03, "Position, λμ", color=CORNERS["1_0"][1], fontsize=6.5,
            transform=ax.transAxes)
    ax.text(.2, 1.03, "Width, λσ", color="0.45", fontsize=6.5, transform=ax.transAxes)
    ax.set_xticks(range(len(ROIS)), [NAME[r] for r in ROIS])
    ax.set_yticks([0, .5, 1])
    ax.set_ylabel("Participant's surface peak")
    ax.set_title("b  Individual peaks scatter over the grid", loc="left", fontweight="bold",
                 pad=12)

    ax = fig.add_subplot(bot[1])
    for i, r in enumerate(ROIS):
        x = 1e3 * (s[s.roi == r].cvr2_refined - s[s.roi == r].c11).to_numpy()
        m, ci = x.mean(), stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        ax.scatter(rng.uniform(-.1, .1, len(x)) + i, x, s=3, color="0.6", alpha=.5, lw=0)
        ax.plot([i, i], [m - ci, m + ci], color="k", lw=1.3)
        ax.plot(i, m, "o", color="k", ms=3.5, mec="white", mew=.5)
    ax.axhline(0, color=CORNERS["1_1"][1], lw=.8)
    ax.set_xticks(range(len(ROIS)), [NAME[r] for r in ROIS], rotation=45, ha="right")
    ax.set_ylabel("Held-out cvR², refined λs\nminus fixed quantile code (×10⁻³)")
    ax.set_title("c  Freeing the λs overfits", loc="left", fontweight="bold", pad=12)

    for a in fig.axes[6:]:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    out = Path(out)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
