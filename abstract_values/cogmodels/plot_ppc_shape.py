"""Plot ppc_shape.py output: does each fit get the SHAPE of the bid distribution?

  a  pooled distribution of bids around their cell median (log density), per
     mapping
  b  share of bids more than 3 CHF off their cell median, per orientation
  c  bid SD around the cell median, per orientation

Observed data is a black line; each fit is its posterior-predictive median as a
coloured line. The 95% predictive interval is not drawn (several overlapping
bands turn into one brown smear); its coverage of the observed curve is printed
next to each fit in the key.

    python -m abstract_values.cogmodels.plot_ppc_shape --in-dir notes/data/ppc_shape_base \\
        [--fits "Free prior + free lapse" "+ spline offsets"]
    -> notes/figures/fig_ppc_shape_<in-dir name>.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.visualize.plot_vmpfc_dropout import style

REPO = Path(__file__).resolve().parents[2]
MAP_LAB = {"cdf": "CDF sessions", "inverse_cdf": "Inverse-CDF sessions"}
FIT_COLS = ["0.6", "#C2560A", "#3B5BA5", "#2E7D6B", "#B8860B"]
OBS_COL = "0.1"
STATS = (("p3", "Bids > 3 CHF off (%)", 100), ("sd", "Bid SD around median (CHF)", 1))


def summarise(df, fits, key, stat):
    """{fit: DataFrame(index=key, columns=[0.025, 0.5, 0.975])} over draws."""
    return {f: df[df.fit == f].groupby(key)[stat].quantile([0.025, 0.5, 0.975]).unstack()
            for f in fits}


def inside(ob, q):
    return 100 * ((ob >= q[0.025]) & (ob <= q[0.975])).mean()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", required=True)
    p.add_argument("--fits", nargs="+", default=None)
    a = p.parse_args()
    ind = Path(a.in_dir)
    H = pd.read_csv(ind / "ppc_shape_hist.tsv", sep="\t")
    C = pd.read_csv(ind / "ppc_shape_cells.tsv", sep="\t")
    fits = a.fits or [f for f in H.fit.unique() if f != "Observed"]
    cols = dict(zip(fits, FIT_COLS))
    style()

    fig = plt.figure(figsize=(7.25, 6.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.45, wspace=0.18,
                          left=0.09, right=0.985, top=0.93, bottom=0.07)
    cov = {f: {} for f in fits}
    for j, mp in enumerate(("cdf", "inverse_cdf")):
        # a: residual distribution
        ax = fig.add_subplot(gs[0, j])
        h = H[H.mapping == mp]
        ob = h[h.fit == "Observed"].set_index("bin").density
        for f, q in summarise(h, fits, "bin", "density").items():
            ax.step(q.index, np.where(q[0.5] > 0, q[0.5], np.nan), where="mid",
                    color=cols[f], lw=1.0, zorder=2)
            cov[f][f"hist_{mp}"] = inside(ob, q)
        ax.step(ob.index, np.where(ob > 0, ob, np.nan), where="mid", color=OBS_COL,
                lw=1.3, zorder=3)
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1)
        ax.set_xlim(-12, 12); ax.set_xticks([-12, -6, -3, 0, 3, 6, 12])
        ax.set_xlabel("Bid − cell median bid (CHF)")
        ax.set_title(MAP_LAB[mp], loc="left", fontweight="bold", fontsize=7.5)
        if j == 0:
            ax.set_ylabel("Share of bids per 0.5 CHF")
            ax.text(-0.15, 1.08, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
        sns.despine(ax=ax, offset=3, trim=True)

        # b, c: per orientation
        c = C[C.mapping == mp]
        for i, (stat, lab, sc) in enumerate(STATS):
            ax = fig.add_subplot(gs[1 + i, j])
            ob = c[c.fit == "Observed"].set_index("orientation")[stat]
            for f, q in summarise(c, fits, "orientation", stat).items():
                ax.plot(q.index, q[0.5] * sc, color=cols[f], lw=1.0, zorder=2)
                cov[f][f"{stat}_{mp}"] = inside(ob, q)
            ax.plot(ob.index, ob.values * sc, color=OBS_COL, lw=1.3, zorder=3)
            ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_ylim(0, None)
            if i == 1:
                ax.set_xlabel("Orientation (°)")
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(lab)
                ax.text(-0.15, 1.08, "bc"[i], transform=ax.transAxes, fontsize=9,
                        fontweight="bold")
            sns.despine(ax=ax, offset=3)

    # One key for the whole figure, in the top margin: line glyphs + coverage
    entries = [("Observed", OBS_COL, 1.3)] + [
        (f"{f}  ({np.mean(list(cov[f].values())):.0f}%)", cols[f], 1.0)
        for f in fits]
    x = 0.09
    for lab, col, lw in entries:
        fig.add_artist(plt.Line2D([x, x + 0.025], [0.985, 0.985], color=col, lw=lw,
                                  solid_capstyle="butt"))
        t = fig.text(x + 0.032, 0.985, lab, fontsize=6.3, va="center",
                     color="0.45" if col == "0.6" else col)
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(fig.transFigure.inverted())
        x = bb.x1 + 0.03
    out = REPO / f"notes/figures/fig_ppc_shape_{ind.name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(pd.DataFrame(cov).T.round(0).to_string())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
