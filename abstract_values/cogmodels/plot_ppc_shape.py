"""Plot ppc_shape.py output: does each fit get the SHAPE of the bid distribution?

  a  pooled distribution of bids around their cell median (log density), per
     mapping: observed points over each fit's 95% predictive band
  b  share of bids more than 3 CHF off their cell median, per orientation
  c  bid SD around the cell median, per orientation

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
MAP_COL = {"cdf": "#5B4A9E", "inverse_cdf": "#B23A6E"}
MAP_LAB = {"cdf": "CDF sessions", "inverse_cdf": "Inverse-CDF sessions"}
FIT_COLS = ["0.55", "#C2560A", "#2E7D6B", "#3B5BA5", "#B8860B"]


def bands(df, fits, key, stat):
    out = {}
    for f in fits:
        s = df[df.fit == f].groupby(key)[stat]
        out[f] = s.quantile([0.025, 0.975]).unstack()
    return out


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

    fig = plt.figure(figsize=(7.25, 7.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, 1], hspace=0.42, wspace=0.16,
                          left=0.09, right=0.985, top=0.93, bottom=0.07)
    cov = {f: {} for f in fits}
    for j, mp in enumerate(("cdf", "inverse_cdf")):
        # a: residual distribution
        ax = fig.add_subplot(gs[0, j])
        h = H[H.mapping == mp]
        ob = h[h.fit == "Observed"].set_index("bin").density
        for f, q in bands(h, fits, "bin", "density").items():
            ax.fill_between(q.index, np.maximum(q[0.025], 1e-5), np.maximum(q[0.975], 1e-5),
                            color=cols[f], alpha=0.3, lw=0, step="mid")
            inside = (ob >= q[0.025]) & (ob <= q[0.975])
            cov[f][f"hist_{mp}"] = 100 * inside.mean()
        ax.plot(ob.index, np.where(ob > 0, ob, np.nan), "o", ms=2.4, color=MAP_COL[mp],
                mec="white", mew=0.3, zorder=3)
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1)
        ax.set_xlim(-12, 12); ax.set_xticks([-12, -6, -3, 0, 3, 6, 12])
        ax.set_xlabel("Bid − cell median bid (CHF)")
        ax.set_title(MAP_LAB[mp], loc="left", color=MAP_COL[mp], fontweight="bold",
                     fontsize=7.5)
        if j == 0:
            ax.set_ylabel("Share of bids per 0.5 CHF")
            ax.text(-0.15, 1.08, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
            tf = ax.transAxes
            for i, f in enumerate(fits):
                yy = 0.95 - i * 0.075
                ax.add_patch(plt.Rectangle((0.03, yy - 0.025), 0.07, 0.05, transform=tf,
                                           color=cols[f], alpha=0.45, lw=0))
                ax.text(0.12, yy, f, transform=tf, fontsize=6.3, va="center", color="0.25")
            yy = 0.95 - len(fits) * 0.075
            ax.plot(0.065, yy, "o", ms=2.6, transform=tf, color="0.3", mec="white", mew=0.3)
            ax.text(0.12, yy, "Observed", transform=tf, fontsize=6.3, va="center",
                    color="0.25")
        sns.despine(ax=ax, offset=3)

        # b, c: per orientation
        c = C[C.mapping == mp]
        for i, (stat, lab, lim) in enumerate((("p3", "Bids > 3 CHF off (%)", 42),
                                              ("sd", "Bid SD around median (CHF)", None))):
            ax = fig.add_subplot(gs[1 + i, j])
            sc = 100 if stat == "p3" else 1
            ob = c[c.fit == "Observed"].set_index("orientation")[stat] * sc
            for f, q in bands(c, fits, "orientation", stat).items():
                ax.fill_between(q.index, q[0.025] * sc, q[0.975] * sc, color=cols[f],
                                alpha=0.3, lw=0)
                cov[f][f"{stat}_{mp}"] = 100 * ((ob >= q[0.025] * sc)
                                                & (ob <= q[0.975] * sc)).mean()
            ax.plot(ob.index, ob.values, "o", ms=3, color=MAP_COL[mp], mec="white",
                    mew=0.4, zorder=3)
            ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_ylim(0, lim)
            if i == 1:
                ax.set_xlabel("Orientation (°)")
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(lab)
                ax.text(-0.15, 1.08, "bc"[i], transform=ax.transAxes, fontsize=9,
                        fontweight="bold")
            sns.despine(ax=ax, offset=3)
    out = REPO / f"notes/figures/fig_ppc_shape_{ind.name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(pd.DataFrame(cov).T.round(0).to_string())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
