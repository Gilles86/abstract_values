"""How distinct is each orientation from its neighbour, per mapping? (no MDS)

For every pair of neighbouring orientations (7.5° apart, plus the seam pair
172.5° / 7.5°), the crossnobis distance between their voxel patterns,
separately for the CDF and inverse-CDF sessions. Each participant's distances
in a region are divided by that participant's mean distance (both sessions),
then averaged; bands ±1 SE across participants.

Reference (thin lines, right axis): how far apart the two neighbours are in
value under each mapping, |Δv|. A code that followed value would make the
neural neighbour distance track |Δv| and differ between mappings; a fixed
orientation code would give the same curve for both mappings.

    python -m abstract_values.geometry.figure_neighbour_distance     # pycortex2 env
    -> notes/figures/fig_neighbour_distance.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.geometry.analyze_crossnobis import load

REPO = Path(__file__).resolve().parents[2]
MAP_COL = {"cdf": "#5B4A9E", "inverse_cdf": "#B23A6E"}
REGIONS = [("V1", ["V1"]), ("IPS0", ["IPS0"]), ("IPS1–2", ["IPS1–2"]), ("IPS3–5", ["IPS3–5"])]


def neighbours(d, rois):
    g = d[d.roi.isin(rois)].copy()
    g["d"] = g.d / g.groupby(["subject", "roi"]).d.transform("mean")
    step = (g.ori_j - g.ori_i).round(1)
    nb = g[(step == 7.5) | ((g.ori_i == 7.5) & (g.ori_j == 172.5))].copy()
    nb["mid"] = np.where((nb.ori_j - nb.ori_i).round(1) == 7.5, (nb.ori_i + nb.ori_j) / 2, 180.0)
    nb["dv"] = (nb.val_i - nb.val_j).abs()
    per = nb.groupby(["subject", "mapping", "mid"]).agg(d=("d", "mean"), dv=("dv", "mean"))
    return per.reset_index()


def main():
    d = load()
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 7, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.6), constrained_layout=True)
    for ax, (name, rois) in zip(axes.ravel(), REGIONS):
        per = neighbours(d, rois)
        ax2 = ax.twinx()
        for m in ("cdf", "inverse_cdf"):
            g = per[per.mapping == m].groupby("mid")
            mu, se = g.d.mean(), g.d.sem()
            x = mu.index.to_numpy()
            ax.fill_between(x, mu - se, mu + se, color=MAP_COL[m], alpha=0.2, lw=0)
            ax.plot(x, mu, "-o", color=MAP_COL[m], ms=2.5, lw=1.4)
            dv = g.dv.mean()
            dv = dv[dv.index < 180]                       # the seam (40 CHF) is off-scale
            ax2.plot(dv.index, dv.values, color=MAP_COL[m], lw=0.8, ls=(0, (2, 1.5)),
                     alpha=0.8)
        ax2.set_ylim(0, 12)
        ax.axvline(180, color="0.8", lw=0.6)
        ax.text(180, ax.get_ylim()[0], "Seam\n(172.5° to 7.5°,\n|Δv| = 40)", fontsize=5,
                color="0.5", ha="right", va="bottom")
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlim(0, 185)
        ax.set_xlabel("Orientation between the two neighbours (°)")
        ax.set_ylabel("Neural distance\n(crossnobis, relative)")
        ax2.set_ylabel("|Δ value| between neighbours (CHF)", color="0.5")
        ax2.tick_params(axis="y", colors="0.5")
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color("0.7")
        ax.set_title(name, loc="left", fontweight="bold")
        r = {}
        for m in ("cdf", "inverse_cdf"):
            q = per[(per.mapping == m) & (per.mid < 180)].groupby("mid")[["d", "dv"]].mean()
            r[m] = np.corrcoef(q.d, q.dv)[0, 1]
        ax.text(0.99, 1.02, f"r(neural, |Δv|), seam excluded: CDF {r['cdf']:+.2f}, inverse {r['inverse_cdf']:+.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color="0.35")
        sns.despine(ax=ax, offset=3, trim=True)
    axes[0, 0].text(0.02, 0.97, "CDF sessions", color=MAP_COL["cdf"], transform=axes[0, 0].transAxes,
                    fontsize=6.5, fontweight="bold", va="top")
    axes[0, 0].text(0.02, 0.87, "Inverse-CDF sessions", color=MAP_COL["inverse_cdf"],
                    transform=axes[0, 0].transAxes, fontsize=6.5, fontweight="bold", va="top")
    axes[0, 0].plot([0.02, 0.09], [0.77, 0.77], transform=axes[0, 0].transAxes, color="0.5",
                    lw=0.8, ls=(0, (2, 1.5)))
    axes[0, 0].text(0.11, 0.77, "|Δ value| (right axis)", transform=axes[0, 0].transAxes,
                    fontsize=6, color="0.5", va="center")
    out = REPO / "notes/figures/fig_neighbour_distance.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
