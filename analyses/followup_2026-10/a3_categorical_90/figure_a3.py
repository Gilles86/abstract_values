"""A3 slide-sized summary figure (SVG + PNG + PDF), from the tables written by
behavior_90.py and analyze_a3.py.

  a  Behaviour: bids that cross the centre value, per centre orientation and mapping
     (at 90 deg = category errors around 22 CHF)
  b  Distance of each orientation to all others (crossnobis, relative to the ROI mean):
     90 deg stands out in V1 and twice as much in IPS
  c  Mapping test: d(90, neighbours) / d(other adjacent pairs) per ROI and mapping,
     against what a value code (d ~ dv^2) predicts
  d  Per participant: IPS global-distinctiveness index against V1's, with reliabilities

    python figure_a3.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT = Path("/data/ds-abstractvalue/derivatives/followup_2026-10/a3_categorical_90")
MAP_COL = {"cdf": "#5B4A9E", "inverse_cdf": "#B23A6E"}
MAP_LAB = {"cdf": "Misaligned", "inverse_cdf": "Aligned"}
ROI_COL = {"V1": "#1B8A80", "IPS": "#D9730D"}
ROIS = ["V1", "V2V3", "IPS0", "IPS12", "IPS345", "IPS", "NPCr"]
ROI_LAB = {"V1": "V1", "V2V3": "V2/V3", "IPS0": "IPS0", "IPS12": "IPS1–2", "IPS345": "IPS3–5",
           "IPS": "IPS", "NPCr": "NPC R"}

# Talk-sized: 15 x 4.4 in, for a full-width 16:9 slide -> 11-13 pt type
mpl.rcParams.update({
    "font.family": "Helvetica", "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial"],
    "font.size": 11.5, "axes.labelsize": 12.5, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 4,
    "ytick.major.size": 4, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "lines.linewidth": 1.8, "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def letter(ax, s):
    ax.text(-0.17, 1.04, s, transform=ax.transAxes, fontsize=14, fontweight="bold",
            va="bottom", ha="right")


def main():
    cr = pd.read_csv(OUT / "behavior_crossing.tsv", sep="\t")
    G = pd.read_csv(OUT / "index_group.tsv", sep="\t")
    W = pd.read_csv(OUT / "subject_table.tsv", sep="\t", index_col=0)
    R = pd.read_csv(OUT / "reliability.tsv", sep="\t")
    RG = pd.read_csv(OUT / "ips_vs_v1.tsv", sep="\t")
    row = pd.read_csv(OUT / "profile_rowmean.tsv", sep="\t")

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.2, 1.2, 1.25, 0.95]))

    # a: behaviour
    ax = axes[0]
    for m, g in cr.groupby("mapping"):
        ax.fill_between(g.centre, g.ci_lo * 100, g.ci_hi * 100, color=MAP_COL[m], alpha=.2, lw=0)
        ax.plot(g.centre, g.rate * 100, color=MAP_COL[m])
    ax.axvline(90, color="0.75", lw=0.8, ls="--", zorder=0)
    ax.text(12, 35, MAP_LAB["inverse_cdf"], color=MAP_COL["inverse_cdf"])
    ax.text(12, 31.5, MAP_LAB["cdf"], color=MAP_COL["cdf"])
    v = cr[cr.centre == 90].set_index("mapping").rate * 100
    ax.annotate(f"Category errors\n{v['cdf']:.1f}% and {v['inverse_cdf']:.1f}%",
                xy=(90, v.max() + 1.5), xytext=(104, 32), ha="left", va="center", fontsize=11,
                arrowprops=dict(arrowstyle="-|>", color="0.2", lw=1.1, mutation_scale=10, shrinkA=3,
                                shrinkB=5, relpos=(0, 0.5),
                                connectionstyle="angle3,angleA=180,angleB=70"))
    ax.set_xticks([15, 45, 90, 135, 165])
    ax.set_ylim(0, 38)
    ax.set_xlabel("Centre orientation (°)")
    ax.set_ylabel("Neighbour bids crossing\nthe centre value (%)")
    letter(ax, "a")

    # b: distance of each orientation to all others
    ax = axes[1]
    for roi in ["V1", "IPS"]:
        g = row[(row.roi == roi) & (row.kind == "crossnobis")]
        ax.plot(g.ori_i, g.d_rel, color=ROI_COL[roi], marker="o", ms=3.5)
        y90 = g[g.ori_i == 90].d_rel.iloc[0]
        ax.text(99, y90, f"{roi}  {y90:.1f}×", color=ROI_COL[roi], va="center")
    ax.axhline(1, color="0.75", lw=0.8, ls="--", zorder=0)
    ax.set_xticks([7.5, 45, 90, 135, 172.5], ["7.5", "45", "90", "135", "172.5"])
    ax.set_yticks([0.5, 1, 1.5, 2])
    ax.set_xlabel("Orientation (°)")
    ax.set_ylabel("Mean distance to all others\n(relative to ROI mean)")
    letter(ax, "b")

    # c: mapping test
    ax = axes[2]
    g = G[(G.kind == "crossnobis") & G.mapping.isin(["cdf", "inverse_cdf"])]
    x0 = np.arange(len(ROIS))
    for k, m in enumerate(["cdf", "inverse_cdf"]):
        gm = g[g.mapping == m].set_index("roi").reindex(ROIS)
        xx = x0 + (k - .5) * .32
        ax.errorbar(xx, gm.ratio, yerr=[gm.ratio - gm.ratio_lo, gm.ratio_hi - gm.ratio],
                    fmt="o", color=MAP_COL[m], ms=6, capsize=0, lw=1.2)
        pred = gm.value_pred_ratio.iloc[0]
        ax.hlines(pred, -0.5, len(ROIS) - .5, color=MAP_COL[m], lw=1.2, ls=":")
        ax.text(len(ROIS) - .4, pred, f"Value code,\n{MAP_LAB[m]}", color=MAP_COL[m],
                fontsize=10.5, va="center")
    ax.axhline(1, color="0.75", lw=0.8, ls="--", zorder=0)
    ax.text(-0.4, 1.12, "No special 90°", color="0.5", fontsize=10.5, va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 60)
    ax.set_yticks([0.1, 1, 10])
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(x0, [ROI_LAB[r] for r in ROIS])
    ax.set_ylabel("d(90°, neighbours) /\nd(other adjacent pairs)")
    letter(ax, "c")

    # d: IPS vs V1 per participant (GDI)
    ax = axes[3]
    w = W[["GDI_V1", "GDI_IPS"]].dropna()
    ax.scatter(w.GDI_V1, w.GDI_IPS, s=26, color="0.25", alpha=.75, lw=0)
    lim = [min(w.min()) - .2, max(w.max()) + .2]
    ax.plot(lim, lim, color="0.75", lw=0.8, ls="--", zorder=0)
    ax.text(lim[1] - .1, lim[1] - .35, "IPS = V1", color="0.5", fontsize=10.5, ha="right", va="top")
    rg = RG[(RG["index"] == "GDI") & (RG.roi == "IPS")].iloc[0]
    ax.text(0.97, 0.03, f"IPS − V1 = {rg.mean_diff:.2f} [{rg.diff_lo:.2f}, {rg.diff_hi:.2f}]\n"
                        f"ρ = {rg.rho:.2f} [{rg.rho_lo:.2f}, {rg.rho_hi:.2f}]\n"
                        f"Reliability V1 {rg.rel_V1:.2f}, IPS {rg.rel_roi:.2f}",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=10.5)
    ax.set_xlabel("V1: 90° distinctiveness")
    ax.set_ylabel("IPS: 90° distinctiveness")
    letter(ax, "d")

    for a in axes:
        sns.despine(ax=a, offset=5)
    # despine's spine offset resets the tick labels, so rotate after it
    plt.setp(axes[2].get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    for ext in ["svg", "png", "pdf"]:
        fig.savefig(OUT / f"fig_a3_categorical_90.{ext}")
    print("wrote", OUT / "fig_a3_categorical_90.svg")


if __name__ == "__main__":
    main()
