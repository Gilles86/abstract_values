"""Which hypothesis generalises across conditions: value or orientation?

Fit each voxel's value tuning on one condition, predict the other condition's
response to every orientation two ways -- reading the same curve at the value
the new condition assigns (value hypothesis) or at the value the training
condition assigned (orientation hypothesis) -- and see which matches. Same
curve, same fit, so the comparison is about read-out, not flexibility.

Both directions of generalisation are averaged, and voxel selection uses the
training condition's R2 only.

    python -m abstract_values.visualize.plot_condition_generalisation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "lines.linewidth": 1.2, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

VAL_C, ORI_C = "#C4442B", "#3B5BA5"
ROIS = [("BensonV1", "V1"), ("NPCr", "NPCr")]


def main(tsv, out):
    d = pd.read_csv(tsv, sep="\t")
    per = (d.groupby(["roi", "subject"])[["r_value", "r_orientation",
                                          "r_advantage"]].mean().reset_index())

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), constrained_layout=True)

    for ax, (roi, label) in zip(axes[:2], ROIS):
        g = per[per.roi == roi]
        for i, (col, c, name) in enumerate([("r_orientation", ORI_C, "Orientation"),
                                            ("r_value", VAL_C, "Value")]):
            ax.scatter(np.full(len(g), i) + np.random.default_rng(i).uniform(-.12, .12, len(g)),
                       g[col], s=14, color=c, alpha=.55, linewidths=0, zorder=3)
            ax.errorbar(i, g[col].mean(), yerr=g[col].sem(), color=c, marker="D",
                        ms=6, mec="0.15", mew=1.2, elinewidth=1.6, zorder=4)
        for _, r in g.iterrows():
            ax.plot([0, 1], [r.r_orientation, r.r_value], color="0.75", lw=0.5,
                    zorder=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Orientation", "Value"])
        ax.set_xlim(-.4, 1.4)
        ax.set_ylabel("Generalisation (r)")
        ax.set_title(label, fontsize=8, color="0.2")
        ax.axhline(0, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=0)

    ax = axes[2]
    for i, (roi, label) in enumerate(ROIS):
        g = per[per.roi == roi]["r_advantage"]
        c = "0.35"
        ax.scatter(np.full(len(g), i) + np.random.default_rng(i + 5).uniform(-.12, .12, len(g)),
                   g, s=14, color=c, alpha=.5, linewidths=0, zorder=3)
        ax.errorbar(i, g.mean(), yerr=g.sem(), color=c, marker="D", ms=6,
                    mec="0.15", mew=1.2, elinewidth=1.6, zorder=4)
        t, p = stats.ttest_1samp(g, 0)
        ax.annotate(f"p = {p:.3f}", (i, g.max()), xytext=(0, 4),
                    textcoords="offset points", fontsize=6.5, color="0.35",
                    ha="center")
    ax.axhline(0, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.annotate("Value better", (1.45, 0), xytext=(0, 4),
                textcoords="offset points", fontsize=6.5, color=VAL_C, ha="right")
    ax.annotate("Orientation better", (1.45, 0), xytext=(0, -10),
                textcoords="offset points", fontsize=6.5, color=ORI_C, ha="right")
    ax.set_xticks([0, 1]); ax.set_xticklabels([l for _, l in ROIS])
    ax.set_xlim(-.4, 1.5)
    ax.set_ylabel("Value − orientation (r)")

    sns.despine(fig=fig, offset=4, trim=True)
    for a in axes:
        a.tick_params(axis="x", length=0)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv", default="notes/data/condition_generalisation.tsv")
    p.add_argument("--out",
                   default="notes/figures/condition_generalisation.pdf")
    a = p.parse_args()
    main(Path(a.tsv), Path(a.out))
