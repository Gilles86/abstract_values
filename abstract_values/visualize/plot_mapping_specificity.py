"""Individual differences in following the mapping change: brain vs behaviour.

Each subject gets one number per measure: the slope of their demeaned bias on
the "ignores the mapping" prediction, +/-(v_inv - v_cdf)/2. Slope 1 means the
response to a given orientation is the same whichever mapping is in force;
slope 0 means it follows that condition's values exactly. Behaviour is the BDM
bid, brain is the NPCr value decode.

Panel a is the per-subject paired comparison, panel b asks whether the subjects
who track the mapping better in their bids also do so in NPCr.

    python -m abstract_values.visualize.plot_mapping_specificity
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

C_BEH, C_BRAIN = "#2A9D8F", "#C4442B"


def main(tsv, out):
    j = pd.read_csv(tsv, sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6), constrained_layout=True)

    ax = axes[0]
    for _, r in j.iterrows():
        ax.plot([0, 1], [r.behaviour, r.neural], color="0.75", lw=0.6, zorder=1)
    for i, (col, c) in enumerate([("behaviour", C_BEH), ("neural", C_BRAIN)]):
        v = j[col]
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.1, .1, len(v)),
                   v, s=15, color=c, alpha=.6, linewidths=0, zorder=3)
        ax.errorbar(i, v.mean(), yerr=v.sem(), color=c, marker="D", ms=6.5,
                    mec="0.15", mew=1.2, elinewidth=1.6, zorder=4)
    ax.axhline(1, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.axhline(0, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.annotate("Ignores the mapping", (1.45, 1), xytext=(0, 3),
                textcoords="offset points", fontsize=6.5, color="0.45", ha="right")
    ax.annotate("Follows it fully", (1.45, 0), xytext=(0, 3),
                textcoords="offset points", fontsize=6.5, color="0.45", ha="right")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Bids", "NPCr decode"])
    ax.set_xlim(-.35, 1.5)
    ax.set_ylabel("Slope on the no-change prediction")
    ax.tick_params(axis="x", length=0)

    ax = axes[1]
    ax.scatter(j.behaviour, j.neural, s=22, color="0.35", alpha=.7, linewidths=0)
    r, p = stats.pearsonr(j.behaviour, j.neural)
    b = np.polyfit(j.behaviour, j.neural, 1)
    xs = np.linspace(j.behaviour.min(), j.behaviour.max(), 20)
    ax.plot(xs, np.polyval(b, xs), color="0.35", lw=1.2)
    ax.annotate(f"r = {r:.2f}, p = {p:.2f}\nn = {len(j)}", (0.04, 0.94),
                xycoords="axes fraction", fontsize=6.5, color="0.3", va="top")
    ax.set_xlabel("Bids: slope")
    ax.set_ylabel("NPCr decode: slope")

    sns.despine(fig=fig, offset=4, trim=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv", default="notes/data/mapping_specificity_subjects.tsv")
    p.add_argument("--out", default="notes/figures/mapping_specificity_subjects.pdf")
    a = p.parse_args()
    main(Path(a.tsv), Path(a.out))
