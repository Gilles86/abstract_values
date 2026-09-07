"""Group decoding quality: fidelity and absolute error, per decoded quantity.

Reads the sidecar TSV written by ``decoding_quality_scatter`` (one row per
subject x quantity x ROI) and draws the cohort summary.

Fidelity is on a common [0, 1] scale for both spaces -- Pearson r for value,
the resultant length of the circular decoding error for orientation -- so the
three columns are directly comparable. Absolute error is not, so it gets its
own panel per space with that space's chance level drawn in: 45 deg for an
orientation guessed uniformly, and for value the mean absolute difference
between two randomly drawn presented values.

    python -m abstract_values.visualize.plot_decoding_quality
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

# Blue = orientation space, red = value space; light/dark = V1 / NPCr.
COND = [("gabor", "BensonV1", "Orientation\nV1", "#3B5BA5"),
        ("gabor", "NPCr", "Orientation\nNPCr", "#8FA6D4"),
        ("value-weighted", "NPCr", "Value\nNPCr", "#C4442B")]


def chance_value_mae(trial_table):
    """Mean |xi - xj| between two randomly drawn presented values."""
    v = pd.read_csv(trial_table, sep="\t")["value"].dropna().to_numpy()
    rng = np.random.default_rng(0)
    a, b = rng.choice(v, 20000), rng.choice(v, 20000)
    return float(np.abs(a - b).mean())


def swarm(ax, df, col, conds, chance=None, chance_label=None):
    """One column per condition: per-subject points, mean +/- SEM diamond."""
    labels = [lab for _, _, lab, _ in conds]
    for i, (q, roi, lab, c) in enumerate(conds):
        d = df[(df["quantity"] == q) & (df["roi"] == roi)][col]
        ax.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-.13, .13, len(d)),
                   d, s=13, color=c, alpha=.55, linewidths=0, zorder=3)
        m, se = d.mean(), d.sem()
        ax.errorbar(i, m, yerr=se, color=c, marker="D", ms=6, mec="0.15",
                    mew=1.2, capsize=0, elinewidth=1.6, zorder=4)
        ax.annotate(f"{m:.2f}" if col == "r" else f"{m:.1f}", (i + .2, m),
                    fontsize=7, color=c, fontweight="bold", va="center")
    if chance is not None:
        ax.axhline(chance, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.annotate(chance_label, (len(conds) - .55, chance), fontsize=6.5,
                    color="0.45", va="bottom", ha="right")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-.5, len(labels) - .35)
    return labels


def main(tsv, trial_table, out):
    df = pd.read_csv(tsv, sep="\t")
    df["subject"] = df["subject"].astype(str)
    labels = [lab for _, _, lab, _ in COND]
    colours = [c for _, _, _, c in COND]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), constrained_layout=True)

    swarm(axes[0], df, "r", COND, chance=0.0, chance_label="Chance")
    axes[0].set_ylabel("Decoding fidelity")
    axes[0].set_ylim(-0.02, 0.95)

    # Only the orientation conditions belong on a degrees axis.
    swarm(axes[1], df, "mae", [c for c in COND if c[0] == "gabor"],
          chance=45.0, chance_label="Chance (45°)")
    axes[1].set_ylabel("Absolute error (deg)")
    axes[1].set_ylim(0, 50)

    cmae = chance_value_mae(trial_table)
    val = df[df["quantity"] == "value-weighted"]
    ax = axes[2]
    d = val["mae"]
    ax.scatter(np.full(len(d), 0) + np.random.default_rng(2).uniform(-.13, .13, len(d)),
               d, s=13, color=COND[2][3], alpha=.55, linewidths=0, zorder=3)
    ax.errorbar(0, d.mean(), yerr=d.sem(), color=COND[2][3], marker="D", ms=6,
                mec="0.15", mew=1.2, capsize=0, elinewidth=1.6, zorder=4)
    ax.annotate(f"{d.mean():.1f}", (0.2, d.mean()), fontsize=7,
                color=COND[2][3], fontweight="bold", va="center")
    ax.axhline(cmae, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"Chance ({cmae:.1f} CHF)", (0.45, cmae), fontsize=6.5,
                color="0.45", va="bottom", ha="right")
    ax.set_xticks([0]); ax.set_xticklabels(["Value\nNPCr"])
    ax.set_xlim(-.5, .65)
    ax.set_ylabel("Absolute error (CHF)")
    ax.set_ylim(0, cmae * 1.15)

    # trim=True on a categorical axis leaves a stub of x-spine under the
    # first tick; the category labels are the axis, so drop it entirely.
    sns.despine(fig=fig, offset=4, trim=True, bottom=True)
    for ax in axes:
        ax.tick_params(axis="x", length=0)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")
    print(df.groupby(["quantity", "roi"])[["r", "mae"]].agg(["mean", "sem"]).round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv", default="notes/data/decoding_quality_alpha10.tsv")
    p.add_argument("--trial-table", default="notes/data/trial_table.tsv")
    p.add_argument("--out", default="notes/figures/decoding_quality.pdf")
    a = p.parse_args()
    main(Path(a.tsv), Path(a.trial_table), Path(a.out))
