"""Is the preferred stimulus stable across sessions? V1 orientation vs NPCr value.

The control for the contraction seen on the value side. If preferred values in
NPCr look unreliable across the two mappings, the first question is whether the
session-shift machinery can recover a stable parameter at all -- so run the
same fit in V1, where the stimulus dimension (orientation) is physically
identical in both sessions and the preferred orientation has no reason to move.

V1 uses the axial circular resultant of the mu difference (period pi) as its
stability measure; the value side uses the correlation and regression slope,
which is what the CHF axis admits. The shared, unit-free comparison is the
median absolute shift as a fraction of each stimulus range, in panel c.

    python -m abstract_values.visualize.plot_mode_stability
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

C_V1, C_NPC = "#3B5BA5", "#C4442B"


def main(v1_tsv, npcr_tsv, out):
    v = pd.read_csv(v1_tsv, sep="\t")
    m = pd.read_csv(npcr_tsv, sep="\t")
    v["mu_1_deg"] = np.rad2deg(v.mode_1) % 180
    v["mu_2_deg"] = np.rad2deg(v.mode_2) % 180
    z = np.exp(1j * 2 * (v.mode_2 - v.mode_1))
    v["shift_deg"] = np.rad2deg(np.angle(z) / 2)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)

    ax = axes[0]
    ax.scatter(v.mu_1_deg, v.mu_2_deg, s=4, alpha=.25, color=C_V1,
               linewidths=0, rasterized=True)
    ax.plot([0, 180], [0, 180], color="0.55", lw=0.8, ls=(0, (4, 3)))
    R = v.groupby("subject").apply(
        lambda g: np.abs(np.exp(1j * 2 * (g.mode_2 - g.mode_1)).mean()),
        include_groups=False)
    ax.annotate(f"R = {R.mean():.2f}", (0.05, 0.93), xycoords="axes fraction",
                fontsize=7, color=C_V1, fontweight="bold")
    ax.set_xlim(0, 180); ax.set_ylim(0, 180)
    ax.set_xticks([0, 90, 180]); ax.set_yticks([0, 90, 180])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("V1 · orientation", fontsize=8, color="0.2")
    ax.set_xlabel("Preferred, session 1 (deg)")
    ax.set_ylabel("Preferred, session 2 (deg)")

    ax = axes[1]
    ax.scatter(m.mode_cdf, m.mode_inv, s=4, alpha=.25, color=C_NPC,
               linewidths=0, rasterized=True)
    ax.plot([0, 45], [0, 45], color="0.55", lw=0.8, ls=(0, (4, 3)))
    sl = m.groupby("subject").apply(
        lambda g: pd.Series(dict(zip(("slope", "inter"),
                                     np.polyfit(g.mode_cdf, g.mode_inv, 1)))),
        include_groups=False)
    xs = np.linspace(0, 45, 40)
    ax.plot(xs, sl["inter"].mean() + sl["slope"].mean() * xs, color=C_NPC, lw=1.6)
    ax.annotate(f"Slope {sl['slope'].mean():.2f}", (0.05, 0.93),
                xycoords="axes fraction", fontsize=7, color=C_NPC,
                fontweight="bold")
    ax.set_xlim(0, 45); ax.set_ylim(0, 45)
    ax.set_xticks([0, 22.5, 45]); ax.set_yticks([0, 22.5, 45])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("NPCr · value", fontsize=8, color="0.2")
    ax.set_xlabel("Preferred, CDF (CHF)")
    ax.set_ylabel("Preferred, inverse CDF (CHF)")

    # c — the comparable number: shift as a fraction of the stimulus range
    ax = axes[2]
    frac = pd.DataFrame({
        "V1": v.groupby("subject")["shift_deg"].apply(
            lambda s: np.median(np.abs(s)) / 180),
        "NPCr": m.groupby("subject")["dshift"].apply(
            lambda s: np.median(np.abs(s)) / 40)})
    for i, (col, c) in enumerate([("V1", C_V1), ("NPCr", C_NPC)]):
        d = frac[col].dropna()
        ax.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-.12, .12, len(d)),
                   100 * d, s=14, color=c, alpha=.6, linewidths=0, zorder=3)
        ax.errorbar(i, 100 * d.mean(), yerr=100 * d.sem(), color=c, marker="D",
                    ms=6, mec="0.15", mew=1.2, elinewidth=1.6, zorder=4)
        ax.annotate(f"{100 * d.mean():.0f}%", (i + .2, 100 * d.mean()),
                    fontsize=7, color=c, fontweight="bold", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["V1\norientation", "NPCr\nvalue"])
    ax.set_xlim(-.45, 1.5)
    ax.set_ylabel("Median shift (% of range)")
    ax.set_ylim(0, None)

    sns.despine(fig=fig, offset=4, trim=True)
    axes[2].tick_params(axis="x", length=0)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")
    print(f"V1 resultant R = {R.mean():.3f}; NPCr slope = {sl['slope'].mean():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v1", default="notes/data/session_shift_modes_v1.tsv")
    p.add_argument("--npcr", default="notes/data/session_shift_modes_npcr.tsv")
    p.add_argument("--out", default="notes/figures/mode_stability_v1_vs_npcr.pdf")
    a = p.parse_args()
    main(Path(a.v1), Path(a.npcr), Path(a.out))
