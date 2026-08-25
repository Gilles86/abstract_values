"""Does the NPCr cross-condition shift relate to behaviour?

The cdf and inverse_cdf mappings are reflections of each other, so between
sessions a participant's value code has to reorganise.  ``aggregate_condition_shift``
summarises that reorganisation per subject from the null-gated NPCr voxels
(cvR2_shift > cvR2_null).  This asks whether any of it shows up in behaviour.

Panels
------
a  The behavioural condition effect itself, on a scale-free axis.  The two
   mappings do NOT span the same CHF range (inverse_cdf spreads values toward
   the tails: value SD 12.3 vs 10.3), so raw bid-error SD is not comparable
   across conditions; each subject's error SD is divided by that condition's
   own stimulus SD.

b  Neural shift magnitude (mean |mode_invcdf - mode_cdf|) against the
   scale-free behavioural condition effect.

c  Efficient-coding alignment.  inverse_cdf spreads density toward the tails,
   so re-tiling should push preferred values OUTWARD from the median.
   frac_outward counts voxels that do; 0.5 is chance.  Read with care: because
   the predicted direction is defined from mode_cdf itself, measurement noise
   in mode_cdf drives this below 0.5 on its own (regression to the mean), so
   the offset is not by itself evidence against efficient coding.

d  Every neural shift measure against every behavioural measure, as a
   correlation heatmap, with tuned-population size included to show which
   apparent shift effects are really that factor in disguise.

Usage:
    python -m abstract_values.visualize.condition_shift_behavior
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
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 4,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "legend.frameon": False, "legend.handlelength": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})
sns.set_context("paper")

COND_C = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
NPCR_C = "#2A9D8F"

NEURAL = [("mean_abs_shift", "Mean |shift|"), ("mean_shift", "Mean signed shift"),
          ("mode_r", "mode$_{cdf}$ vs mode$_{invcdf}$ r"),
          ("outward_index", "Outward index"),
          ("n_shift_vox", "N shift voxels"),
          ("log_tuned_vonmises", "Tuned vertices (log)")]
BEHAV = [("behav_sd", "Bid error SD"), ("behav_shift_norm", "Condition effect\n(scale-free)"),
         ("bias_profile_r", "Bias profile r"), ("bias_profile_rmsd", "Bias profile RMSD")]


def panel_condition_effect(ax, agg, norm):
    for i, (s, row) in enumerate(norm.iterrows()):
        ax.plot([0, 1], [row["cdf"], row["inverse_cdf"]], color="0.7", lw=0.7, zorder=1)
    for x, cond in [(0, "cdf"), (1, "inverse_cdf")]:
        ax.scatter(np.full(len(norm), x), norm[cond], s=28, color=COND_C[cond],
                   zorder=3, edgecolor="white", linewidth=0.5)
        ax.plot([x - 0.16, x + 0.16], [norm[cond].mean()] * 2,
                color=COND_C[cond], lw=2.4, zorder=4)
    t, p = stats.ttest_rel(norm["cdf"], norm["inverse_cdf"])
    ax.annotate(f"Paired t({len(norm)-1}) = {t:+.2f}\np = {p:.1e}",
                xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top",
                fontsize=8, color="0.2")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["CDF", "Inverse CDF"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Bid error SD / stimulus SD")
    ax.set_title("Behavioural condition effect", fontsize=9, color="0.2")


def panel_scatter(ax, agg, xc, yc, xlabel, ylabel, colour=NPCR_C):
    d = agg[[xc, yc]].dropna()
    x, y = d[xc].values, d[yc].values
    r, p = stats.pearsonr(x, y)
    ax.scatter(x, y, s=26, color=colour, alpha=0.75, edgecolor="white", linewidth=0.5)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=colour, lw=1.4)
    ax.annotate(f"r = {r:+.2f}, p = {p:.3f}", xy=(0.04, 0.95),
                xycoords="axes fraction", ha="left", va="top", fontsize=8, color="0.2")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


def panel_outward(ax, agg):
    v = agg["frac_outward"].dropna().sort_values()
    ax.barh(np.arange(len(v)), v - 0.5, left=0.5, color=NPCR_C, alpha=0.75, height=0.75)
    ax.axvline(0.5, color="0.25", lw=1.0, ls="--")
    t, p = stats.ttest_1samp(v, 0.5)
    ax.annotate(f"All {len(v)}/{len(v)} below chance\nt({len(v)-1}) = {t:.1f}, p = {p:.0e}",
                xy=(0.04, 0.04), xycoords="axes fraction", ha="left", va="bottom",
                fontsize=7.5, color="0.2")
    ax.set_yticks([]); ax.set_ylabel("Subject (sorted)", fontsize=8)
    ax.set_xlabel("Fraction of voxels shifting outward")
    ax.set_title("Efficient-coding alignment\n(0.5 = chance; see caveat)",
                 fontsize=8.5, color="0.2")


def panel_heatmap(ax, agg):
    M = np.full((len(NEURAL), len(BEHAV)), np.nan)
    P = np.full_like(M, np.nan)
    for i, (nc, _) in enumerate(NEURAL):
        for j, (bc, _) in enumerate(BEHAV):
            d = agg[[nc, bc]].dropna()
            M[i, j], P[i, j] = stats.pearsonr(d[nc], d[bc])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:+.2f}" + ("*" if P[i, j] < 0.05 else ""),
                    ha="center", va="center", fontsize=7.5,
                    color="white" if abs(M[i, j]) > 0.45 else "0.15")
    ax.set_xticks(range(len(BEHAV)))
    ax.set_xticklabels([l for _, l in BEHAV], fontsize=7.5)
    ax.set_yticks(range(len(NEURAL)))
    ax.set_yticklabels([l for _, l in NEURAL], fontsize=7.5)
    ax.set_title("Neural × behavioural correlations (* p < 0.05, uncorrected)",
                 fontsize=8.5, color="0.2")
    for sp in ax.spines.values():
        sp.set_visible(False)
    plt.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")


def run(summary_tsv, norm_tsv, out):
    agg = pd.read_csv(summary_tsv, sep="\t").set_index("subject")
    norm = pd.read_csv(norm_tsv, sep="\t").set_index("subject")

    fig = plt.figure(figsize=(11.5, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    panel_condition_effect(fig.add_subplot(gs[0, 0]), agg, norm)
    panel_scatter(fig.add_subplot(gs[0, 1]), agg, "mean_abs_shift", "behav_shift_norm",
                  "Mean |mode$_{invcdf}$ − mode$_{cdf}$| (CHF)",
                  "Behavioural condition effect\n(CDF − InvCDF, scale-free)")
    panel_outward(fig.add_subplot(gs[0, 2]), agg)
    panel_heatmap(fig.add_subplot(gs[1, :]), agg)

    fig.suptitle(f"NPCr cross-condition shift vs behaviour  ·  n = {len(agg)} subjects",
                 fontsize=10, y=1.02, color="0.15")
    sns.despine(fig=fig, offset=4, trim=False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-tsv", default="notes/data/brain_behavior_subject_summary.tsv")
    p.add_argument("--norm-tsv", default="notes/data/behav_sd_normalised.tsv")
    p.add_argument("--out", default="notes/figures/condition_shift_behavior.pdf")
    a = p.parse_args()
    run(a.summary_tsv, a.norm_tsv, a.out)


if __name__ == "__main__":
    main()
