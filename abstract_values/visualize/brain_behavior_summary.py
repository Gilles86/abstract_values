"""Do decoding quality, tuned-population size, and the cross-condition shift
relate to behaviour?

Every measure is per subject, and every one is related to the same behavioural
outcome: the SD of the participant's bid error.  The BDM auction is
truth-telling, so ``error = response - value`` (see CLAUDE.md).

Neural measures
---------------
Decoding fidelity -- the within-subject correlation between the *ground truth*
and the *decoded posterior mean* across trials.  Pearson r for the value
decoders; Jammalamadaka-Sarma circular correlation (pi-periodic) for the
orientation decoders.  This is the project's standard decoding-quality metric,
not posterior SD or |decoding error|.

Tuned-population size -- how many fsaverage vertices the encoding model
explains out of sample (cvR2 > 0), from the surface-sampled cvR2 maps; the
vonmises (orientation) and aPRF (value) models are counted separately.  Counts
span three orders of magnitude across subjects, so they enter on a log10 axis.
The NPCr session-shift voxel count (cvR2_shift > cvR2_null) is the same idea
restricted to the ROI and to voxels whose tuning genuinely remaps.

Cross-condition shift -- per subject, the mean |mode_invcdf - mode_cdf| over
null-gated NPCr voxels, plus the correlation between the two per-condition
preferred values.  How much the value code reorganises when the
orientation->CHF mapping flips.

Head motion (mean FD) is partialled out throughout: it predicts tuned-population
size outright, so a raw brain-behaviour correlation could be data quality in
disguise.

Usage:
    python -m abstract_values.visualize.brain_behavior_summary
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

NPCR_C, V1_C, EXT_C, NULL_C = "#2A9D8F", "#E76F51", "#264653", "0.55"

# (column, short label, colour) -- ordered as they appear in the forest panel.
MEASURES = [
    ("log_tuned_vonmises", "Tuned vertices, vonmises (log)", EXT_C),
    ("log_tuned_aprf",     "Tuned vertices, aPRF (log)",     EXT_C),
    ("n_shift_vox",        "NPCr session-shift voxels",      EXT_C),
    ("npcr_val_r",         "NPCr value decoding r",          NPCR_C),
    ("v1_ori_r",           "V1 orientation decoding r",      V1_C),
    ("v1_val_r",           "V1 value decoding r",            V1_C),
    ("npcr_ori_r",         "NPCr orientation decoding r",    NULL_C),
    ("mean_abs_shift",     "NPCr condition shift (CHF)",     NULL_C),
    ("mode_r",             "NPCr mode_cdf vs mode_invcdf r", NULL_C),
]


def partial_r(x, y, z):
    """Pearson r between x and y with z partialled out, plus two-sided p."""
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[ok], y[ok], z[ok]
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    r = np.corrcoef(rx, ry)[0, 1]
    n = len(x)
    t = r * np.sqrt((n - 3) / max(1e-12, 1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), n - 3)


def panel_scatter(ax, agg, col, label, colour, xlabel=None):
    d = agg[[col, "behav_sd"]].dropna()
    x, y = d[col].values, d["behav_sd"].values
    r, p = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    ax.scatter(x, y, s=26, color=colour, alpha=0.75, edgecolor="white", linewidth=0.5)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=colour, lw=1.4)
    ax.set_xlabel(xlabel or label)
    ax.set_ylabel("Behavioural SD of bid error (CHF)")
    star = " *" if p < 0.05 else ""
    ax.annotate(f"r = {r:+.2f}, p = {p:.3f}{star}\n" + r"$\rho$ = " + f"{rho:+.2f}",
                xy=(0.04, 0.95), xycoords="axes fraction", ha="left", va="top",
                fontsize=8, color="0.2")


def panel_forest(ax, agg):
    rows = []
    for col, label, colour in MEASURES:
        d = agg[[col, "behav_sd"]].dropna()
        r_raw, p_raw = stats.pearsonr(d[col], d["behav_sd"])
        r_par, p_par = partial_r(agg[col].values, agg["behav_sd"].values,
                                 agg["mean_fd"].values)
        rows.append((label, colour, r_raw, p_raw, r_par, p_par))
    rows = rows[::-1]
    ys = np.arange(len(rows))
    for y, (label, colour, r_raw, p_raw, r_par, p_par) in zip(ys, rows):
        ax.plot([0, r_raw], [y + 0.16] * 2, color=colour, lw=1.0, alpha=0.45)
        ax.scatter(r_raw, y + 0.16, s=34, color=colour, zorder=3,
                   edgecolor="white", linewidth=0.5)
        ax.scatter(r_par, y - 0.16, s=34, facecolor="white", zorder=3,
                   edgecolor=colour, linewidth=1.2)
        if p_raw < 0.05:
            ax.annotate("*", xy=(r_raw - 0.035, y + 0.16), ha="center", va="center",
                        fontsize=12, color=colour)
    ax.axvline(0, color="0.75", lw=0.8, ls="--", zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Correlation with behavioural SD of bid error")
    ax.set_title("Filled = raw · open = head-motion (mean FD) partialled · * p < 0.05",
                 fontsize=8, color="0.35")
    ax.set_ylim(-0.7, len(rows) - 0.3)


def panel_shift(ax, agg):
    """The cross-condition shift itself: how far NPCr preferred values move."""
    d = agg[["mean_abs_shift", "mode_r"]].dropna().sort_values("mean_abs_shift")
    ys = np.arange(len(d))
    ax.barh(ys, d["mean_abs_shift"], color=NPCR_C, alpha=0.7, height=0.72)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{int(s):02d}" for s in d.index], fontsize=6.5)
    ax.set_xlabel("Mean |mode$_{invcdf}$ − mode$_{cdf}$| (CHF)")
    ax.set_ylabel("Subject", fontsize=8)
    m = d["mean_abs_shift"].mean()
    ax.axvline(m, color="0.25", lw=1.0, ls="--")
    ax.annotate(f"Mean {m:.1f} CHF", xy=(m, len(d) - 0.5), xytext=(3, 0),
                textcoords="offset points", fontsize=7.5, color="0.25",
                ha="left", va="top")
    ax.set_title("Cross-condition shift in NPCr\n(null-gated voxels)",
                 fontsize=8, color="0.35")


def run(summary_tsv, out):
    agg = pd.read_csv(summary_tsv, sep="\t").set_index("subject")
    if "log_tuned_vonmises" not in agg:
        agg["log_tuned_vonmises"] = np.log10(agg["tuned_vonmises"].clip(lower=1))
        agg["log_tuned_aprf"] = np.log10(agg["tuned_aprf"].clip(lower=1))
    n = len(agg)

    fig = plt.figure(figsize=(11.5, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15])

    panel_scatter(fig.add_subplot(gs[0, 0]), agg, "log_tuned_vonmises",
                  "", EXT_C, xlabel="Tuned vertices, vonmises (log$_{10}$)")
    panel_scatter(fig.add_subplot(gs[0, 1]), agg, "n_shift_vox",
                  "", EXT_C, xlabel="NPCr session-shift voxels (cvR² > null)")
    panel_scatter(fig.add_subplot(gs[0, 2]), agg, "npcr_val_r",
                  "", NPCR_C, xlabel="NPCr value decoding r (truth vs decoded)")

    panel_forest(fig.add_subplot(gs[1, 0:2]), agg)
    panel_shift(fig.add_subplot(gs[1, 2]), agg)

    fig.suptitle(f"Neural measures vs behavioural precision  ·  n = {n} subjects",
                 fontsize=10, y=1.02, color="0.15")
    sns.despine(fig=fig, offset=4, trim=False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-tsv",
                   default="notes/data/brain_behavior_subject_summary.tsv")
    p.add_argument("--out", default="notes/figures/brain_behavior_summary.pdf")
    a = p.parse_args()
    run(a.summary_tsv, a.out)


if __name__ == "__main__":
    main()
