"""Does a less acute neural value code go with more regression to the prior?

The Bayesian account: a bid combines a noisy likelihood (the neural value
estimate) with the prior over values.  The noisier the likelihood, the more the
bid is pulled toward the centre of the stimulus distribution -- i.e. the slope
of response-on-value drops below 1.  So wherever the NPCr value code is less
acute, the regression index (1 - slope) should be larger.

The manipulation gives two handles on this:

  Within subject -- the cdf and inverse_cdf mappings differ in how precisely
  NPCr encodes value, and each subject does both.  This is the powerful test,
  because every between-subject nuisance cancels.

  Between subjects -- do people with a more acute code regress less?  Same
  prediction, far less power at n = 26.

V1 acts as the control throughout: it is orientation-tuned, so its expected
decoding uncertainty should not care which mapping is in force.

Usage:
    python -m abstract_values.visualize.bayesian_bias_vs_acuity
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
NPCR_C, V1_C = "#2A9D8F", "#E76F51"


def paired_panel(ax, agg, ca, cb, ylabel, title, unit=""):
    a, b = agg[ca].dropna(), agg[cb].dropna()
    for va, vb in zip(a, b):
        ax.plot([0, 1], [va, vb], color="0.75", lw=0.7, zorder=1)
    ax.scatter(np.zeros(len(a)), a, s=24, color=COND_C["cdf"], zorder=3,
               edgecolor="white", linewidth=0.5)
    ax.scatter(np.ones(len(b)), b, s=24, color=COND_C["inverse_cdf"], zorder=3,
               edgecolor="white", linewidth=0.5)
    for x, v, c in [(0, a, "cdf"), (1, b, "inverse_cdf")]:
        ax.plot([x - 0.17, x + 0.17], [v.mean()] * 2, color=COND_C[c], lw=2.4, zorder=4)
    t, p = stats.ttest_rel(a, b)
    ax.annotate(f"t({len(a)-1}) = {t:+.2f}\np = {p:.1e}", xy=(0.5, 0.98),
                xycoords="axes fraction", ha="center", va="top", fontsize=7.5, color="0.2")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["CDF", "Inv CDF"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(ylabel + (f" ({unit})" if unit else ""))
    ax.set_title(title, fontsize=8.5, color="0.2")


def panel_slopes(ax, agg):
    """Regression of bid on true value, per condition -- slope < 1 = pulled to prior."""
    for c, col in [("cdf", "slope_cdf"), ("inverse_cdf", "slope_invcdf")]:
        v = agg[col].dropna()
        parts = ax.violinplot([v], positions=[0 if c == "cdf" else 1], widths=0.6,
                              showextrema=False, showmedians=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(COND_C[c]); pc.set_alpha(0.28); pc.set_edgecolor("none")
        ax.scatter(np.full(len(v), 0 if c == "cdf" else 1) +
                   np.random.default_rng(1).uniform(-0.09, 0.09, len(v)),
                   v, s=20, color=COND_C[c], alpha=0.8, edgecolor="white", linewidth=0.4)
        t, p = stats.ttest_1samp(v, 1.0)
        ax.annotate(f"{v.mean():.3f}\np = {p:.3f}", xy=(0 if c == "cdf" else 1, 1.13),
                    ha="center", va="top", fontsize=7.5, color=COND_C[c])
    ax.axhline(1.0, color="0.3", ls="--", lw=1.0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["CDF", "Inv CDF"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Slope of bid on true value")
    ax.set_title("Regression to the prior\n(slope < 1 = pulled to centre)",
                 fontsize=8.5, color="0.2")


def panel_within(ax, agg, ycol, ylabel, title):
    a = agg.reset_index()
    long = pd.concat([
        pd.DataFrame({"subject": a["subject"], "npcr_sd": a["npcr_sd_cdf"], "y": a[f"{ycol}_cdf"]}),
        pd.DataFrame({"subject": a["subject"], "npcr_sd": a["npcr_sd_invcdf"], "y": a[f"{ycol}_invcdf"]}),
    ]).dropna()
    for c in ["npcr_sd", "y"]:
        long[c] = long.groupby("subject")[c].transform(lambda s: s - s.mean())
    r, p = stats.pearsonr(long["npcr_sd"], long["y"])
    ax.scatter(long["npcr_sd"], long["y"], s=22, color=NPCR_C, alpha=0.7,
               edgecolor="white", linewidth=0.4)
    xs = np.linspace(long["npcr_sd"].min(), long["npcr_sd"].max(), 40)
    ax.plot(xs, np.polyval(np.polyfit(long["npcr_sd"], long["y"], 1), xs),
            color=NPCR_C, lw=1.5)
    ax.axhline(0, color="0.85", lw=0.7, zorder=0); ax.axvline(0, color="0.85", lw=0.7, zorder=0)
    ax.annotate(f"r = {r:+.2f}, p = {p:.4f}", xy=(0.04, 0.95), xycoords="axes fraction",
                ha="left", va="top", fontsize=8, color="0.2")
    ax.set_xlabel("NPCr decoder SD (condition-centred)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8.5, color="0.2")


def panel_between(ax, agg):
    meas = [("npcr_val_r", "NPCr value\ndecoding r", NPCR_C),
            ("log_tuned_aprf", "Tuned vertices\naPRF (log)", NPCR_C),
            ("v1_val_r", "V1 value\ndecoding r", V1_C),
            ("log_tuned_vonmises", "Tuned vertices\nvonmises (log)", V1_C),
            ("v1_ori_r", "V1 orientation\ndecoding r", "0.55")]
    ys = np.arange(len(meas))[::-1]
    for y, (col, lab, colour) in zip(ys, meas):
        d = agg[[col, "reg_mean"]].dropna()
        r, p = stats.pearsonr(d[col], d["reg_mean"])
        ax.plot([0, r], [y] * 2, color=colour, lw=1.0, alpha=0.45)
        ax.scatter(r, y, s=36, color=colour, zorder=3, edgecolor="white", linewidth=0.5)
        ax.annotate(f"p = {p:.2f}", xy=(r, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=7, color="0.35")
    ax.axvline(0, color="0.75", lw=0.8, ls="--", zorder=0)
    ax.set_yticks(ys); ax.set_yticklabels([m[1] for m in meas], fontsize=7.5)
    ax.set_xlim(-0.55, 0.35)
    ax.set_xlabel("Correlation with regression index")
    ax.set_title("Between subjects: does acuity predict bias?\n"
                 "(predicted negative; none reach p < 0.05)", fontsize=8.5, color="0.2")


def run(summary_tsv, out):
    agg = pd.read_csv(summary_tsv, sep="\t").set_index("subject")
    fig = plt.figure(figsize=(12.0, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    paired_panel(fig.add_subplot(gs[0, 0]), agg, "npcr_sd_cdf", "npcr_sd_invcdf",
                 "NPCr decoder SD", "NPCr value code\nis noisier under Inv CDF", "CHF")
    paired_panel(fig.add_subplot(gs[0, 1]), agg, "v1_sd_cdf", "v1_sd_invcdf",
                 "V1 decoder SD", "V1 control:\nno condition effect", "deg")
    panel_slopes(fig.add_subplot(gs[0, 2]), agg)
    paired_panel(fig.add_subplot(gs[0, 3]), agg, "reg_cdf", "reg_invcdf",
                 "Regression index (1 − slope)", "More bias under Inv CDF —\nthe noisier condition")

    panel_within(fig.add_subplot(gs[1, 0]), agg, "behav_sd",
                 "Bid error SD (condition-centred)",
                 "Within subject: neural noise → behavioural noise")
    panel_within(fig.add_subplot(gs[1, 1]), agg, "reg",
                 "Regression index (condition-centred)",
                 "Within subject: neural noise → bias")
    panel_between(fig.add_subplot(gs[1, 2:]), agg)

    fig.suptitle("Bayesian account: less acute value code → more regression to the prior  ·  n = 26",
                 fontsize=10, y=1.03, color="0.15")
    sns.despine(fig=fig, offset=4, trim=False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-tsv", default="notes/data/brain_behavior_subject_summary.tsv")
    p.add_argument("--out", default="notes/figures/bayesian_bias_vs_acuity.pdf")
    a = p.parse_args()
    run(a.summary_tsv, a.out)


if __name__ == "__main__":
    main()
