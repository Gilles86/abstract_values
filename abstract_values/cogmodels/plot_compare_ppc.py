"""Fixed vs free prior steepness (categorical model, n = 35): posterior predictive checks.

Reads the outputs of compare_ppc.py (rsync'd into notes/data) and the free fit's
per-subject parameters, and draws

  a, b   group bid error and bid SD per orientation, per mapping: observed
         points over the 95% predictive band of each model
  c      per-participant coverage of the 80% predictive band (bid error), fixed
         vs free
  d      per-participant ELPD gain of the free fit against its prior weight w
  e      the four participants who gain most: bid error per orientation with
         both models' 80% bands

    python -m abstract_values.cogmodels.plot_compare_ppc        # abstract_values env
    -> notes/figures/fig_ppc_prior_weight.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.visualize.plot_vmpfc_dropout import style

REPO = Path(__file__).resolve().parents[2]
DATA, FIG = REPO / "notes/data", REPO / "notes/figures"
MAP_COL = {"cdf": "#5B4A9E", "inverse_cdf": "#B23A6E"}
MAP_LAB = {"cdf": "CDF", "inverse_cdf": "Inverse CDF"}
FIX, FREE = "Fixed prior", "Free prior steepness"
FIT_COL = {FIX: "0.55", FREE: "#D9730D"}


def band(ax, x, lo, hi, col, alpha):
    ax.fill_between(x, lo, hi, color=col, alpha=alpha, lw=0)


def main():
    style()
    grp = pd.read_csv(DATA / "ppc_compare_group.tsv", sep="\t")
    sub = pd.read_csv(DATA / "ppc_compare_subjects.tsv", sep="\t")
    loo = pd.read_csv(DATA / "ppc_compare_loo.tsv", sep="\t")
    ls = pd.read_csv(DATA / "ppc_compare_loo_subj.tsv", sep="\t")
    w = pd.read_csv(DATA / "categorical_freeprior_n35_subject_params.tsv", sep="\t")
    w = w.groupby("subject").prior_weight.first() if "subject" in w else w.prior_weight

    fig = plt.figure(figsize=(7.25, 8.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.15, 1.25], hspace=0.45,
                          left=0.08, right=0.98, top=0.95, bottom=0.06)

    # a, b: group curves
    ga = gs[0].subgridspec(2, 2, hspace=0.18, wspace=0.12)
    cov_txt = {}
    for i, (stat, yl) in enumerate((("bias", "Bid − value (CHF)"), ("sd", "Bid SD (CHF)"))):
        for j, m in enumerate(("cdf", "inverse_cdf")):
            ax = fig.add_subplot(ga[i, j])
            for fit, alpha in ((FIX, 0.35), (FREE, 0.3)):
                s = grp[(grp.fit == fit) & (grp.mapping == m) & (grp.draw >= 0)]
                q = s.groupby("orientation")[stat].quantile([0.025, 0.975]).unstack()
                band(ax, q.index, q[0.025], q[0.975], FIT_COL[fit], alpha)
                ob = grp[(grp.fit == fit) & (grp.mapping == m) & (grp.draw < 0)] \
                    .set_index("orientation")[stat]
                cov_txt[(stat, m, fit)] = ((ob >= q[0.025]) & (ob <= q[0.975])).mean()
            ax.plot(ob.index, ob.values, "o", ms=3, color=MAP_COL[m], mec="white", mew=0.4,
                    zorder=3)
            if stat == "bias":
                ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
                ax.set_title(MAP_LAB[m] + " sessions", loc="left", color=MAP_COL[m],
                             fontweight="bold", fontsize=7.5)
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Orientation (°)")
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_xlim(0, 180)
            if j == 0:
                ax.set_ylabel(yl)
            ax.text(0.99, 0.03, f"95% band covers  fixed {100 * cov_txt[(stat, m, FIX)]:.0f}%"
                    f"  ·  free {100 * cov_txt[(stat, m, FREE)]:.0f}%", transform=ax.transAxes,
                    ha="right", fontsize=5.8, color="0.35")
            sns.despine(ax=ax, offset=3)
            if i == 0 and j == 0:
                ax.text(-0.14, 1.08, "a", transform=ax.transAxes, fontsize=9,
                        fontweight="bold")
                for k, fit in enumerate((FIX, FREE)):
                    ax.add_patch(plt.Rectangle((0.03, 0.9 - k * 0.1), 0.06, 0.06,
                                               transform=ax.transAxes, color=FIT_COL[fit],
                                               alpha=0.4, lw=0))
                    ax.text(0.11, 0.93 - k * 0.1, fit, transform=ax.transAxes, fontsize=6.3,
                            va="center", color="0.25")

    # c: subject coverage ; d: ELPD gain vs w ; LOO text
    gc = gs[1].subgridspec(1, 3, wspace=0.45, width_ratios=[1, 1, 0.9])
    cov = sub.assign(inside=lambda d: (d.bias >= d["bias_0.1"]) & (d.bias <= d["bias_0.9"])) \
        .groupby(["fit", "subject"]).inside.mean().unstack(0)
    ax = fig.add_subplot(gc[0])
    ax.scatter(cov[FIX], cov[FREE], s=9, color=FIT_COL[FREE], lw=0, alpha=0.8)
    lim = (min(cov.min().min(), 0.4) - 0.02, 1.0)
    ax.plot(lim, lim, color="0.7", lw=0.6, ls=(0, (3, 2)))
    ax.axhline(0.8, color="0.85", lw=0.6); ax.axvline(0.8, color="0.85", lw=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Coverage, fixed prior")
    ax.set_ylabel("Coverage, free prior")
    ax.set_title("Bid error inside the 80% band", loc="left", fontsize=7)
    ax.text(-0.3, 1.1, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")
    sns.despine(ax=ax, offset=3)

    e = ls.pivot(index="subject", columns="fit", values="elpd")
    gain = (e[FREE] - e[FIX]).rename("gain")
    dd = pd.concat([gain, w.rename("w")], axis=1).dropna()
    ax = fig.add_subplot(gc[1])
    ax.scatter(dd.w, dd.gain, s=9, color=FIT_COL[FREE], lw=0, alpha=0.8)
    ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
    ax.axvline(0.5, color="0.85", lw=0.6)
    ax.set_xlabel("Prior peakedness w (0.5 = fixed prior)")
    ax.set_ylabel("ELPD gain, free − fixed")
    ax.text(-0.3, 1.1, "c", transform=ax.transAxes, fontsize=9, fontweight="bold")
    sns.despine(ax=ax, offset=3)

    ax = fig.add_subplot(gc[2]); ax.axis("off")
    r = loo.set_index("fit")
    best = r.index[0]
    lines = [f"PSIS-LOO (n = {sub.subject.nunique()})", ""]
    for f in r.index:
        lines.append(f"{f}: ELPD {r.loc[f, 'elpd_loo']:.0f}")
    other = r.index[1]
    lines += ["", f"{best} better by", f"{r.loc[other, 'elpd_diff']:.1f} ± {r.loc[other, 'dse']:.1f}",
              f"({r.loc[other, 'elpd_diff'] / r.loc[other, 'dse']:.1f} SE)",
              "", f"Participants gaining: {(gain > 0).sum()} of {len(gain)}"]
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=6.6, color="0.2", linespacing=1.35)

    # e: four participants who gain most
    top = gain.sort_values(ascending=False).index[:4]
    ge = gs[2].subgridspec(1, 4, wspace=0.3)
    for k, s in enumerate(top):
        ax = fig.add_subplot(ge[k])
        for m in ("cdf", "inverse_cdf"):
            for fit, alpha in ((FIX, 0.3), (FREE, 0.3)):
                q = sub[(sub.subject == s) & (sub.mapping == m) & (sub.fit == fit)] \
                    .sort_values("orientation")
                band(ax, q.orientation, q["bias_0.1"], q["bias_0.9"],
                     FIT_COL[fit] if fit == FIX else MAP_COL[m], alpha * (1 if fit == FREE else 0.8))
            ob = q.set_index("orientation").bias
            ax.plot(ob.index, ob.values, "o", ms=2.3, color=MAP_COL[m], mec="white", mew=0.3)
        ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
        ax.set_xticks([0, 90, 180]); ax.set_xlim(0, 180)
        ax.set_xlabel("Orientation (°)")
        ax.set_title(f"sub-{int(s):02d}  w = {w.get(s, np.nan):.2f}  ΔELPD +{gain[s]:.0f}",
                     loc="left", fontsize=6.5)
        if k == 0:
            ax.set_ylabel("Bid − value (CHF)")
            ax.text(-0.35, 1.12, "d", transform=ax.transAxes, fontsize=9, fontweight="bold")
        sns.despine(ax=ax, offset=3)
    fig.text(0.985, 0.305, "Bands (80%): grey = fixed prior, coloured = free prior",
             fontsize=6, color="0.4", ha="right")
    fig.savefig(FIG / "fig_ppc_prior_weight.pdf")
    fig.savefig(FIG / "fig_ppc_prior_weight.png", dpi=200)
    print(loo.to_string())
    print("coverage (80% band, bias), median per subject:", cov.median().round(3).to_dict())
    print(f"Wrote {FIG / 'fig_ppc_prior_weight.pdf'}")


if __name__ == "__main__":
    main()
