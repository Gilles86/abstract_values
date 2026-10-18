"""Posterior predictive checks of two efficient-coding fits, side by side.

Reads compare_ppc.py output (--in-dir) and compares a reference fit (--a) with
an extended one (--b):

  a  group bid error and bid SD per orientation and mapping: observed points
     over both fits' 95% predictive bands
  b  the same for three example participants (bid error only), the ones whose
     fit improves most
  c  per-cell coverage of the participant-level predictive intervals (every
     participant x mapping x orientation cell; nominal 50 / 80 / 95%)
  d  PSIS-LOO gain of b over a per participant

    python -m abstract_values.cogmodels.plot_ppc_pair --in-dir notes/data/compare_ppc_spline \\
        --a "No offsets" --b "Spline offsets"
    -> notes/figures/fig_ppc_<in-dir name>.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.visualize.plot_vmpfc_dropout import style

REPO = Path(__file__).resolve().parents[2]
MAP_COL = {"cdf": "#5B4A9E", "inverse_cdf": "#B23A6E"}
MAP_LAB = {"cdf": "CDF", "inverse_cdf": "Inverse CDF"}
COL_A, COL_B = "0.45", "#C2560A"
LEVELS = ((0.25, 0.75, 50), (0.1, 0.9, 80), (0.025, 0.975, 95))


def band(grp, fit, m, stat):
    s = grp[(grp.fit == fit) & (grp.mapping == m)]
    q = s[s.draw >= 0].groupby("orientation")[stat].quantile([0.025, 0.975]).unstack()
    return q, s[s.draw < 0].set_index("orientation")[stat]


def coverage(sub, fit, stat):
    s = sub[sub.fit == fit]
    return {n: 100 * ((s[stat] >= s[f"{stat}_{lo}"]) & (s[stat] <= s[f"{stat}_{hi}"])).mean()
            for lo, hi, n in LEVELS}


def key(ax, entries, x=0.03, y=0.95, dy=0.09):
    for i, (lab, col, alpha) in enumerate(entries):
        yy = y - i * dy
        ax.add_patch(plt.Rectangle((x, yy - 0.025), 0.07, 0.05, transform=ax.transAxes,
                                   color=col, alpha=alpha, lw=0))
        ax.text(x + 0.09, yy, lab, transform=ax.transAxes, fontsize=6.3, va="center",
                color="0.25")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    a = p.parse_args()
    ind = Path(a.in_dir)
    grp = pd.read_csv(ind / "ppc_compare_group.tsv", sep="\t")
    sub = pd.read_csv(ind / "ppc_compare_subjects.tsv", sep="\t")
    ls = pd.read_csv(ind / "ppc_compare_loo_subj.tsv", sep="\t")
    loo = pd.read_csv(ind / "ppc_compare_loo.tsv", sep="\t").set_index("fit")
    style()

    fig = plt.figure(figsize=(7.25, 8.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.05, 1.1], hspace=0.45,
                          left=0.1, right=0.98, top=0.965, bottom=0.06)

    ga = gs[0].subgridspec(2, 2, hspace=0.15, wspace=0.14)
    for i, (stat, yl) in enumerate((("bias", "Bid − value (CHF)"), ("sd", "Bid SD (CHF)"))):
        for j, m in enumerate(("cdf", "inverse_cdf")):
            ax = fig.add_subplot(ga[i, j])
            cov = {}
            for fit, col, al in ((a.a, COL_A, 0.28), (a.b, COL_B, 0.35)):
                q, ob = band(grp, fit, m, stat)
                ax.fill_between(q.index, q[0.025], q[0.975], color=col, alpha=al, lw=0)
                cov[fit] = 100 * ((ob >= q[0.025]) & (ob <= q[0.975])).mean()
            ax.plot(ob.index, ob.values, "o", ms=3, color=MAP_COL[m], mec="white", mew=0.4,
                    zorder=3)
            ax.set_xticks([0, 45, 90, 135, 180]); ax.set_xlim(0, 180)
            if stat == "bias":
                ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
                ax.set_title(MAP_LAB[m] + " sessions", loc="left", color=MAP_COL[m],
                             fontweight="bold", fontsize=7.5)
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Orientation (°)")
            if j == 0:
                ax.set_ylabel(yl)
            ax.text(0.99, 0.03, f"Group points inside 95% band: {cov[a.a]:.0f}% vs {cov[a.b]:.0f}%",
                    transform=ax.transAxes, ha="right", fontsize=5.8,
                    color="0.35")
            sns.despine(ax=ax, offset=3)
            if i == 0 and j == 0:
                ax.text(-0.14, 1.08, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
                key(ax, [(a.a, COL_A, 0.4), (a.b, COL_B, 0.5)])

    # b: example participants -- largest per-participant LOO gain
    gain = (ls[ls.fit == a.b].set_index("subject").elpd
            - ls[ls.fit == a.a].set_index("subject").elpd).sort_values(ascending=False)
    gb = gs[1].subgridspec(1, 3, wspace=0.22)
    for k, s in enumerate(gain.index[:3]):
        ax = fig.add_subplot(gb[k])
        for m in ("cdf", "inverse_cdf"):
            d = sub[(sub.fit == a.a) & (sub.subject == s) & (sub.mapping == m)
                    ].set_index("orientation")
            for e in ("bias_0.025", "bias_0.975"):
                ax.plot(d.index, d[e], color=MAP_COL[m], lw=0.6, ls=(0, (2, 1.5)), alpha=0.8)
            d = sub[(sub.fit == a.b) & (sub.subject == s) & (sub.mapping == m)
                    ].set_index("orientation")
            ax.fill_between(d.index, d["bias_0.025"], d["bias_0.975"], color=MAP_COL[m],
                            alpha=0.25, lw=0)
            ax.plot(d.index, d.bias, "o", ms=2.6, color=MAP_COL[m], mec="white", mew=0.3,
                    zorder=3)
        ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
        ax.set_xticks([0, 90, 180]); ax.set_xlim(0, 180)
        ax.set_xlabel("Orientation (°)")
        ax.set_title(f"sub-{s:02d}: ΔELPD +{gain[s]:.0f}", loc="left", fontsize=7)
        if k == 0:
            ax.set_ylabel("Bid − value (CHF)")
            ax.text(-0.3, 1.3, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")
            tf = ax.transAxes
            ax.plot([0.0, 0.1], [1.27, 1.27], transform=tf, color="0.3", lw=0.6,
                    ls=(0, (2, 1.5)), clip_on=False)
            ax.text(0.12, 1.27, f"95% band, {a.a}", transform=tf, fontsize=6.3, va="center",
                    color="0.25")
            ax.add_patch(plt.Rectangle((0.88, 1.245), 0.1, 0.05, transform=tf, color="0.3",
                                       alpha=0.25, lw=0, clip_on=False))
            ax.text(1.0, 1.27, f"95% band, {a.b}  (purple: CDF, pink: inverse CDF)",
                    transform=tf, fontsize=6.3, va="center", color="0.25")
        sns.despine(ax=ax, offset=3)

    gc = gs[2].subgridspec(1, 2, wspace=0.35, width_ratios=[1, 1.3])
    ax = fig.add_subplot(gc[0])
    for stat, mk in (("bias", "o"), ("sd", "s")):
        for fit, col, dx in ((a.a, COL_A, -1.6), (a.b, COL_B, 1.6)):
            dx += -0.6 if stat == "bias" else 0.6
            c = coverage(sub, fit, stat)
            ax.plot(np.array(list(c)) + dx, list(c.values()), mk, ms=4, color=col,
                    mfc=col if stat == "bias" else "white", mew=1)
    ax.plot([45, 100], [45, 100], color="0.6", lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks([50, 80, 95]); ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xlim(42, 100); ax.set_ylim(15, 100)
    ax.set_xlabel("Nominal interval (%)")
    ax.set_ylabel("Participant cells inside (%)")
    for i, (lab, mfc) in enumerate((("Bid error", "0.3"), ("Bid SD", "white"))):
        ax.plot(0.06, 0.92 - i * 0.09, "o" if i == 0 else "s", transform=ax.transAxes,
                ms=4, color="0.3", mfc=mfc, mew=1)
        ax.text(0.11, 0.92 - i * 0.09, lab, transform=ax.transAxes, fontsize=6.3,
                va="center", color="0.25")
    ax.text(-0.3, 1.1, "c", transform=ax.transAxes, fontsize=9, fontweight="bold")
    sns.despine(ax=ax, offset=3)

    ax = fig.add_subplot(gc[1])
    g = gain.sort_values()
    ax.bar(range(len(g)), g.values, color=np.where(g.values > 0, COL_B, COL_A), width=0.75)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks([])
    ax.set_xlabel("Participants (sorted)")
    ax.set_ylabel("ΔELPD per participant")
    d = loo.loc[a.a] if loo.loc[a.b, "rank"] == 0 else loo.loc[a.b]
    sign = 1 if loo.loc[a.b, "rank"] == 0 else -1
    ax.set_title(f"{a.b} vs {a.a}: ΔELPD {sign * d.elpd_diff:+.0f} ± {d.dse:.0f} (PSIS-LOO)\n"
                 f"Better for {(gain > 0).sum()}/{len(gain)}", loc="left", fontsize=6.5,
                 color="0.3")
    ax.text(-0.17, 1.1, "d", transform=ax.transAxes, fontsize=9, fontweight="bold")
    sns.despine(ax=ax, offset=3, bottom=True)

    out = REPO / f"notes/figures/fig_ppc_{ind.name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    for fit in (a.a, a.b):
        print(fit, {s: {k: round(v, 1) for k, v in coverage(sub, fit, s).items()}
                    for s in ("bias", "sd")})
    print(loo[["elpd_loo", "elpd_diff", "dse", "p_loo"]].round(1).to_string())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
