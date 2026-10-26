"""Model comparison: value code vs quantile code, position and width separately.

a: the four codes for one voxel (MixPositionWidthGaussianPRF corners).
b: leave-one-run-out cvR^2 of each code relative to the value code, per ROI; one dot
   per participant, mean and 95% CI; stars: paired t-test against the value code.
c: share of participants whose best code is each of the four, per ROI.
d: recovery: betas simulated from each code (each voxel's real Misaligned preferred
   value, width and signal level; noise twice the real residual SD, which matches the
   observed cvR^2 level), refitted the same way; which code wins (participants x ROIs),
   with the observed data underneath.

Input: notes/data/mix_lambda_width.tsv (fit_mix_lambda.py --model width, per ROI).

    python -m abstract_values.visualize.figure_quantile_model_comparison
    -> notes/figures/fig_quantile_model_comparison.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from abstract_values.visualize import qc_value_shift as q   # noqa: F401 house rcParams
from abstract_values.visualize.figure_mix_lambda import CORNERS, NAME, schematic

REPO = Path(__file__).resolve().parents[2]
ROIS = ["V1", "hV4", "V3ab", "IPS0", "IPS12", "IPS345"]
ALT = ["1_0", "0_1", "1_1"]
SHORT = {"0_0": "Value", "1_0": "Position", "0_1": "Width", "1_1": "Quantile"}


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def main(out=REPO / "notes/figures/fig_quantile_model_comparison.pdf"):
    d = pd.read_csv(REPO / "notes/data/mix_lambda_width.tsv", sep="\t", dtype={"subject": str})
    for k in ALT:
        d[f"d{k}"] = 1e3 * (d[f"cvr2_fixed{k}"] - d["cvr2_fixed0_0"])
    cor = [f"cvr2_fixed{k}" for k in CORNERS]
    d["win"] = d[cor].idxmax(axis=1).str.replace("cvr2_fixed", "")
    o = d[d.cond == "observed"]
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(7.25, 6.4))
    top = fig.add_gridspec(1, 4, left=.05, right=.99, top=.92, bottom=.78, wspace=.15)
    schematic([fig.add_subplot(top[i]) for i in range(4)])
    fig.text(.01, .975, "a  Four codes: does tuning keep its value, or its quantile, when the "
             "value distribution changes?", fontweight="bold", fontsize=8)

    mid = fig.add_gridspec(1, 2, left=.08, right=.99, top=.66, bottom=.42, wspace=.22,
                           width_ratios=[3, 1])
    ax = fig.add_subplot(mid[0])
    for i, r in enumerate(ROIS):
        for j, k in enumerate(ALT):
            x = o[o.roi == r][f"d{k}"].to_numpy()
            col = CORNERS[k][1]
            xp = i + (j - 1) * .26
            ax.scatter(xp + rng.uniform(-.07, .07, len(x)), x, s=3, color=col, alpha=.35, lw=0)
            m, ci = x.mean(), stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
            ax.plot([xp, xp], [m - ci, m + ci], color=col, lw=1.4, solid_capstyle="butt")
            ax.plot(xp, m, "o", color=col, ms=4, mec="white", mew=.5, zorder=3)
            ax.text(xp, 8.3, stars(stats.ttest_1samp(x, 0).pvalue), ha="center", fontsize=5.5,
                    color=col if k != "0_1" else "0.4")
    ax.axhline(0, color=CORNERS["0_0"][1], lw=.8)
    ax.text(len(ROIS) + .35, .25, "Value code", color=CORNERS["0_0"][1], fontsize=6,
            ha="right", va="bottom")
    ax.axvspan(2.5, 5.5, color="0.95", lw=0, zorder=-2)
    ax.set_xticks(range(len(ROIS)), [NAME[r] for r in ROIS])
    ax.set_xlim(-.6, len(ROIS) + .4)
    ax.set_ylim(-10.5, 9)
    ax.set_ylabel("Held-out cvR² vs value code\n(×10⁻³, one dot per participant)")
    for j, k in enumerate(ALT):
        ax.text(.01 + .27 * j, .025, ("Position follows quantile", "Width follows quantile",
                                      "Quantile code (both)")[j],
                color=CORNERS[k][1] if k != "0_1" else "0.4", fontsize=6.3,
                transform=ax.transAxes)
    ax.set_title("b  Which code predicts held-out runs best (n = 35)", loc="left",
                 fontweight="bold")

    ax = fig.add_subplot(mid[1])
    w = o.groupby("roi").win.value_counts(normalize=True).unstack().reindex(
        index=ROIS, columns=list(CORNERS)).fillna(0)
    for i, r in enumerate(ROIS):
        left = 0
        for k, (_, col) in CORNERS.items():
            ax.barh(i, w.loc[r, k], left=left, color=col, height=.72, lw=0)
            left += w.loc[r, k]
    ax.set_yticks(range(len(ROIS)), [NAME[r] for r in ROIS])
    ax.set_ylim(len(ROIS) - .5, -.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .5, 1])
    ax.set_xlabel("Share of participants")
    ax.set_title("c  Best code", loc="left", fontweight="bold")

    bot = fig.add_gridspec(1, 2, left=.17, right=.99, top=.3, bottom=.07, wspace=.5,
                           width_ratios=[1.25, 1])
    ax = fig.add_subplot(bot[0])
    rows = [(f"sim-corner{k}-noise2", f"Simulated {SHORT[k].lower()} code") for k in CORNERS]
    rows += [("observed", "Observed")]
    M = np.full((len(rows), 4), np.nan)
    for i, (c, _) in enumerate(rows):
        x = d[d.cond == c].win.value_counts(normalize=True)
        M[i] = [x.get(k, 0) for k in CORNERS]
    ax.imshow(M, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(4):
            ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] > .5 else "0.2")
    ax.axhline(3.5, color="white", lw=3)
    ax.set_xticks(range(4), [SHORT[k] for k in CORNERS])
    for t, k in zip(ax.get_xticklabels(), CORNERS):
        t.set_color(CORNERS[k][1] if k != "0_1" else "0.4")
    ax.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax.set_xlabel("Winning code (% of participants × ROIs)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("d  The comparison recovers the true code", loc="left", fontweight="bold")

    ax = fig.add_subplot(bot[1])
    ips = d[d.roi.isin(["IPS0", "IPS12", "IPS345"])].groupby(["cond", "subject"])[
        [f"d{k}" for k in ALT]].mean().reset_index()
    for i, (c, lab) in enumerate(rows):
        for j, k in enumerate(ALT):
            x = ips[ips.cond == c][f"d{k}"]
            m, se = x.mean(), x.std() / np.sqrt(len(x))
            y = i + (j - 1) * .24
            col = CORNERS[k][1]
            ax.plot([m - se, m + se], [y, y], color=col, lw=1.2)
            ax.plot(m, y, "o", color=col, ms=3.2, mew=0)
    ax.axvline(0, color=CORNERS["0_0"][1], lw=.8)
    ax.axhline(3.5, color="0.85", lw=.6)
    ax.set_yticks(range(len(rows)), [SHORT[r[0][10:13]] if r[0] != "observed" else "Observed"
                                     for r in rows])
    ax.set_ylim(len(rows) - .5, -.6)
    ax.set_xlabel("IPS cvR² vs value code (×10⁻³)")
    ax.set_title("e  IPS: simulated vs observed", loc="left", fontweight="bold")

    for a in fig.axes[4:]:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    out = Path(out)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
