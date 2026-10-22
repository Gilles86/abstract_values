"""Model comparison figure: fixed value vs quantile vs wrapped quantile vs flexible.

Top row: what each aPRF variant predicts for one voxel under the two mappings
(response against presented CHF value; dots are the 23 stimuli).
Bottom row: leave-one-run-out cvR² − cvR²(null), mean over all voxels of each
ROI (no selection, n = 35):
  e  every model per ROI
  f  quantile − fixed value (same number of parameters: the fair test of
     whether the preferred value moves with the session's value distribution)
  g  wrapped − unwrapped quantile (circle vs line: orientation-like vs ordered)
  h  flexible − fixed value (one extra parameter: preferred value free per session)

Caveat in the panel titles: orientations are uniform and both mappings are
monotone, so the value quantile IS the orientation rank — 'quantile' is an
unwrapped Gaussian on orientation, 'quantile, wrapped' a von Mises on it.

Input: notes/data/quantile_models_per_subject.tsv (plot_quantile_models.py).

Usage:
    python -m abstract_values.visualize.figure_model_comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from abstract_values.visualize import qc_value_shift as q   # noqa: F401 house rcParams
from abstract_values.simulations.v1_ips_mapping import (
    MAPPINGS, STIM_ORI, THETA, mapping_fn)

REPO = Path(__file__).resolve().parents[2]
ROIS = [("BensonV1", "V1"), ("BensonV2", "V2"), ("BensonV3", "V3"),
        ("hV4", "hV4"), ("V3ab", "V3a/b"), ("IPS0", "IPS0"),
        ("IPS12", "IPS1–2"), ("IPS345", "IPS3–5")]
EARLY, IPS = ["BensonV1", "BensonV2", "BensonV3"], ["IPS0", "IPS12", "IPS345"]
SES_COL = {"cdf": "#3B5BA5", "inverse_cdf": "#C44E52"}
SES_NAME = {"cdf": "Misaligned (cdf)", "inverse_cdf": "Aligned (inverse cdf)"}
MODELS = [  # column, label, colour
    ("linear", "Linear ramp", "0.72"),
    ("gaussian", "Fixed value", "0.25"),
    ("quantile", "Quantile", "#D08C2E"),
    ("quantile_wrapped", "Wrapped", "#8172B2"),
    ("gauss_session_shift", "Flexible value", "#5D8C3F"),
]
COL = {m: c for m, _, c in MODELS}


def schematic(axes):
    """One example voxel per model, both mappings, response vs CHF."""
    th = THETA
    qn = (th - th[0]) / (th[-1] - th[0])          # orientation rank on [0, 1]
    qs = (STIM_ORI - th[0]) / (th[-1] - th[0])

    def gauss(x, mu, sd):
        return np.exp(-0.5 * ((x - mu) / sd) ** 2)

    def vm(x, mu, kappa):                          # circular on [0, 1)
        return np.exp(kappa * (np.cos(2 * np.pi * (x - mu)) - 1))

    specs = [
        ("gaussian", "Fixed value",
         "Same preferred value\nin both sessions", "4 parameters",
         lambda m, v, x: gauss(v, 22, 6)),
        ("quantile", "Quantile",
         "Same quantile, i.e. same\norientation rank", "4 parameters",
         lambda m, v, x: gauss(x, .32, .1)),
        ("quantile_wrapped", "Wrapped quantile",
         "As quantile, but 2 and\n42 CHF are neighbours", "4 parameters",
         lambda m, v, x: vm(x, .04, 4.)),
        ("gauss_session_shift", "Flexible value",
         "Preferred value free\nper session", "5 parameters",
         lambda m, v, x: gauss(v, {"cdf": 15, "inverse_cdf": 27}[m], 6)),
    ]
    for ax, (key, title, sub, npar, fn) in zip(axes, specs):
        for m in ("cdf", "inverse_cdf"):
            f, _ = mapping_fn(m)
            v, vs = f(th), np.interp(STIM_ORI, np.arange(0, 180.1, 7.5), MAPPINGS[m])
            ax.plot(v, fn(m, v, qn), color=SES_COL[m], lw=1.1)
            ax.plot(vs, fn(m, vs, qs), "o", ms=2.2, color=SES_COL[m], mew=0)
        ax.set_title(title, color=COL[key], fontweight="bold", loc="left", pad=11)
        ax.text(0, 1.03, npar, transform=ax.transAxes, fontsize=6, color="0.4")
        ax.text(.98, .97, sub, transform=ax.transAxes, ha="right", va="top",
                fontsize=6, color="0.3", linespacing=1.1)
        ax.set_xlim(0, 44)
        ax.set_xticks([2, 12, 22, 32, 42])
        ax.set_ylim(-.05, 1.35)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Presented value (CHF)")
    axes[0].set_ylabel("Predicted response")
    for i, m in enumerate(("cdf", "inverse_cdf")):
        axes[2].text(.52, .24 - i * .1, SES_NAME[m], transform=axes[2].transAxes,
                     color=SES_COL[m], fontsize=6, ha="center")


def roi_x(ax, labels):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axvspan(4.5, 7.6, color="0.94", lw=0, zorder=0)


def ladder(ax, d):
    """Every model, mean ± SEM over subjects (smoothed betas)."""
    x = np.arange(len(ROIS))
    ends = []
    for col, lab, c in MODELS:
        g = d.groupby("roi")[col]
        m = g.mean().reindex([r for r, _ in ROIS])
        se = g.sem().reindex([r for r, _ in ROIS])
        ax.errorbar(x, m, se, color=c, lw=1.1, marker="o", ms=2.5, capsize=0,
                    elinewidth=.8)
        ends.append([m.iloc[-1], lab, c])
    ends.sort(key=lambda e: e[0])                  # spread labels, min gap
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + .17)
    for y, lab, c in ends:
        ax.text(x[-1] + .35, y, lab, color=c, fontsize=6, va="center")
    ax.axhline(0, color="0.5", lw=.6, ls=":")
    roi_x(ax, [l for _, l in ROIS])
    ax.set_xlim(-.4, len(ROIS) + 1.8)
    ax.set_ylabel("cvR² gain over null (pp)")
    ax.set_title("All models, smoothed", loc="left", fontweight="bold")


def contrast(ax, ps, a, b, title, col):
    """Per-subject difference a − b per ROI; filled = smoothed, open = unsmoothed."""
    x = np.arange(len(ROIS))
    out = {}
    for sm, dx, mfc in ((True, -.13, col), (False, .13, "white")):
        d = ps[ps.smoothed == sm]
        diff = (d.set_index(["subject", "roi"])[a]
                - d.set_index(["subject", "roi"])[b]).rename("diff").reset_index()
        for i, (roi, _) in enumerate(ROIS):
            y = diff.loc[diff.roi == roi, "diff"].dropna()
            ax.plot(np.full(len(y), i + dx) + np.random.default_rng(i).uniform(
                -.05, .05, len(y)), y, ".", ms=1.6, color=col, alpha=.25, mew=0)
            t = stats.ttest_1samp(y, 0)
            ax.errorbar(i + dx, y.mean(), y.sem(), color=col, marker="o", ms=3,
                        mfc=mfc, mew=.8, capsize=0, elinewidth=1)
            if sm:
                out[roi] = (y.mean(), t.pvalue)
        piv = diff.pivot(index="subject", columns="roi", values="diff")
        out[("group", sm)] = stats.ttest_rel(piv[IPS].mean(1), piv[EARLY].mean(1))
    ax.axhline(0, color="0.5", lw=.6, ls=":")
    roi_x(ax, [l for _, l in ROIS])
    ax.set_title(title, loc="left", fontweight="bold", color=col)
    ax.set_ylabel(f"Difference in cvR² (pp)")
    return out


def stars(ax, res, y):
    for i, (roi, _) in enumerate(ROIS):
        p = res[roi][1]
        if p < .05:
            ax.text(i - .13, y, "**" if p < .001 else "*", ha="center",
                    fontsize=7, transform=ax.get_xaxis_transform())


def main(tsv, out):
    ps = pd.read_csv(tsv, sep="\t", dtype={"subject": str})
    ps = ps[(ps.selection == "All voxels") & ps.roi.isin([r for r, _ in ROIS])]

    fig = plt.figure(figsize=(7.25, 5.2))
    gt = fig.add_gridspec(1, 4, wspace=.35, left=.07, right=.97, top=.9,
                          bottom=.66)
    gb = fig.add_gridspec(1, 4, width_ratios=[1.45, 1, 1, 1], wspace=.5,
                          left=.07, right=.99, top=.5, bottom=.1)
    top = [fig.add_subplot(gt[0, i]) for i in range(4)]
    schematic(top)
    bot = [fig.add_subplot(gb[0, i]) for i in range(4)]
    ladder(bot[0], ps[ps.smoothed])

    res = {}
    for ax, (a, b, title, col, ylim) in zip(bot[1:], [
            ("quantile", "gaussian", "Quantile − fixed value", COL["quantile"],
             (-1, 1.5)),
            ("quantile_wrapped", "quantile", "Wrapped − unwrapped",
             COL["quantile_wrapped"], (-.9, .9)),
            ("gauss_session_shift", "gaussian", "Flexible − fixed value",
             COL["gauss_session_shift"], (-2, 1))]):
        r = contrast(ax, ps, a, b, title, col)
        ax.set_ylim(*ylim)
        stars(ax, r, .93)
        res[title] = r
    for i, (lab, mfc) in enumerate((("Smoothed", COL["quantile"]),
                                    ("Unsmoothed", "white"))):
        y = .13 - i * .08
        bot[1].plot([.06], [y], "o", ms=3, mfc=mfc, mew=.8, color=COL["quantile"],
                    transform=bot[1].transAxes, clip_on=False)
        bot[1].text(.11, y, lab, transform=bot[1].transAxes, fontsize=6,
                    color="0.25", va="center")
    g = res["Wrapped − unwrapped"][("group", True)]
    bot[2].text(.03, .06, f"IPS vs V1–3: t(34) = {g.statistic:.1f}, "
                + ("p < .001" if g.pvalue < .001 else f"p = {g.pvalue:.2g}"),
                transform=bot[2].transAxes, fontsize=6)
    bot[2].text(.97, .80, "Circle fits better", transform=bot[2].transAxes,
                ha="right", fontsize=6, color="0.4")
    bot[2].text(.97, .18, "Line fits better", transform=bot[2].transAxes,
                ha="right", fontsize=6, color="0.4")
    bot[3].text(.97, .06, "Extra parameter\ncosts out of sample",
                transform=bot[3].transAxes, ha="right", fontsize=6, color="0.4")

    for ax, l in zip(top + bot, "abcdefgh"):
        ax.text(-.12 if ax in top else (-.22 if l == "e" else -.32),
                1.2 if ax in top else 1.08, l,
                transform=ax.transAxes, fontsize=9, fontweight="bold")
    for t, r in res.items():
        print(t, {k: tuple(np.round(v, 4)) if isinstance(v, tuple) else
                  (round(v.statistic, 2), float(f"{v.pvalue:.3g}"))
                  for k, v in r.items()})
    out = Path(out)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", default=str(REPO / "notes/data/quantile_models_per_subject.tsv"))
    ap.add_argument("--out", default=str(REPO / "notes/figures/fig_model_comparison.pdf"))
    a = ap.parse_args()
    main(a.tsv, a.out)
