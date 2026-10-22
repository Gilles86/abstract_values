"""Where do preferred values go when the mapping changes?

Per voxel, the session-shift aPRF (free preferred value per session) gives the
preferred value under the Misaligned (cdf) and the Aligned (inverse cdf)
mapping. Two reference codes predict where a voxel preferring v under one
mapping should prefer under the other:

  fixed value (lambda = 0)  the same CHF amount: the diagonal
  quantile    (lambda = 1)  the same quantile of the session's value
                            distribution = the same orientation
  mixture     (lambda)      x = (1 - lambda) * v + lambda * (2 + 40 q(v)),
                            the axis of the mix-<lambda> aPRFs

  a     the predictions for lambda = 0, .5, 1
  b–f   density of (cdf, inverse-cdf) preferred values per ROI, pooled over
        participants, with the two reference codes. Descriptive only: the
        per-voxel preferred values are noisy, and noise along the x axis
        pulls any curve fitted through these clouds towards the diagonal, so
        lambda itself is estimated from the cvR² profile of the mix-<lambda>
        fits (collect_mix_lambda.py), not from this plot
  g, h  preferred-value distributions per session vs the value density p(v)
        of that mapping, V1–3 and IPS0–5 (an efficient population puts more
        preferred values where values are dense)
  i, j  the same preferred values as orientations (each through its own
        session's mapping): a quantile / orientation-rank code overlaps here,
        a fixed-value code does not

Voxels: permutation-selected on the fixed-value (gaussian) model, which is
neutral to or against shifting; preferred values off the 2/42 CHF grid edges.
Input: notes/data/permnull_voxels.tsv.gz + permnull_gauss_voxels.tsv.gz.

Usage:
    python -m abstract_values.visualize.figure_rf_shift
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
from abstract_values.simulations.v1_ips_mapping import MAPPINGS, ORI_TABLE

REPO = Path(__file__).resolve().parents[2]
ROIS = [("BensonV1", "V1"), ("BensonV3", "V3"), ("IPS0", "IPS0"),
        ("IPS12", "IPS1–2"), ("IPS345", "IPS3–5")]
IPS = ["IPS0", "IPS12", "IPS345"]
SES_COL = {"cdf": "#3B5BA5", "inverse_cdf": "#C44E52"}
SES_NAME = {"cdf": "Misaligned (cdf)", "inverse_cdf": "Aligned (inverse cdf)"}
FIX_COL, Q_COL, FIT_COL = "0.35", "#D08C2E", "#5D8C3F"
THETA = np.linspace(0, 180, 2401)
LAMS = np.round(np.arange(-0.25, 1.2501, 0.05), 2)


def value(m, theta=THETA):
    return np.interp(theta, ORI_TABLE, MAPPINGS[m])


def axis(m, lam, theta=THETA):
    """Mixture axis x(v) along the stimulus range, for mapping m."""
    qn = (theta / 7.5 - 0.5) / 23            # mid-rank quantile, continuous
    return (1 - lam) * value(m, theta) + lam * (2 + 40 * qn)


def predict(v, src, dst, lam):
    """Preferred value under mapping dst of a voxel preferring v under src."""
    xs, xd = axis(src, lam), axis(dst, lam)
    x0 = np.interp(v, value(src), xs)
    o = np.argsort(xd)
    return np.interp(x0, xd[o], value(dst)[o])


def best_lambda(a, b):
    """Descriptive: lambda minimising squared prediction error in both directions."""
    err = [np.mean((predict(a, "cdf", "inverse_cdf", l) - b) ** 2)
           + np.mean((predict(b, "inverse_cdf", "cdf", l) - a) ** 2) for l in LAMS]
    return LAMS[int(np.argmin(err))], np.asarray(err)


def boot_lambda(d, n=500, seed=0):
    rng = np.random.default_rng(seed)
    subs = d.subject.unique()
    out = []
    for _ in range(n):
        pick = rng.choice(subs, len(subs))
        x = pd.concat([d[d.subject == s] for s in pick])
        out.append(best_lambda(x.mode_cdf.to_numpy(), x.mode_invcdf.to_numpy())[0])
    return np.percentile(out, [2.5, 97.5])


def load(smoothed):
    v = pd.read_csv(REPO / "notes/data/permnull_voxels.tsv.gz", sep="\t",
                    dtype={"subject": str})
    g = pd.read_csv(REPO / "notes/data/permnull_gauss_voxels.tsv.gz", sep="\t",
                    dtype={"subject": str})
    qc = [c for c in g if c.startswith("q_")]
    v = v.drop(columns=[c for c in v if c.startswith("q_")]).merge(
        g[["subject", "smoothed", "flat_index"] + qc],
        on=["subject", "smoothed", "flat_index"])
    v = v[(v.smoothed == smoothed) & v.mode_cdf.between(3, 41)
          & v.mode_invcdf.between(3, 41)]
    return v


def sel(v, rois):
    m = np.zeros(len(v), bool)
    for r in rois:
        m |= v[f"in_{r}"].astype(bool).to_numpy() & (v[f"q_{r}"].fillna(1) <= .05).to_numpy()
    return v[m]


def ref_lines(ax, lam_hat=None):
    g = np.linspace(2, 42, 200)
    ax.plot(g, g, color=FIX_COL, lw=1, ls="--")
    ax.plot(g, predict(g, "cdf", "inverse_cdf", 1.0), color=Q_COL, lw=1.4)
    if lam_hat is not None:
        ax.plot(g, predict(g, "cdf", "inverse_cdf", lam_hat), color=FIT_COL, lw=1.4)


def square(ax):
    ax.set_xlim(2, 42)
    ax.set_ylim(2, 42)
    ax.set_xticks([2, 12, 22, 32, 42])
    ax.set_yticks([2, 12, 22, 32, 42])
    ax.set_aspect("equal")


def main(out, smoothed):
    v = load(smoothed)
    fig = plt.figure(figsize=(7.25, 7.0))
    gt = fig.add_gridspec(1, 6, wspace=.5, left=.07, right=.99, top=.93, bottom=.68)
    gb = fig.add_gridspec(2, 2, wspace=.3, hspace=.6, left=.07, right=.99, top=.55, bottom=.07)
    axes = [fig.add_subplot(gt[0, i]) for i in range(6)]
    rows = []

    a = axes[0]
    g = np.linspace(2, 42, 200)
    for lam, c, lab in ((0, FIX_COL, "λ = 0: fixed value"), (.5, "0.6", "λ = .5"),
                        (1, Q_COL, "λ = 1: quantile")):
        a.plot(g, predict(g, "cdf", "inverse_cdf", lam), color=c,
               lw=1.4, ls="--" if lam == 0 else "-")
    square(a)
    a.set_title("Predictions", loc="left", fontweight="bold")
    a.set_xlabel("Preferred value,\nMisaligned (CHF)")
    a.set_ylabel("Preferred value, Aligned (CHF)")
    for i, (lab, c, ls) in enumerate((("Fixed value (λ = 0)", FIX_COL, "--"),
                                      ("λ = .5", "0.6", "-"),
                                      ("Quantile (λ = 1)", Q_COL, "-"))):
        y = .26 - i * .1
        a.plot([.42, .52], [y, y], transform=a.transAxes, color=c, ls=ls, lw=1.4)
        a.text(.55, y, lab, transform=a.transAxes, fontsize=5.5, va="center", color=c)

    for ax, (roi, lab) in zip(axes[1:], ROIS):
        d = sel(v, [roi])
        ax.hexbin(d.mode_cdf, d.mode_invcdf, gridsize=14, extent=(2, 42, 2, 42),
                  cmap="Greys", mincnt=1, linewidths=0, bins="log")
        ref_lines(ax)
        square(ax)
        ax.set_yticklabels([])
        ax.set_title(lab, loc="left", fontweight="bold")
        ax.text(.97, .03, f"{len(d)} voxels\n{d.subject.nunique()} people",
                transform=ax.transAxes, ha="right", fontsize=5.5, color="0.4")
        ax.set_xlabel("Misaligned (CHF)")
        rows.append(dict(roi=roi, n_vox=len(d),
                         n_sub=d.subject.nunique(),
                         r=stats.spearmanr(d.mode_cdf, d.mode_invcdf)[0]))

    # g: preferred-value distributions vs p(v), IPS0-5
    d = sel(v, IPS)
    for j, (ax, roi_lab, dd) in enumerate(((fig.add_subplot(gb[0, 0]), "V1–3",
                                            sel(v, ["BensonV1", "BensonV2", "BensonV3"])),
                                           (fig.add_subplot(gb[0, 1]), "IPS0–5", d))):
        grid = np.linspace(2, 42, 300)
        for m, col in (("cdf", "mode_cdf"), ("inverse_cdf", "mode_invcdf")):
            kde = stats.gaussian_kde(dd[col], bw_method=.12)(grid)
            ax.plot(grid, kde / kde.max(), color=SES_COL[m], lw=1.5)
            pv = np.histogram(value(m), bins=np.linspace(2, 42, 41), density=True)[0]
            ctr = np.linspace(2.5, 41.5, 40)
            pv = np.convolve(pv, np.ones(3) / 3, mode="same")
            ax.fill_between(ctr, 0, pv / pv.max(), color=SES_COL[m], alpha=.12, lw=0)
        ax.set_xlim(2, 42)
        ax.set_xticks([2, 12, 22, 32, 42])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Preferred value (CHF)")
        ax.set_title(f"{roi_lab}: where the preferred values sit", loc="left",
                     fontweight="bold")
        if j == 0:
            for i, m in enumerate(("cdf", "inverse_cdf")):
                ax.text(.36, .72 - i * .1, SES_NAME[m], transform=ax.transAxes,
                        color=SES_COL[m], fontsize=6, va="top")
            ax.plot([.55, .62], [.95, .95], transform=ax.transAxes, color="0.3", lw=1.5)
            ax.text(.64, .95, "Preferred values (density)", transform=ax.transAxes,
                    fontsize=6, va="center")
            ax.add_patch(plt.Rectangle((.55, .82), .07, .06, transform=ax.transAxes,
                                       color="0.3", alpha=.15, lw=0))
            ax.text(.64, .85, "Value density p(v)", transform=ax.transAxes,
                    fontsize=6, va="center")

    # i, j: the same voxels' preferred ORIENTATION per session (value -> the
    # orientation that carries it under that session's mapping). A quantile /
    # orientation-rank code puts both sessions on top of each other here; a
    # fixed-value code separates them.
    ori_grid = np.linspace(0, 180, 361)
    for j, (roi_lab, dd) in enumerate((("V1–3", sel(v, ["BensonV1", "BensonV2", "BensonV3"])),
                                       ("IPS0–5", d))):
        ax = fig.add_subplot(gb[1, j])
        for m, col in (("cdf", "mode_cdf"), ("inverse_cdf", "mode_invcdf")):
            th = np.interp(dd[col], MAPPINGS[m], ORI_TABLE)
            kde = stats.gaussian_kde(th, bw_method=.12)(ori_grid)
            ax.plot(ori_grid, kde / kde.max(), color=SES_COL[m], lw=1.5)
        a_ = np.interp(dd.mode_cdf, MAPPINGS["cdf"], ORI_TABLE)
        b_ = np.interp(dd.mode_invcdf, MAPPINGS["inverse_cdf"], ORI_TABLE)
        r_th = stats.spearmanr(a_, b_)[0]
        r_v = stats.spearmanr(dd.mode_cdf, dd.mode_invcdf)[0]
        ax.text(.98, .95, f"Cross-session ρ: orientation {r_th:.2f}, value {r_v:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6)
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Preferred orientation (°), via each session's mapping")
        ax.set_title(f"{roi_lab}: the same voxels in orientation space", loc="left",
                     fontweight="bold")

    for ax, l in zip(axes + fig.axes[6:], "abcdefghij"):
        ax.text(-.32 if ax is axes[0] else -.12, 1.12, l, transform=ax.transAxes,
                fontsize=9, fontweight="bold")
    res = pd.DataFrame(rows)
    res.to_csv(REPO / f"notes/data/rf_shift_lambda{'_smoothed' if smoothed else ''}.tsv",
               sep="\t", index=False)
    print(res.round(3).to_string(index=False))
    out = Path(out)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unsmoothed", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or str(REPO / f"notes/figures/fig_rf_shift{'' if not a.unsmoothed else '_unsmoothed'}.pdf")
    main(out, not a.unsmoothed)
