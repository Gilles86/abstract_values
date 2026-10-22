"""Explainer: orientation-wired vs value-wired coupling, and the two-component fit.

Pure schematic (simulated voxels, the analysis' own template functions):

  a  One IPS voxel, preferred value 16 CHF: which V1 orientation carries 16 CHF
     differs between the mappings.
  b  Orientation-wired (quantile) voxel: it listens to the same V1 orientations
     in both sessions -> identical coupling profiles.
  c  Value-wired voxel: the coupling peak moves to wherever 16 CHF sits.
  d  The current coupling score (this-session minus other-session template fit)
     for orientation-wired, value-wired and 50/50 voxels: orientation wiring
     scores 0, and a quantile label would score 0 by construction.
  e  The new analysis on a 50/50 voxel: both sessions' profiles fitted at once
     with a fixed orientation template (weight a) and a moving value template
     (weight b).
  f  Recovered (a, b) for the three voxel types.
  g  What the per-voxel test asks: does b drop where the tuning prefers the
     quantile model over the fixed-value model?

Usage:
    python -m abstract_values.visualize.figure_coupling_components
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from abstract_values.visualize import qc_value_shift as q   # noqa: F401 house rcParams
from abstract_values.connective_fields.gates import colcorr, load_mappings
from abstract_values.connective_fields.test_coupling import (
    quantile_prediction, two_component_fit, unit, value_prediction)

REPO = Path(__file__).resolve().parents[2]
SES = ("cdf", "inverse_cdf")
SES_COL = {"cdf": "#3B5BA5", "inverse_cdf": "#C44E52"}
SES_NAME = {"cdf": "Misaligned (cdf)", "inverse_cdf": "Aligned (inverse cdf)"}
ORI_COL, VAL_COL = "#8172B2", "#D08C2E"
GRID = np.arange(0, 180, 2.5)
V0, FWHM, MU0, SD0 = 16.0, 8.0, 63.0, 15.0


def chan_values(ori, maps, m):
    return np.interp(GRID, ori, maps[m])


def templates(ori, maps, mode=V0, fwhm=FWHM, mu=MU0, sd=SD0):
    q_t = quantile_prediction(np.atleast_1d(mu), np.atleast_1d(sd), GRID)
    v_t = {m: value_prediction(np.atleast_1d(mode), np.atleast_1d(fwhm),
                               chan_values(ori, maps, m)) for m in SES}
    return q_t, v_t


def cf(w_ori, q_t, v_t, m, noise=0.0, rng=None):
    x = w_ori * unit(q_t) + (1 - w_ori) * unit(v_t[m])
    if noise:
        x = x + noise * rng.standard_normal(x.shape)
    return x


def old_score(cfs, v_t):
    return np.mean([colcorr(cfs[m], v_t[m]) - colcorr(cfs[m], v_t[o])
                    for m, o in zip(SES, SES[::-1])], 0)


def profile(ax, curves, title, sub):
    for m, y in curves.items():
        ax.plot(GRID, y, color=SES_COL[m], lw=1.3)
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("V1 orientation channel (°)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(title, loc="left", fontweight="bold", pad=13)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=6, color="0.4")


def main(out):
    ori, maps = load_mappings()
    rng = np.random.default_rng(1)
    q_t, v_t = templates(ori, maps)
    th = {m: float(np.interp(V0, maps[m], ori)) for m in SES}

    fig = plt.figure(figsize=(7.25, 5.3))
    g1 = fig.add_gridspec(1, 4, wspace=.45, left=.06, right=.98, top=.9, bottom=.6)
    g2 = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=.45, left=.06,
                          right=.98, top=.4, bottom=.1)
    ax = [fig.add_subplot(g1[0, i]) for i in range(4)] + \
         [fig.add_subplot(g2[0, i]) for i in range(3)]

    # a: mapping and the two orientations that carry 16 CHF
    a = ax[0]
    for m in SES:
        a.plot(ori, maps[m], color=SES_COL[m], lw=1.3)
        a.plot([th[m], th[m]], [0, V0], color=SES_COL[m], lw=.8, ls=":")
        a.plot(th[m], V0, "o", ms=3.5, color=SES_COL[m])
    a.axhline(V0, color="0.3", lw=.8)
    a.text(178, V0 - 3, "IPS voxel\nprefers 16 CHF", ha="right", va="top", fontsize=6)
    a.set_xlim(0, 180)
    a.set_ylim(0, 44)
    a.set_xticks([0, 45, 90, 135, 180])
    a.set_yticks([2, 12, 22, 32, 42])
    a.set_xlabel("Orientation (°)")
    a.set_ylabel("Value (CHF)")
    a.set_title("Two mappings", loc="left", fontweight="bold", pad=13)
    a.text(0, 1.03, f"16 CHF = {th['cdf']:.0f}° vs {th['inverse_cdf']:.0f}°",
           transform=a.transAxes, fontsize=6, color="0.4")
    for i, m in enumerate(SES):
        a.text(.03, .93 - i * .09, SES_NAME[m], transform=a.transAxes,
               color=SES_COL[m], fontsize=6)

    # b, c: coupling profiles of pure voxel types
    profile(ax[1], {m: cf(1, q_t, v_t, m)[:, 0] for m in SES},
            "Orientation-wired", "Same V1 input in both sessions")
    ax[1].title.set_color(ORI_COL)
    profile(ax[2], {m: cf(0, q_t, v_t, m)[:, 0] for m in SES},
            "Value-wired", "Input follows the 16-CHF orientation")
    ax[2].title.set_color(VAL_COL)
    ax[1].set_ylabel("Coupling with IPS voxel")

    # d: current score
    d = ax[3]
    n = 400
    cases = [("Orientation-\nwired", 1.0, ORI_COL), ("50/50", 0.5, "0.45"),
             ("Value-\nwired", 0.0, VAL_COL)]
    for i, (lab, w, c) in enumerate(cases):
        cfs = {m: cf(w, np.repeat(q_t, n, 1), {k: np.repeat(v, n, 1) for k, v in v_t.items()},
                     m, noise=.08, rng=rng) for m in SES}
        s = old_score(cfs, v_t)
        d.bar(i, s.mean(), color=c, width=.65)
    d.axhline(0, color="0.3", lw=.8)
    d.set_xticks(range(3))
    d.set_xticklabels([c[0] for c in cases], fontsize=6)
    d.set_ylabel("Current coupling score")
    d.set_title("Current score", loc="left", fontweight="bold", pad=13)
    d.text(0, 1.03, "Sees only the moving part", transform=d.transAxes, fontsize=6,
           color="0.4")
    d.annotate("0 by construction,\nas with quantile\nlabels", (0, 0.005),
               (0.03, .66), textcoords="axes fraction", fontsize=6, color=ORI_COL,
               arrowprops=dict(arrowstyle="-", color=ORI_COL, lw=.6))

    # e: decomposition of a 50/50 voxel
    e = ax[4]
    off = 180 + 25
    cfs = {m: cf(.5, q_t, v_t, m, noise=.012, rng=rng) for m in SES}
    a_hat, b_hat = two_component_fit([dict(cf=cfs[m]) for m in SES], q_t,
                                     [v_t[m] for m in SES])
    for j, m in enumerate(SES):
        x = GRID + j * off
        e.plot(x, cfs[m][:, 0], color=SES_COL[m], lw=1.6)
        e.plot(x, a_hat[0] * unit(q_t)[:, 0], color=ORI_COL, lw=1, ls="--")
        e.plot(x, b_hat[0] * unit(v_t[m])[:, 0], color=VAL_COL, lw=1, ls="--")
        e.text(x.mean(), 1.0, SES_NAME[m], transform=e.get_xaxis_transform(),
               ha="center", color=SES_COL[m], fontsize=6)
    e.set_xticks([0, 90, 180, off, off + 90, off + 180])
    e.set_xticklabels(["0", "90", "180"] * 2)
    e.set_xlabel("V1 orientation channel (°), both sessions stacked")
    e.set_yticks([])
    e.spines["left"].set_visible(False)
    e.set_title("New: fit both components at once", loc="left", fontweight="bold",
                pad=12)
    lo, hi = e.get_ylim()
    e.set_ylim(lo - .55 * (hi - lo), hi)
    for i, (txt, c) in enumerate((
            (f"a × fixed orientation template (a = {a_hat[0]:.2f})", ORI_COL),
            (f"b × template moving with the mapping (b = {b_hat[0]:.2f})", VAL_COL),
            ("Coupling profile (data)", "0.3"))):
        y = .2 - i * .08
        e.plot([.03, .09], [y, y], transform=e.transAxes, color=c,
               lw=1 if i < 2 else 1.6, ls="--" if i < 2 else "-")
        e.text(.11, y, txt, transform=e.transAxes, va="center", fontsize=6, color=c)

    # f: recovered weights
    f = ax[5]
    for lab, w, c in cases:
        cfs = {m: cf(w, np.repeat(q_t, n, 1), {k: np.repeat(v, n, 1) for k, v in v_t.items()},
                     m, noise=.08, rng=rng) for m in SES}
        ah, bh = two_component_fit([dict(cf=cfs[m]) for m in SES], np.repeat(q_t, n, 1),
                                   [np.repeat(v_t[m], n, 1) for m in SES])
        f.plot(ah.mean(), bh.mean(), "o", ms=5, color=c)
        f.text(ah.mean() + .04, bh.mean() + .04, lab.replace("-\n", "-"), fontsize=6,
               color=c)
    f.plot([1, 0], [0, 1], color="0.8", lw=.8, zorder=0)
    f.set_xlim(-.1, 1.35)
    f.set_ylim(-.1, 1.25)
    f.set_xticks([0, .5, 1])
    f.set_yticks([0, .5, 1])
    f.set_xlabel("a: orientation-wired weight")
    f.set_ylabel("b: value-wired weight")
    f.set_title("A mixture estimate", loc="left", fontweight="bold")

    # g: the per-voxel question
    gx = ax[6]
    x = np.linspace(-1, 1, 60)
    gx.plot(x, .5 - .3 * x, color="0.25", lw=1.3)
    gx.plot(x, np.full_like(x, .5), color="0.25", lw=1.3, ls="--")
    gx.text(.36, .86, "Two voxel types", transform=gx.transAxes, fontsize=6)
    gx.text(.98, .53, "Tuning ≠ coupling", transform=gx.transAxes, ha="right",
            va="bottom", fontsize=6)
    gx.axvline(0, color="0.8", lw=.8, zorder=0)
    gx.set_xticks([-1, 0, 1])
    gx.set_xticklabels(["Fixed\nvalue", "0", "Quantile"], fontsize=6)
    gx.set_yticks([])
    gx.set_xlabel("Tuning preference (Δ cvR²)")
    gx.set_ylabel("b: value-wired weight")
    gx.set_title("Per-voxel test", loc="left", fontweight="bold")

    for axx, l in zip(ax, "abcdefg"):
        axx.text(-.18 if axx is not ax[4] else -.1, 1.17, l, transform=axx.transAxes,
                 fontsize=9, fontweight="bold")
    out = Path(out)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "notes/figures/fig_coupling_components.pdf"))
    a = ap.parse_args()
    main(a.out)
