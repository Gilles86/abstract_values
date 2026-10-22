"""Explainer and results for the tuning x V1-connectivity model.

Two figures (``connective_fields/fit_tuning_connectivity.py``):

fig_tuning_connectivity_explainer.pdf (schematic, the analysis' own axes)
  a  The two mappings.
  b  lambda = 0: one tuning curve on value, the same in both sessions.
  c  lambda = 1: a tuning curve on orientation rank (quantile); in value
     space its peak moves with the session's value distribution.
  d  The lambda continuum: preferred value per session for one voxel.
  e  V1 -> orientation channels -> trial-specific residual (one trial).
  f  The connectivity profile h over V1 channels, orientation-wired
     (lambda' = 1) vs value-wired (lambda' = 0).
  g  The three models and the shuffle control.

fig_tuning_connectivity.pdf (results, n = 35, rows unsmoothed / smoothed)
  a  Coding: cvR^2(lambda) - cvR^2(lambda = 0).
  b  Connectivity alone: gain over the within-orientation V1 shuffle per lambda'.
  c  Joint model at lambda = 1: connectivity gain over shuffle per lambda'.
  d  Joint gain over (lambda, lambda'), mean over IPS0-5.

Usage:
    python -m abstract_values.visualize.figure_tuning_connectivity
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.visualize import qc_value_shift as q   # noqa: F401 house rcParams
from abstract_values.connective_fields.gates import load_mappings
from abstract_values.connective_fields.test_coupling import IEM_GRID

REPO = Path(__file__).resolve().parents[2]
SES = ("cdf", "inverse_cdf")
SES_COL = {"cdf": "#3B5BA5", "inverse_cdf": "#C44E52"}
SES_NAME = {"cdf": "Misaligned (cdf)", "inverse_cdf": "Aligned (inverse cdf)"}
ORI_COL, VAL_COL = "#8172B2", "#D08C2E"
ROIS = ["IPS0", "IPS12", "IPS345", "V3ab", "hV4"]
ROI_NAME = {"IPS0": "IPS0", "IPS12": "IPS1–2", "IPS345": "IPS3–5", "V3ab": "V3a/b",
            "hV4": "hV4"}
_m = sns.color_palette("mako", as_cmap=True)
ROI_COL = {"IPS0": _m(.25), "IPS12": _m(.45), "IPS345": _m(.65), "V3ab": "0.55",
           "hV4": "0.75"}
V = np.linspace(2, 42, 401)
X0, SIG = 16.0, 4.0


def x_lam(v, ori, val, lam):
    """Mix axis for value v under one mapping (continuous mid-rank quantile)."""
    th = np.interp(v, val, ori)
    qq = (th / 7.5 - 0.5) / 23
    return (1 - lam) * v + lam * (2 + 40 * qq)


def gauss(x, mu=X0, s=SIG):
    return np.exp(-.5 * ((x - mu) / s) ** 2)


def title(ax, t, sub=None, col="k"):
    ax.set_title(t, loc="left", fontweight="bold", pad=13 if sub else 4, color=col)
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=6, color="0.4")


def value_axis(ax):
    ax.set_xlim(2, 42)
    ax.set_xticks([2, 12, 22, 32, 42])
    ax.set_xlabel("Value (CHF)")


def ori_axis(ax, label="V1 orientation channel (°)"):
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel(label)


def no_y(ax):
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


# ── explainer ────────────────────────────────────────────────────────────────

def explainer(out):
    ori, maps = load_mappings()
    fig = plt.figure(figsize=(7.25, 5.0))
    g1 = fig.add_gridspec(1, 4, wspace=.42, left=.06, right=.98, top=.9, bottom=.6)
    g2 = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=.38, left=.06,
                          right=.98, top=.4, bottom=.1)
    ax = [fig.add_subplot(g1[0, i]) for i in range(4)] + \
         [fig.add_subplot(g2[0, i]) for i in range(3)]

    # a: mappings
    a = ax[0]
    for m in SES:
        a.plot(ori, maps[m], color=SES_COL[m], lw=1.3)
    ori_axis(a, "Orientation (°)")
    a.set_ylim(0, 44)
    a.set_yticks([2, 12, 22, 32, 42])
    a.set_ylabel("Value (CHF)")
    title(a, "Two mappings", "One per session, both monotone")
    for i, m in enumerate(SES):
        a.text(.03, .93 - i * .09, SES_NAME[m], transform=a.transAxes,
               color=SES_COL[m], fontsize=6)

    # b, c: one voxel at lambda = 0 and lambda = 1, in value space
    for axx, lam, col, t, sub in ((ax[1], 0., VAL_COL, "λ = 0: fixed value",
                                   "Same preferred value in both sessions"),
                                  (ax[2], 1., ORI_COL, "λ = 1: quantile",
                                   "Preferred value follows p(v)")):
        for m in SES:
            y = gauss(x_lam(V, ori, maps[m], lam))
            axx.plot(V, y, color=SES_COL[m], lw=1.3)
            axx.plot(V[np.argmax(y)], 1.08, "v", ms=3.5, color=SES_COL[m],
                     clip_on=False)
            # where the session's 23 values sit
            axx.plot(maps[m][1:-1], np.full(23, -.08 - .07 * SES.index(m)), "|",
                     ms=3, color=SES_COL[m], mew=.6, clip_on=False)
        value_axis(axx)
        axx.set_ylim(-.24, 1.12)
        no_y(axx)
        title(axx, t, sub, col)
    ax[1].set_ylabel("Response of one voxel")
    ax[1].text(41, .05, "Values shown", ha="right", va="bottom", fontsize=5.5,
               color="0.4")

    # d: continuum
    d = ax[3]
    lams = np.linspace(-.5, 1.5, 81)
    for m in SES:
        pv = [V[np.argmax(gauss(x_lam(V, ori, maps[m], l)))] for l in lams]
        d.plot(lams, pv, color=SES_COL[m], lw=1.3)
    for l, lab in ((0, "Fixed\nvalue"), (1, "Quantile")):
        d.axvline(l, color="0.85", lw=.8, zorder=0)
    d.set_xticks([-.5, 0, .5, 1, 1.5])
    d.set_xlim(-.5, 1.5)
    d.set_ylim(2, 42)
    d.set_yticks([2, 12, 22, 32, 42])
    d.set_xlabel("λ")
    d.set_ylabel("Preferred value (CHF)")
    title(d, "One axis, λ", "x = (1 − λ)·value + λ·quantile")
    d.text(1.48, 39, "Overshoot", ha="right", fontsize=6, color="0.4")
    d.text(-.47, 39, "Opposite", ha="left", fontsize=6, color="0.4")

    # e: V1 channel response on one trial, stimulus part vs trial-specific part
    e = ax[4]
    rng = np.random.default_rng(3)
    phi = np.linspace(0, 180, 181)
    stim = 60.
    mean = np.exp(4 * (np.cos(np.deg2rad(2 * (phi - stim))) - 1))
    dev = .35 * np.exp(6 * (np.cos(np.deg2rad(2 * (phi - 105))) - 1)) \
        + .08 * np.convolve(rng.standard_normal(181 + 30), np.ones(31) / 5.5, "valid")
    e.fill_between(phi, mean, color="0.85", lw=0)
    e.plot(phi, mean + dev, color="0.2", lw=1.2)
    e.plot(phi, dev - .55, color="#2A9D8F", lw=1.2)
    e.axhline(-.55, color="0.8", lw=.6, zorder=0)
    e.text(2, .78, "Orientation\nmean", fontsize=6, color="0.45",
           va="top")
    e.text(178, .55, "This trial", fontsize=6, color="0.2", ha="right")
    e.text(2, -.2, "Trial-specific part,\nc_resid (used)", fontsize=6,
           color="#2A9D8F", ha="left")
    ori_axis(e)
    no_y(e)
    e.set_ylabel("V1 channel response")
    title(e, "V1 input", "Linear IEM inversion, stimulus removed")

    # f: connectivity profiles h over channels
    f = ax[5]
    ch = IEM_GRID
    qq = (ch / 7.5 - .5) / 23
    off = 1.35
    for k, (lam, col, lab) in enumerate(((1., ORI_COL, "λ′ = 1: orientation-wired"),
                                        (0., VAL_COL, "λ′ = 0: value-wired"))):
        for m in SES:
            xc = (1 - lam) * np.interp(ch, ori, maps[m]) + lam * (2 + 40 * qq)
            f.plot(ch, gauss(xc) - k * off, color=SES_COL[m], lw=1.2)
        f.text(178, .75 - k * off, lab.replace(": ", ":\n"), fontsize=6, color=col,
               ha="right", va="top")
    ori_axis(f)
    no_y(f)
    f.set_ylim(-off - .1, 1.3)
    f.set_ylabel("Weight h on V1 channel")
    title(f, "Connectivity profile", "Same voxel, both sessions")

    # g: the models
    gx = ax[6]
    gx.axis("off")
    title(gx, "Three models, leave-one-run-out cvR²")
    rows = [("Coding", "y = b + a·g(x_λ(v))", "Which λ?"),
            ("Connectivity", "r = b + β·(h_λ′ · c_resid)", "IPS residual from V1 residual"),
            ("Joint", "y = b + a·g(x_λ(v)) + β·(h_λ′ · c_resid)",
             "g and h share mode and width"),
            ("Control", "c_resid shuffled within orientation",
             "Keeps stimulus, breaks V1 trial noise")]
    for i, (n, eq, note) in enumerate(rows):
        y = .9 - i * .25
        gx.text(0, y, n, fontsize=6.5, fontweight="bold", transform=gx.transAxes)
        gx.text(.27, y, eq, fontsize=6.5, transform=gx.transAxes)
        gx.text(.27, y - .09, note, fontsize=6, color="0.4", transform=gx.transAxes)

    for axx, l in zip(ax, "abcdefg"):
        axx.text(-.2 if axx is not ax[6] else -.05, 1.17, l, transform=axx.transAxes,
                 fontsize=9, fontweight="bold")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(f"Wrote {out}")


# ── results ──────────────────────────────────────────────────────────────────

def load(tsv):
    d = pd.read_csv(tsv, sep="\t", dtype={"subject": str})
    d["smoothed"] = d["smoothed"].astype(str) == "True"
    return d


def sem_line(ax, piv, col, label=None, lw=1.3):
    """piv: subjects x levels. Mean +- SEM band, label at the right end."""
    m, s = piv.mean(), piv.sem()
    x = piv.columns.to_numpy(float)
    ax.fill_between(x, m - s, m + s, color=col, alpha=.2, lw=0)
    ax.plot(x, m, color=col, lw=lw)
    return m


def label_ends(ax, ends, x):
    """Direct labels at the right end, nudged apart."""
    ys = sorted(ends.items(), key=lambda kv: kv[1][0])
    lo, hi = ax.get_ylim()
    gap = .075 * (hi - lo)
    placed = []
    for roi, (y, col) in ys:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.text(x, y, ROI_NAME[roi], color=col, fontsize=6, va="center")


def results(tsv, out):
    d = load(tsv)
    L = np.sort(d["lam"].unique())
    fig, ax = plt.subplots(2, 4, figsize=(7.25, 4.4),
                           gridspec_kw=dict(width_ratios=[1, 1, 1, .9], wspace=.85,
                                            hspace=.75, left=.07, right=.93, top=.9,
                                            bottom=.1))
    for r, sm in enumerate((False, True)):
        dd = d[d.smoothed == sm]
        tag = "Smoothed" if sm else "Unsmoothed"
        # a: coding relative to lambda = 0
        a = ax[r, 0]
        ends = {}
        for roi in ROIS:
            p = dd[(dd.roi == roi) & (dd.model == "coding")].pivot(
                index="subject", columns="lam", values="cvr2")
            p = (p.sub(p[0.0], axis=0)) * 1e3
            m = sem_line(a, p, ROI_COL[roi])
            ends[roi] = (m.iloc[-1], ROI_COL[roi])
        a.axhline(0, color="0.6", lw=.6, zorder=0)
        for l in (0, 1):
            a.axvline(l, color="0.9", lw=.6, zorder=0)
        a.set_ylabel("Δ cvR² vs λ = 0 (×10⁻³)")
        a.set_xlabel("Tuning axis λ")
        title(a, "Coding", f"{tag}; 0 = fixed value, 1 = quantile")

        # b: connectivity gain over shuffle
        b = ax[r, 1]
        for roi in ROIS:
            sel = dd[dd.roi == roi]
            c = sel[sel.model == "connectivity"].pivot(index="subject", columns="lam",
                                                       values="cvr2")
            s = sel[sel.model == "connectivity_shuffled"].pivot(
                index="subject", columns="lam", values="cvr2")
            sem_line(b, (c - s) * 1e3, ROI_COL[roi])
        b.set_ylim(bottom=0)
        b.set_ylabel("Gain over shuffle (×10⁻³)")
        b.set_xlabel("Connectivity axis λ′")
        title(b, "V1 connectivity", "IPS residual from V1 residual")

        # c: joint, lambda = 1, per lambda'
        c_ = ax[r, 2]
        ends_c = {}
        for roi in ROIS:
            sel = dd[(dd.roi == roi) & (dd.lam == 1.0)]
            j = sel[sel.model == "joint"].pivot(index="subject", columns="lam_conn",
                                                values="cvr2")
            s = sel[sel.model == "joint_shuffled"].pivot(index="subject",
                                                         columns="lam_conn",
                                                         values="cvr2")
            m = sem_line(c_, (j - s) * 1e3, ROI_COL[roi])
            ends_c[roi] = (m.iloc[-1], ROI_COL[roi])
        c_.axhline(0, color="0.6", lw=.6, zorder=0)
        c_.set_ylabel("Gain over shuffle (×10⁻³)")
        c_.set_xlabel("Connectivity axis λ′")
        title(c_, "Joint, tuning-matched", "λ = 1; h has the voxel’s own tuning")
        for axx in (a, b, c_):
            axx.set_xticks([-.5, 0, .5, 1, 1.5])
            axx.set_xlim(-.5, 1.5)
        label_ends(a, ends, 1.55)
        label_ends(c_, ends_c, 1.55)

        # d: joint gain heat map over IPS0-5
        h = ax[r, 3]
        sel = dd[dd.roi.isin(["IPS0", "IPS12", "IPS345"]) & dd.model.isin(
            ["joint", "joint_shuffled"])]
        g = sel.pivot_table(index=["subject", "roi", "lam", "lam_conn"],
                            columns="model", values="cvr2")
        g = (g["joint"] - g["joint_shuffled"]).groupby(["lam", "lam_conn"]).mean()
        M = g.unstack("lam_conn").loc[L, L].to_numpy() * 1e3
        im = h.imshow(M, origin="lower", cmap="mako", aspect="auto",
                      extent=(L[0] - .125, L[-1] + .125, L[0] - .125, L[-1] + .125))
        i, k = np.unravel_index(np.argmax(M), M.shape)
        h.plot(L[k], L[i], "o", ms=4, mfc="none", mec="w", mew=.8)
        h.set_xticks([-.5, .5, 1.5])
        h.set_yticks([-.5, .5, 1.5])
        h.set_xlabel("Connectivity axis λ′")
        h.set_ylabel("Tuning axis λ")
        for sp in h.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=h, fraction=.08, pad=.04)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=5.5, length=2)
        cb.set_label("Gain (×10⁻³)", fontsize=6, labelpad=2)
        title(h, "Joint gain, IPS0–5", "Set by λ′, hardly by λ")
    for axx in ax[:, :3].ravel():
        sns.despine(ax=axx, offset=3, trim=True)
    for r in range(2):
        for k, l in enumerate("abcd"):
            ax[r, k].text(-.32, 1.2, l if r == 0 else l.upper() if False else
                          "efgh"[k], transform=ax[r, k].transAxes, fontsize=9,
                          fontweight="bold")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(f"Wrote {out}")

    # numbers for the caption
    for sm in (False, True):
        dd = d[(d.smoothed == sm) & (d.model == "coding")]
        for roi in ROIS:
            p = dd[dd.roi == roi].pivot(index="subject", columns="lam", values="cvr2")
            t = stats.ttest_rel(p[1.0], p[0.0])
            print(f"  coding {'sm' if sm else 'un'} {roi:6s} λ1-λ0 "
                  f"{1e3 * (p[1.0] - p[0.0]).mean():+.2f}e-3  p={t.pvalue:.2g}  "
                  f"argmax λ={p.mean().idxmax():+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", default=str(REPO / "notes/data/tuning_connectivity_summary.tsv"))
    ap.add_argument("--out-dir", default=str(REPO / "notes/figures"))
    a = ap.parse_args()
    od = Path(a.out_dir)
    explainer(od / "fig_tuning_connectivity_explainer.pdf")
    results(a.tsv, od / "fig_tuning_connectivity.pdf")
