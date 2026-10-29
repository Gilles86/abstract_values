#!/usr/bin/env python3
"""A6 figure: is there a motor-noise signature in M1? (exploratory)

Reads the tables written by analyze_m1_motor_noise.py and writes
derivatives/followup_2026-10/a6_m1_motor_noise/figures/a6_m1_motor_noise.pdf

    ~/mambaforge/bin/python plot_m1_motor_noise.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 4, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "lines.markersize": 4, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

OUT = Path("/data/ds-abstractvalue/derivatives/followup_2026-10/a6_m1_motor_noise")
TAB = OUT / "tables"
A1 = Path("/data/ds-abstractvalue/derivatives/followup_2026-10/a1_motor_stage/tables")
ROIS = ["M1lhand", "M1rhand", "preSMA", "V1", "IPS12", "IPS345"]
LABEL = {"M1lhand": "L M1 hand", "M1rhand": "R M1 hand", "preSMA": "Pre-SMA", "V1": "V1",
         "IPS12": "IPS1–2", "IPS345": "IPS3–5"}
BLUE, GREY, RED = "#3B5BA5", "#7F7F7F", "#C44E52"
COL = {r: (BLUE if r == "M1lhand" else GREY) for r in ROIS}


def mci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), h


def rot(ax):
    for t in ax.get_xticklabels():
        t.set_rotation(40)
        t.set_ha("right")
        t.set_rotation_mode("anchor")


def letter(ax, s):
    ax.text(-0.2, 1.06, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="right")


def key(ax, entries, x=.04, y=.96, dy=.085, seg=.06, fs=6.5):
    tf = ax.transAxes
    for i, (lab, col, kind, o) in enumerate(entries):
        yy = y - i * dy
        if kind == "marker":
            ax.plot(x + seg / 2, yy, o.get("marker", "o"), transform=tf, ms=o.get("ms", 3.6),
                    color=col, mfc=o.get("mfc", col), mew=o.get("mew", 0.9), clip_on=False)
        else:
            ax.plot([x, x + seg], [yy, yy], transform=tf, color=col, ls=o.get("ls", "-"),
                    lw=o.get("lw", 1.2), clip_on=False, solid_capstyle="butt")
        ax.text(x + seg + .03, yy, lab, transform=tf, color=o.get("tc", "0.2"), fontsize=fs,
                va="center")


def roi_panel(ax, sub, col, epochs=("gabor", "response"), ylabel=""):
    """Mean +/- 95% CI over subjects per ROI; filled = gabor epoch, open = response."""
    off = {"gabor": -0.14, "response": 0.14} if len(epochs) == 2 else {epochs[0]: 0.0}
    for i, r in enumerate(ROIS):
        for ep in epochs:
            v = sub[(sub.roi == r) & (sub.epoch == ep)][col]
            m, h = mci(v)
            x = i + off[ep]
            ax.plot([x, x], [m - h, m + h], color=COL[r], lw=1.0, zorder=2)
            ax.plot(x, m, "o", ms=4, color=COL[r], mfc=COL[r] if ep == "gabor" else "white",
                    mew=1.0, zorder=3)
    ax.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
    ax.set_xticks(range(len(ROIS)))
    ax.set_xticklabels([LABEL[r] for r in ROIS])
    ax.set_ylabel(ylabel)


def main():
    t1 = pd.read_csv(TAB / "a6_t1_subject_single.tsv", sep="\t", dtype={"subject": str})
    t24 = pd.read_csv(TAB / "a6_t2_t4_subject.tsv", sep="\t", dtype={"subject": str})
    cells = pd.read_csv(TAB / "a6_t3_cells.tsv", sep="\t", dtype={"subject": str})
    t3 = pd.read_csv(TAB / "a6_t3_subject.tsv", sep="\t", dtype={"subject": str})
    sm = pd.read_csv(A1 / "stats_subject_measures.tsv", sep="\t", dtype={"subject": str})
    sm["subject"] = sm.subject.replace({"01": "pil01", "02": "pil02"})

    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.9), constrained_layout=True)
    (a, b, c), (d, e, f) = axes

    # a: signed error tracking
    roi_panel(a, t1, "coef", ylabel="Signed error ~ decoded bid\n(CHF per SD)")
    key(a, [("Stimulus epoch", "0.25", "marker", {}),
            ("Response epoch", "0.25", "marker", {"mfc": "white"})], x=.05, y=.97)
    a.set_ylim(-0.12, 0.36)
    letter(a, "a")

    # b: Fitts at cell level
    cells["bin"] = cells.groupby("subject").travel_c.transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    w = cells.assign(wo=cells.obs_absdev * cells.n, wf=cells.pred_absdev_full * cells.n,
                     wn=cells.pred_absdev_nomotor * cells.n, wt=cells.travel_c * cells.n)
    g = w.groupby(["subject", "bin"])[["wo", "wf", "wn", "wt", "n"]].sum()
    g = g[["wo", "wf", "wn", "wt"]].div(g.n, axis=0).reset_index()
    agg = g.groupby("bin").agg(t=("wt", "mean"), obs=("wo", lambda v: mci(v)[0]),
                               obs_h=("wo", lambda v: mci(v)[1]), full=("wf", "mean"),
                               nom=("wn", "mean"))
    b.plot(agg.t, agg.full, color=RED, lw=1.3, zorder=2)
    b.plot(agg.t, agg.nom, color=RED, lw=1.1, ls=(0, (3, 2)), zorder=2)
    b.errorbar(agg.t, agg.obs, yerr=agg.obs_h, fmt="o", color="0.15", ms=3.8, lw=0.9,
               capsize=0, zorder=3)
    key(b, [("Observed", "0.15", "marker", {}),
            ("Model, stationary motor noise", RED, "line", {}),
            ("Model, motor noise off", RED, "line", {"ls": (0, (3, 2))})], x=.04, y=.97)
    s = t3[t3.null == "full"].slope_travel
    m, h = mci(s)
    b.text(.97, .05, f"Misfit slope {m:+.3f} [{m - h:+.3f}, {m + h:+.3f}]\nCHF per SD travel",
           transform=b.transAxes, ha="right", va="bottom", fontsize=6.5, color="0.25")
    b.set_xlabel("Expected slider travel (CHF)")
    b.set_ylabel("Mean |bid − model mean| (CHF)")
    b.set_ylim(0, 2.9)
    letter(b, "b")

    # c: individual differences
    ind = t3[t3.null == "full"].set_index("subject")[["slope_travel"]].join(
        sm.set_index("subject")[["travel_M1lhand_gabor"]]).dropna()
    c.scatter(ind.travel_M1lhand_gabor, ind.slope_travel, s=10, color=BLUE, alpha=.7, lw=0)
    rho, p = stats.spearmanr(ind.travel_M1lhand_gabor, ind.slope_travel)
    z, se = np.arctanh(rho), 1.06 / np.sqrt(len(ind) - 3)
    c.text(.97, .95, f"ρ = {rho:.2f} [{np.tanh(z - 1.96 * se):.2f}, "
                     f"{np.tanh(z + 1.96 * se):.2f}]",
           transform=c.transAxes, ha="right", va="top", fontsize=6.5, color="0.25")
    c.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
    c.set_ylim(top=max(c.get_ylim()[1], ind.slope_travel.max()) + 0.12)
    c.set_xlabel("L M1 hand travel β (A1)")
    c.set_ylabel("Distance–error slope\n(CHF per SD travel)")
    letter(c, "c")

    # d: |error| ~ ROI beta (Fitts, trial level)
    roi_panel(d, t24, "t2_main", ylabel="|Error| ~ ROI beta\n(CHF per SD)")
    d.annotate("Fitts predicts > 0", xy=(0.0, 0.04), xytext=(0.0, 0.058),
               fontsize=6.5, ha="left", va="center", color="0.25")
    d.set_ylim(-0.085, 0.07)
    letter(d, "d")

    # e: overshoot in the expected movement direction (response epoch)
    for i, r in enumerate(ROIS):
        for k, (term, mfc) in enumerate([("t4_travel_resid", None),
                                         ("t4_spline_resid", "white")]):
            v = t24[(t24.roi == r) & (t24.epoch == "response")][term]
            mm, hh = mci(v)
            x = i + (-0.14 if k == 0 else 0.14)
            e.plot([x, x], [mm - hh, mm + hh], color=COL[r], lw=1.0)
            e.plot(x, mm, "s", ms=3.8, color=COL[r], mfc=mfc or COL[r], mew=1.0)
    e.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
    e.set_xticks(range(len(ROIS)))
    e.set_xticklabels([LABEL[r] for r in ROIS])
    e.set_ylabel("Response-epoch signed error ~\nbeta × direction (CHF per SD)")
    key(e, [("Beta minus travel(bid)", "0.25", "marker", {"marker": "s"}),
            ("Beta minus any f(bid)", "0.25", "marker", {"marker": "s", "mfc": "white"})],
        x=.42, y=.97)
    e.set_ylim(-0.035, 0.07)
    letter(e, "e")

    # f: unique cross-validated R^2, M1 vs IPS (joint model, from a6_t1_subject_joint)
    tj = pd.read_csv(TAB / "a6_t1_subject_joint.tsv", sep="\t", dtype={"subject": str})
    tj = tj[tj.motor_roi == "M1lhand"]
    for k, (col, lab, cc) in enumerate([("uniq_motor_x1e3", "L M1 hand", BLUE),
                                        ("uniq_ips_x1e3", "IPS1–5", GREY)]):
        for j, ep in enumerate(("gabor", "response")):
            mm, hh = mci(tj[tj.epoch == ep][col])
            x = k + (-0.14 if j == 0 else 0.14)
            f.plot([x, x], [mm - hh, mm + hh], color=cc, lw=1.0)
            f.plot(x, mm, "o", ms=4, color=cc, mfc=cc if ep == "gabor" else "white", mew=1.0)
    f.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
    f.set_xticks([0, 1])
    f.set_xticklabels(["L M1 hand", "IPS1–5"])
    f.set_xlim(-0.6, 1.6)
    f.set_ylabel("Unique cross-validated R² of\nsigned error (×10⁻³)")
    letter(f, "f")

    for ax in axes.flat:
        sns.despine(ax=ax, offset=4, trim=False)
    for ax in (a, d, e):
        ax.tick_params(axis="x", labelrotation=40)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "a6_m1_motor_noise.pdf")
    fig.savefig(OUT / "figures" / "a6_m1_motor_noise.png", dpi=200)


if __name__ == "__main__":
    main()
