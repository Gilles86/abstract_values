"""The vmPFC/OFC story in four questions, for collaborators.

  1  Where is the signal lost?           group tSNR (MNI) + ROI tSNR
  2  Is it our protocol?                 normalised mOFC tSNR and signal void across 12 lab datasets
  3  Is there still a task response?     evoked time courses + single-trial betas
  4  Does it carry value?                value effect over time + value decoding per ROI

Reads only the summary files written by combine_vmpfc_dropout, collect_roi_decoding_r
and the dropout_survey combine step.

    python -m abstract_values.visualize.figure_vmpfc_narrative     # abstract_values env
    -> notes/figures/fig_vmpfc_narrative.pdf
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.dropout_survey.datasets import DATASETS
from abstract_values.visualize.plot_vmpfc_dropout import (COL, DATA, FIG, SHORT, bbox,
                                                          mni_slices, plot_curves, show_slice,
                                                          style, subject_curves)

GROUPS = [("Siemens 3T", ["abstractvalue", "numloss", "foodvalue"], "#C44E52"),
          ("Philips 3T", ["numrisk", "dnumrisk", "stressrisk", "neuralpriors", "tmsrisk", "risk3t"],
           "#3B5BA5"),
          ("7T", ["risk7t", "retsupp", "valuecapture"], "#5A8F3A")]
INK = "#222222"
ANSWER = "#8A3B12"


def chapter(fig, y, n, question, answer):
    fig.text(0.015, y, str(n), fontsize=15, fontweight="bold", color="0.75", va="top")
    fig.text(0.055, y - 0.002, question, fontsize=8.5, fontweight="bold", color=INK, va="top")
    fig.text(0.055, y - 0.024, answer, fontsize=7.2, color=ANSWER, va="top", linespacing=1.35)


def strip(ax, d, rois, col, ylabel, ref=None):
    for i, r in enumerate(rois):
        v = d.loc[d.roi == r, col].dropna().to_numpy()
        jit = np.random.default_rng(i).uniform(-0.2, 0.2, len(v))
        ax.scatter(i + jit, v, s=4, color=COL[r], alpha=0.45, lw=0)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=COL[r], lw=2.2,
                solid_capstyle="butt")
    if ref is not None:
        ax.axhline(ref, color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels([SHORT[r] for r in rois])
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax, offset=3)
    ax.tick_params(axis="x", labelrotation=45, labelsize=6.3)
    for t in ax.get_xticklabels():
        t.set_ha("right"); t.set_rotation_mode("anchor")
    ax.set_xlim(-0.6, len(rois) - 0.4)


def main():
    style()
    rois = pd.read_csv(DATA / "vmpfc_dropout_rois.tsv", sep="\t", dtype={"subject": str})
    trials = pd.read_csv(DATA / "vmpfc_dropout_trials.tsv", sep="\t", dtype={"subject": str})
    ep = np.load(DATA / "vmpfc_dropout_epochs.npz")
    mni = np.load(DATA / "vmpfc_dropout_mni.npz")
    surv = pd.read_csv(DATA / "dropout_survey_rois.tsv", sep="\t", dtype={"subject": str})
    dec = pd.read_csv(DATA / "vmpfc_decoding_r.tsv", sep="\t", dtype={"subject": str})

    fig = plt.figure(figsize=(7.25, 9.8))

    # ── 1  where ────────────────────────────────────────────────────────────
    chapter(fig, 0.985, 1, "Where is the signal lost?",
            "Over the orbitofrontal cortex, just\nabove the sinuses. Medial and\n"
            "lateral OFC keep about half the\nsignal of the rest of cortex;\n"
            "pregenual ACC and dorsomedial\nPFC are fine.")
    sl, (xm, ym, zm) = mni_slices(mni)
    im = None
    for c, (view, lab) in enumerate((("sag", f"x = {xm} mm"), ("ax", f"z = {zm} mm"))):
        ax = fig.add_axes([0.37 + c * 0.155, 0.80, 0.145, 0.175])
        br = sl["brain"][view] > 0
        im = show_slice(ax, sl["template"][view], np.where(br, sl["tsnr"][view], np.nan),
                        sl["lab"][view], vmax=100, crop=bbox(br, pad=2),
                        alpha_over=br.astype(float), lw=0.9)
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, color="white", fontsize=6, va="top")
    cax = fig.add_axes([0.68, 0.83, 0.007, 0.12])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("Group tSNR", fontsize=6.3, labelpad=1)
    cax.tick_params(labelsize=5.8, length=2)
    ax = fig.add_axes([0.815, 0.835, 0.17, 0.14])
    strip(ax, rois, ["mOFC", "lOFC", "rACC", "dmPFC", "V1", "IPS12", "WM"],
          "tsnr_median", "Median voxel tSNR")
    ax.set_ylim(0, 130)

    # ── 2  protocol ────────────────────────────────────────────────────────
    y2 = 0.735
    chapter(fig, y2, 2, "Is it our protocol?",
            "Mostly not. All 12 lab datasets\nlose medial-OFC signal, and ours\n"
            "sits mid-pack. Siemens 3T\nprotocols lose somewhat more\n"
            "than the older Philips 3T ones\n(shorter TE, no multiband).")
    m = surv[surv.roi == "mOFC"]
    order = [d for _, g, _ in GROUPS for d in g if d in set(m.dataset)]
    colour = {d: c for _, g, c in GROUPS for d in g}
    axa = fig.add_axes([0.50, y2 - 0.215, 0.245, 0.215])
    axb = fig.add_axes([0.79, y2 - 0.215, 0.195, 0.215])
    for ax, col, xl in ((axa, "tsnr_norm", "Medial-OFC tSNR / whole-cortex median"),
                        (axb, "frac_void", "Medial-OFC voxels in void (%)")):
        for i, ds in enumerate(order):
            v = m.loc[m.dataset == ds, col].dropna().to_numpy()
            if col == "frac_void":
                v = v * 100
            jit = np.random.default_rng(i).uniform(-0.22, 0.22, len(v))
            hl = ds == "abstractvalue"
            ax.scatter(v, i + jit, s=3.5, color=colour[ds], alpha=0.35, lw=0)
            ax.plot([np.median(v)] * 2, [i - 0.33, i + 0.33], color=colour[ds],
                    lw=3.2 if hl else 2.0, solid_capstyle="butt")
            if hl:
                ax.axhspan(i - 0.5, i + 0.5, color="#F3E3D3", zorder=0, lw=0)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_yticks(range(len(order)))
        ax.set_xlabel(xl)
        sns.despine(ax=ax, offset=3, left=True)
        ax.tick_params(axis="y", length=0)
    axa.set_yticklabels([DATASETS[d].label + (f" (n = {m[m.dataset == d].subject.nunique()})")
                         for d in order], fontsize=6.3)
    for t, d in zip(axa.get_yticklabels(), order):
        t.set_color(colour[d]); t.set_fontweight("bold" if d == "abstractvalue" else "normal")
    axb.set_yticklabels([])
    axa.axvline(1, color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=0)
    axa.text(1.0, -0.9, "Rest of cortex", fontsize=5.8, color="0.45", ha="center")
    axa.set_xlim(0, 1.15)
    axb.set_xlim(0, 85)
    for j, (g, _, c) in enumerate(GROUPS):
        axb.text(1.0, 0.24 - j * 0.07, g, transform=axb.transAxes, color=c, fontsize=6.3,
                 ha="right", va="top")

    # ── 3  response ────────────────────────────────────────────────────────
    y3 = 0.455
    chapter(fig, y3, 3, "Is there still a task response?",
            "Yes. Medial PFC deactivates on\nevery trial, as the default-mode\n"
            "network does during a task\n(GLMsingle betas, t = −4 to −11).\n"
            "Dorsal ACC and motor cortex\nactivate for the bid.")
    C, croi, _ = subject_curves(ep, trials)
    tg = ep["t_gabor"]
    delay, rt = trials.bar_delay.median(), trials.rt.median()
    ax = fig.add_axes([0.40, y3 - 0.17, 0.28, 0.165])
    ax.axvspan(0, 1.5, color="#1B8A80", alpha=0.12, lw=0)
    ax.axvspan(delay, delay + rt, color="#D9730D", alpha=0.12, lw=0)
    show = ["mOFC", "rACC", "dmPFC", "dACC", "Lprecentral"]
    plot_curves(ax, tg, C["mean_g"], croi, show)
    ax.set_xlim(tg[0], tg[-1])
    ax.set_xlabel("Time from gabor onset (s)")
    ax.set_ylabel("Signal change (%)")
    sns.despine(ax=ax, offset=3)
    ax.text(0.3, 1.0, "Gabor", color="#1B8A80", fontsize=6, transform=ax.get_xaxis_transform(),
            va="bottom")
    ax.text(delay + 0.3, 1.0, "Bid", color="#D9730D", fontsize=6,
            transform=ax.get_xaxis_transform(), va="bottom")
    for k, r in enumerate(show):
        ax.text(1.0, 0.98 - k * 0.085, SHORT[r], color=COL[r], transform=ax.transAxes,
                ha="right", va="top", fontsize=6.2)
    ax = fig.add_axes([0.79, y3 - 0.17, 0.195, 0.165])
    strip(ax, rois, ["mOFC", "lOFC", "rACC", "dmPFC", "dACC", "Lprecentral", "V1", "IPS12", "WM"],
          "beta_gabor", "Single-trial β, gabor (%)", ref=0)

    # ── 4  value ───────────────────────────────────────────────────────────
    y4 = 0.205
    chapter(fig, y4, 4, "Does it carry value?",
            "No. Value does not move the\nmedial-PFC response, not even in\n"
            "dorsomedial PFC where the signal\nis good (the bid doesn't either).\n"
            "Value is decodable from V1 and\nIPS, not from vmPFC/OFC.")
    ax = fig.add_axes([0.40, y4 - 0.15, 0.28, 0.145])
    ax.axvspan(0, 1.5, color="#1B8A80", alpha=0.12, lw=0)
    ax.axvspan(delay, delay + rt, color="#D9730D", alpha=0.12, lw=0)
    plot_curves(ax, tg, C["value"], croi, ["mOFC", "rACC", "dmPFC"])
    ax.set_xlim(tg[0], tg[-1])
    ax.set_ylim(-0.04, 0.04)
    ax.set_xlabel("Time from gabor onset (s)")
    ax.set_ylabel("% signal per SD of value")
    sns.despine(ax=ax, offset=3)
    for k, r in enumerate(["mOFC", "rACC", "dmPFC"]):
        ax.text(1.0, 0.98 - k * 0.085, SHORT[r], color=COL[r], transform=ax.transAxes,
                ha="right", va="top", fontsize=6.2)
    ax = fig.add_axes([0.79, y4 - 0.15, 0.195, 0.145])
    dmap = [("BensonV1", "V1", COL["V1"]), ("IPS0", "IPS0", COL["IPS0"]),
            ("IPS12", "IPS1–2", COL["IPS12"]), ("IPS345", "IPS3–5", COL["IPS345"]),
            ("vmPFCOFCaparc", "vmPFC/OFC", COL["mOFC"])]
    for i, (r, lab, c) in enumerate(dmap):
        v = dec.loc[dec.roi == r].groupby("subject").r.mean().to_numpy()
        jit = np.random.default_rng(i).uniform(-0.2, 0.2, len(v))
        ax.scatter(i + jit, v, s=4, color=c, alpha=0.45, lw=0)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=c, lw=2.2, solid_capstyle="butt")
    ax.axhline(0, color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(range(len(dmap)))
    ax.set_xticklabels([d[1] for d in dmap])
    ax.set_ylabel("Decoded vs true value, r")
    sns.despine(ax=ax, offset=3)
    ax.tick_params(axis="x", labelrotation=45, labelsize=6.3)
    for t in ax.get_xticklabels():
        t.set_ha("right"); t.set_rotation_mode("anchor")

    for y in (0.752, 0.472, 0.222):
        fig.add_artist(plt.Line2D([0.015, 0.985], [y, y], color="0.85", lw=0.6))
    fig.savefig(FIG / "fig_vmpfc_narrative.pdf")
    fig.savefig(FIG / "fig_vmpfc_narrative.png", dpi=200)
    print(f"Wrote {FIG / 'fig_vmpfc_narrative.pdf'}")


if __name__ == "__main__":
    main()
