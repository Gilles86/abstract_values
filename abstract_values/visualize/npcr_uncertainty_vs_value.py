"""NPCr decoder uncertainty (expected SD) as a function of CHF value,
overlaid with the per-condition stimulus density.

Tests the density-discriminability tradeoff prediction:
  SD should peak where stimulus density peaks, because voxels clump
  their preferred CHF in the dense region (efficient code) and lose
  fine discrimination there.

Same EU TSVs as mapping_invariance / compare_v1_npcr_uncertainty —
``aprf-session-shift/sub-XX/ses-Y/func/`` with the spherical-noise tag.

Layout per page:
  - Top:    NPCr expected SD vs CHF, per condition, mean ± SEM band.
  - Bottom: Per-condition stimulus density (KDE) on the same CHF
            axis, scaled so peaks are visually comparable to the SD
            curves above.

One page per (selection, smoothing) combo, with spherical-only
(matching the user's preference established in compare_v1_npcr_uncertainty).

Usage:
    python -m abstract_values.visualize.npcr_uncertainty_vs_value
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
from matplotlib.backends.backend_pdf import PdfPages

from abstract_values.utils.data import BIDS_FOLDER, Subject

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

DERIV = Path(BIDS_FOLDER) / "derivatives" / "encoding_models" / "aprf-session-shift"
DEFAULT_OUT = (Path(BIDS_FOLDER) / "derivatives" / "qa"
               / "npcr_uncertainty_vs_value.pdf")
COND_COLOUR = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
SELECTIONS  = ("nvoxels-fdr05", "nvoxels-100")
SMOOTHINGS  = (False, True)


def _npcr_tsv(subject, session, sel_tag, smoothed, noise):
    smooth = "_smoothed" if smoothed else ""
    noise_tag = f"_noise-{noise}" if noise else ""
    return (DERIV / f"sub-{subject}" / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-abstractvalue"
              f"_mask-NPCr_{sel_tag}_nsims-1000{noise_tag}{smooth}"
              f"_desc-expected_decoded_pe.tsv")


def discover_subjects():
    return sorted({p.name.removeprefix("sub-") for p in DERIV.glob("sub-*")},
                   key=lambda s: (0 if s[0].isdigit() else 1, s))


def load_npcr(subjects, sel_tag, smoothed, noise):
    rows = []
    for s in subjects:
        try:
            sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
        except Exception:
            continue
        for ses in sub.get_sessions():
            p = _npcr_tsv(s, ses, sel_tag, smoothed, noise)
            if not p.exists():
                continue
            df = pd.read_csv(p, sep="\t")
            df["sd_E"]     = np.sqrt(df["var_E"])
            df["subject"]  = s
            df["session"]  = ses
            df["condition"] = sub.get_mapping(ses)
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_stimulus_density(subjects):
    """Per-condition pooled gabor CHF values across the cohort."""
    out = {"cdf": [], "inverse_cdf": []}
    for s in subjects:
        try:
            sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
            for ses in sub.get_sessions():
                cond = sub.get_mapping(ses)
                ev = sub.get_events(ses, sub.get_runs(ses))
                vs = ev[ev.event_type == "gabor"]["value"].astype(float).values
                out[cond].extend(vs.tolist())
        except Exception:
            pass
    return {k: np.asarray(v) for k, v in out.items()}


def _aggregate(df, x_col, y_col, grid):
    per_sub = []
    for _, g in df.groupby("subject"):
        g = g.sort_values(x_col)
        if g[x_col].nunique() < 3:
            continue
        per_sub.append(np.interp(grid, g[x_col].values, g[y_col].values,
                                  left=np.nan, right=np.nan))
    if not per_sub:
        return None, None, 0
    arr = np.asarray(per_sub)
    n_eff = np.maximum(np.sum(~np.isnan(arr), axis=0), 1)
    return (np.nanmean(arr, axis=0),
            np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_eff),
            arr.shape[0])


def page(subjects, sel_tag, smoothed, noise, stim_dist, pdf):
    df = load_npcr(subjects, sel_tag, smoothed, noise)
    if df.empty:
        return
    chf_lo = max(0.0, float(df["value"].min()) - 1)
    chf_hi = float(df["value"].max()) + 3
    chf_grid = np.linspace(chf_lo, chf_hi, 80)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.2),
                              constrained_layout=True, sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    smooth_lbl = "smoothed" if smoothed else "unsmoothed"
    fig.suptitle(
        f"NPCr uncertainty vs value (CHF axis)  ·  "
        f"{sel_tag}  ·  {smooth_lbl}  ·  noise: {noise.upper()}",
        fontsize=10, y=1.03, color="0.15")

    # ── Top: SD vs CHF, per condition ──────────────────────────────────────
    ax = axes[0]
    last_xy = {}
    for cond, sub_df in df.groupby("condition"):
        mean, sem, n = _aggregate(sub_df, "value", "sd_E", chf_grid)
        if mean is None: continue
        ax.plot(chf_grid, mean, color=COND_COLOUR[cond], lw=1.8,
                label="_nolegend_")
        ax.fill_between(chf_grid, mean - sem, mean + sem,
                         color=COND_COLOUR[cond], alpha=0.22, linewidth=0)
        last_xy[cond] = (chf_grid[-1], float(mean[-1]), n)
    # Direct end-of-line labels
    for cond, (x, y, n) in last_xy.items():
        ax.text(x + 0.6, y, "CDF" if cond == "cdf" else "InvCDF",
                color=COND_COLOUR[cond], fontsize=8.5, fontweight="bold",
                va="center")
    ax.set_ylabel("NPCr expected SD (CHF)")
    ax.set_title("Decoded SD per CHF — peaks where stimuli are dense?",
                  fontsize=9, color="0.2")

    # ── Bottom: stimulus density KDE, per condition ───────────────────────
    ax = axes[1]
    for cond, vs in stim_dist.items():
        if len(vs) == 0: continue
        sns.kdeplot(vs, ax=ax, color=COND_COLOUR[cond], fill=True,
                     alpha=0.22, lw=1.4, clip=(chf_lo, chf_hi), cut=0)
        med = float(np.median(vs))
        ax.axvline(med, color=COND_COLOUR[cond], lw=0.6, ls="--",
                    zorder=0, alpha=0.8)
    ax.set_xlabel("True CHF value")
    ax.set_ylabel("Stimulus density")
    ax.set_yticks([])
    ax.set_title("Per-condition stimulus density (KDE)",
                  fontsize=9, color="0.2")
    ax.set_xlim(chf_lo, chf_hi + 3)
    axes[0].set_xlim(chf_lo, chf_hi + 3)
    sns.despine(fig=fig, offset=4, left=True, top=True, right=True,
                 bottom=False)
    # Keep the SD panel's left spine — only the bottom (density) panel
    # loses its y-spine since there's no quantitative y-scale to show.
    sns.despine(ax=axes[0], offset=4)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def run(subjects, out):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No subjects with aprf-session-shift fits.")
    print(f"Subjects: {subjects}")
    stim_dist = load_stimulus_density(subjects)
    print(f"  stimulus density n: CDF={len(stim_dist['cdf'])}  "
          f"InvCDF={len(stim_dist['inverse_cdf'])}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for sel_tag in SELECTIONS:
            for smoothed in SMOOTHINGS:
                print(f"\n  {sel_tag}  smoothed={smoothed}")
                page(subjects, sel_tag, smoothed, "spherical",
                     stim_dist, pdf)
    print(f"\nWrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, Path(args.out))


if __name__ == "__main__":
    main()
