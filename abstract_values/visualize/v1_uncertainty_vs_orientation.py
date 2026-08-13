"""V1 decoder discriminability (1/SD) as a function of orientation,
with stimulus density on the same x-axis.

V1 analogue of npcr_uncertainty_vs_value. The setup is symmetric to
NPCr (same EU pipeline, same noise model, same selection) but the
stimulus density is by construction FLAT across the 23 trained
orientations — the experiment samples them with equal frequency.

So the 1/SD curve here isn't predicted by the stimulus distribution.
Instead it's diagnostic for the V1 population's preferred-orientation
structure:
  - Cardinal-preferred voxels are SPARSE (see notes/v1_decoder_edge_effect/);
    voxels cluster around the obliques (~45° and 135°).
  - That predicts 1/SD HIGH on the flanks of the oblique clusters,
    LOW at the cardinals (0/90/180) — same lognormal-flank geometry
    as NPCr.

Reads vonmises-session-shift EU TSVs (the symmetric-fits set).

Usage:
    python -m abstract_values.visualize.v1_uncertainty_vs_orientation
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

DERIV = Path(BIDS_FOLDER) / "derivatives" / "encoding_models" / "vonmises-session-shift"
DEFAULT_OUT = (Path(BIDS_FOLDER) / "derivatives" / "qa"
               / "v1_uncertainty_vs_orientation.pdf")
COND_COLOUR = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
SELECTIONS  = ("nvoxels-fdr05", "nvoxels-100")
SMOOTHINGS  = (False, True)


def _v1_tsv(subject, session, sel_tag, smoothed, noise):
    smooth = "_smoothed" if smoothed else ""
    noise_tag = f"_noise-{noise}" if noise else ""
    return (DERIV / f"sub-{subject}" / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-abstractvalue"
              f"_mask-BensonV1_hemi-LR_{sel_tag}_nsims-1000{noise_tag}"
              f"{smooth}_desc-expected_decoded_orientation_pe.tsv")


def discover_subjects():
    return sorted({p.name.removeprefix("sub-") for p in DERIV.glob("sub-*")},
                   key=lambda s: (0 if s[0].isdigit() else 1, s))


def load_v1(subjects, sel_tag, smoothed, noise):
    rows = []
    for s in subjects:
        try:
            sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
        except Exception:
            continue
        for ses in sub.get_sessions():
            p = _v1_tsv(s, ses, sel_tag, smoothed, noise)
            if not p.exists():
                continue
            df = pd.read_csv(p, sep="\t")
            df["orientation_deg"] = np.rad2deg(df["value"])
            df["sd_deg"]   = np.rad2deg(np.sqrt(df["var_E"]))
            df["inv_sd"]   = 1.0 / np.where(df["sd_deg"] > 1e-6,
                                              df["sd_deg"], np.nan)
            df["subject"]  = s
            df["session"]  = ses
            df["condition"] = sub.get_mapping(ses)
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_orientation_density(subjects):
    """Per-condition pooled gabor orientations (deg) across the cohort.

    By design the experiment uses the same 23-point grid in both
    conditions, so the two histograms should overlap.
    """
    out = {"cdf": [], "inverse_cdf": []}
    for s in subjects:
        try:
            sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
            for ses in sub.get_sessions():
                cond = sub.get_mapping(ses)
                ev = sub.get_events(ses, sub.get_runs(ses))
                ors = ev[ev.event_type == "gabor"]["orientation"].astype(float).values
                out[cond].extend(ors.tolist())
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
        return None, None, None, np.empty((0, len(grid))), 0
    arr = np.asarray(per_sub)
    return (np.nanmedian(arr, axis=0),
            np.nanpercentile(arr, 25, axis=0),
            np.nanpercentile(arr, 75, axis=0),
            arr, arr.shape[0])


def page(subjects, sel_tag, smoothed, noise, ori_dist, pdf):
    df = load_v1(subjects, sel_tag, smoothed, noise)
    if df.empty:
        return
    ori_lo = max(0.0, float(df["orientation_deg"].min()) - 2)
    ori_hi = float(df["orientation_deg"].max()) + 8
    grid = np.linspace(ori_lo, float(df["orientation_deg"].max()) + 0.5, 80)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.2),
                              constrained_layout=True, sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    smooth_lbl = "smoothed" if smoothed else "unsmoothed"
    fig.suptitle(
        f"V1 discriminability (1/SD) vs orientation\n"
        f"({sel_tag}  ·  {smooth_lbl}  ·  noise: {noise.upper()}  ·  "
        f"n={df['subject'].nunique()})",
        fontsize=10, y=1.04, color="0.15")

    # ── Top: 1/SD vs orientation ──────────────────────────────────────────
    ax = axes[0]
    last_xy = {}
    y_max = 0.0
    for cond, sub_df in df.groupby("condition"):
        med, q25, q75, per_sub_arr, n = _aggregate(
            sub_df, "orientation_deg", "inv_sd", grid)
        if med is None: continue
        for row in per_sub_arr:
            ax.plot(grid, row, color=COND_COLOUR[cond], lw=0.5,
                     alpha=0.18, zorder=1)
        ax.fill_between(grid, q25, q75, color=COND_COLOUR[cond],
                         alpha=0.22, linewidth=0, zorder=2)
        ax.plot(grid, med, color=COND_COLOUR[cond], lw=2.0, zorder=3,
                 label="_nolegend_")
        last_xy[cond] = (grid[-1], float(med[-1]), n)
        y_max = max(y_max, float(np.nanpercentile(q75, 99)))
    for cond, (x, y, n) in last_xy.items():
        ax.text(x + 1.5, y, "CDF" if cond == "cdf" else "InvCDF",
                color=COND_COLOUR[cond], fontsize=8.5,
                fontweight="bold", va="center")
    # Cardinal references
    for c in (45, 90, 135):
        ax.axvline(c, color="0.8", lw=0.5, ls=":", zorder=0)
    if y_max > 0:
        ax.set_ylim(0, y_max * 1.4)
    ax.set_ylabel(r"V1 1/SD  (deg$^{-1}$)")
    ax.set_title("Decoded discriminability per orientation  "
                  "(thin = per subject; thick = median; band = IQR)",
                  fontsize=9, color="0.2")

    # ── Bottom: stimulus density (should be ≈ flat by design) ─────────────
    ax = axes[1]
    for cond, vs in ori_dist.items():
        if len(vs) == 0: continue
        sns.kdeplot(vs, ax=ax, color=COND_COLOUR[cond], fill=True,
                     alpha=0.22, lw=1.4, clip=(0, 180), cut=0)
    for c in (45, 90, 135):
        ax.axvline(c, color="0.8", lw=0.5, ls=":", zorder=0)
    ax.set_xlabel("Orientation (deg)")
    ax.set_ylabel("Stimulus density")
    ax.set_yticks([])
    ax.set_title("Per-condition orientation density (KDE)  "
                  "— flat by design (same 23 orientations in both)",
                  fontsize=9, color="0.2")
    ax.set_xlim(ori_lo, ori_hi)
    axes[0].set_xlim(ori_lo, ori_hi)
    sns.despine(ax=axes[0], offset=4)
    sns.despine(ax=axes[1], offset=4, left=True)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def run(subjects, out):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No subjects with vonmises-session-shift EU TSVs.")
    print(f"Subjects: {subjects}")
    ori_dist = load_orientation_density(subjects)
    print(f"  ori density n: CDF={len(ori_dist['cdf'])}  "
          f"InvCDF={len(ori_dist['inverse_cdf'])}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for sel_tag in SELECTIONS:
            for smoothed in SMOOTHINGS:
                print(f"\n  {sel_tag}  smoothed={smoothed}")
                page(subjects, sel_tag, smoothed, "spherical",
                     ori_dist, pdf)
    print(f"\nWrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, Path(args.out))


if __name__ == "__main__":
    main()
