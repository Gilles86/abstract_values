"""Paired comparison of expected-decoded MAE and SD under residual vs
spherical noise models, across ROI × voxel-selection × smoothing.

For each per-(subject, session, condition, stimulus) cell we collapse
the per-stimulus TSV to two summary numbers: mean absolute error
(MAE) and mean SD across the trained stimulus grid. Then we plot
residual-noise vs spherical-noise on a paired scatter (one point per
collapsed cell, color-coded by smoothing × selection). The identity
line is the reference: points below the diagonal mean spherical noise
gives lower error / tighter SD on that cell.

Two ROIs:
  - V1 (vonmises, decoded orientation; values are in degrees)
  - NPCr (aprf-session-shift, decoded value; values are in CHF)

Each gets its own page. A summary line in each panel reports n
paired cells, the median fractional change (spherical−residual)/residual,
and a paired t-test p-value.

Usage:
    python -m abstract_values.visualize.compare_noise_models
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
from scipy import stats

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

DERIV = Path(BIDS_FOLDER) / "derivatives" / "encoding_models"
DEFAULT_OUT = (Path(BIDS_FOLDER) / "derivatives" / "qa"
               / "compare_noise_models.pdf")

# Cell-level palette: one color per (selection, smoothing) combination.
CELL_COLOR = {
    ("nvoxels-100",   False): "#1f77b4",
    ("nvoxels-100",   True):  "#1f77b4",
    ("nvoxels-fdr05", False): "#d62728",
    ("nvoxels-fdr05", True):  "#d62728",
}
CELL_MARKER = {False: "o", True: "s"}      # circle = unsmoothed, square = smoothed
SEL_TAGS = ("nvoxels-100", "nvoxels-fdr05")
SMOOTHINGS = (False, True)


def _vonmises_tsv(subject, session, sel_tag, smoothed, noise):
    smooth = "_smoothed" if smoothed else ""
    noise_tag = f"_noise-{noise}" if noise else ""
    return (DERIV / "vonmises" / f"sub-{subject}" / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-abstractvalue"
              f"_mask-BensonV1_hemi-LR_{sel_tag}_nsims-1000"
              f"{noise_tag}{smooth}_desc-expected_decoded_orientation_pe.tsv")


def _aprf_tsv(subject, session, sel_tag, smoothed, noise):
    smooth = "_smoothed" if smoothed else ""
    noise_tag = f"_noise-{noise}" if noise else ""
    return (DERIV / "aprf-session-shift" / f"sub-{subject}"
            / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-abstractvalue"
              f"_mask-NPCr_{sel_tag}_nsims-1000"
              f"{noise_tag}{smooth}_desc-expected_decoded_pe.tsv")


def _collect_one(path_fn, subjects, roi_label):
    """Load TSVs (residual + spherical, all sel × smoothing × sub × ses)
    and collapse each to one mean-MAE + mean-SD scalar per cell."""
    rows = []
    for s in subjects:
        try:
            sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
            sessions = sub.get_sessions()
            mapping_lookup = {ses: sub.get_mapping(ses) for ses in sessions}
        except Exception:
            continue
        for sel_tag in SEL_TAGS:
            for smoothed in SMOOTHINGS:
                for ses in sessions:
                    paths = {
                        "residual":  path_fn(s, ses, sel_tag, smoothed, ""),
                        "spherical": path_fn(s, ses, sel_tag, smoothed, "spherical"),
                    }
                    metrics = {}
                    for noise, p in paths.items():
                        if not p.exists():
                            metrics[noise] = None
                            continue
                        df = pd.read_csv(p, sep="\t")
                        if df.empty:
                            metrics[noise] = None; continue
                        df["sd_E"] = np.sqrt(df["var_E"])
                        metrics[noise] = {
                            "mae": float(df["mean_abs_error"].mean()),
                            "sd":  float(df["sd_E"].mean()),
                        }
                    # Only keep cells where both noise variants are present.
                    if metrics["residual"] is None or metrics["spherical"] is None:
                        continue
                    rows.append({
                        "roi":       roi_label,
                        "subject":   s,
                        "session":   ses,
                        "condition": mapping_lookup.get(ses, "?"),
                        "selection": sel_tag,
                        "smoothed":  smoothed,
                        "mae_residual":   metrics["residual"]["mae"],
                        "mae_spherical":  metrics["spherical"]["mae"],
                        "sd_residual":    metrics["residual"]["sd"],
                        "sd_spherical":   metrics["spherical"]["sd"],
                    })
    return pd.DataFrame(rows)


def discover_subjects():
    seen = set()
    for p in (DERIV / "vonmises").glob("sub-*"):
        seen.add(p.name.removeprefix("sub-"))
    for p in (DERIV / "aprf-session-shift").glob("sub-*"):
        seen.add(p.name.removeprefix("sub-"))
    return sorted(seen, key=lambda s: (0 if s[0].isdigit() else 1, s))


def _paired_scatter(ax, df, metric, label_unit):
    """Plot residual (x) vs spherical (y) for `metric` ∈ {'mae', 'sd'}.
    Adds identity line, per-cell scatter with selection×smoothing
    markers, and a summary annotation with paired stats."""
    xcol = f"{metric}_residual"
    ycol = f"{metric}_spherical"
    x = df[xcol].values
    y = df[ycol].values
    if len(x) == 0:
        ax.set_title(f"{metric.upper()}  —  no paired cells", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        return
    lo = float(min(x.min(), y.min())) * 0.95
    hi = float(max(x.max(), y.max())) * 1.05
    ax.plot([lo, hi], [lo, hi], "-", color="0.7", lw=0.8, zorder=0)
    for (sel, sm), sub in df.groupby(["selection", "smoothed"]):
        ax.scatter(sub[xcol], sub[ycol],
                   s=22, alpha=0.7,
                   color=CELL_COLOR[(sel, sm)],
                   marker=CELL_MARKER[sm],
                   edgecolor="white", linewidth=0.4,
                   label=f"{sel.replace('nvoxels-', '')}  "
                         f"{'smoothed' if sm else 'unsmoothed'}")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Residual-noise {metric.upper()}  ({label_unit})")
    ax.set_ylabel(f"Spherical-noise {metric.upper()}  ({label_unit})")

    # Summary: median fractional change + paired t
    delta = (y - x) / x
    med_pct = 100 * np.median(delta)
    t, p = stats.ttest_rel(y, x)
    sign = "lower" if med_pct < 0 else "higher"
    title = (f"{metric.upper()}  —  spherical median "
             f"{abs(med_pct):.1f}% {sign} than residual  "
             f"(n={len(x)}, paired t={t:.1f}, p={p:.1e})")
    ax.set_title(title, fontsize=8, color="0.2")
    ax.legend(loc="lower right", fontsize=7)


def page_roi(df_roi, roi_label, units, pdf):
    if df_roi.empty:
        # Empty page with a note rather than skipping silently — the
        # reader needs to know data is missing, not assume the figure
        # is complete.
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.axis("off")
        ax.text(0.5, 0.5,
                f"{roi_label}: no paired (residual, spherical) cells yet.\n"
                f"Spherical jobs may still be queued — re-run after they land.",
                ha="center", va="center", fontsize=10, color="0.3")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                              constrained_layout=True)
    fig.suptitle(f"{roi_label}: residual vs spherical noise — paired cells "
                 f"per (sub, ses, sel, smooth)", fontsize=10, y=1.03,
                 color="0.15")
    _paired_scatter(axes[0], df_roi, "mae", units)
    _paired_scatter(axes[1], df_roi, "sd",  units)
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def run(subjects, out):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No subjects discovered.")
    print(f"Subjects: {subjects}")
    out.parent.mkdir(parents=True, exist_ok=True)

    df_v1   = _collect_one(_vonmises_tsv, subjects, "V1 (vonmises)")
    df_npcr = _collect_one(_aprf_tsv,     subjects, "NPCr (aprf-session-shift)")
    print(f"  V1 paired cells:    {len(df_v1)}")
    print(f"  NPCr paired cells:  {len(df_npcr)}")

    # Persist the paired tables so plot tweaks don't require re-globbing
    # the cluster output every time.
    tsv = out.with_suffix(".tsv")
    pd.concat([df_v1, df_npcr], ignore_index=True).to_csv(
        tsv, sep="\t", index=False)
    print(f"Wrote {tsv}")

    with PdfPages(out) as pdf:
        page_roi(df_v1,   "V1 (vonmises, orientation)",        "deg", pdf)
        page_roi(df_npcr, "NPCr (aprf-session-shift, value)",  "CHF", pdf)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, Path(args.out))


if __name__ == "__main__":
    main()
