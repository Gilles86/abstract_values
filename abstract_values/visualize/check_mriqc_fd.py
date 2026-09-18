"""QA: head-motion summary from MRIQC — %FD>0.2mm per subject, ranked.

Reads MRIQC's per-run BOLD group table and summarizes framewise
displacement per subject: mean/median/max percentage of frames exceeding
the standard 0.2 mm threshold (`fd_perc`), averaged across all runs and
sessions. Study subjects are ranked and compared to the group mean ± SD;
`sub-pil##` (MRI-protocol pilots, not study participants — see project
CLAUDE.md) are shown separately and excluded from group statistics.

Output is a 2-page PDF:
    page 1 — ranked bar chart, worst subject at top, group mean ± 1 SD band
    page 2 — full per-subject table (mean/median/max %FD, mean FD, z, percentile)

Input:
    derivatives/mriqc/group_bold.tsv

Output:
    derivatives/qa/mriqc_fd_qc.pdf
    derivatives/qa/mriqc_fd_summary.tsv

Usage:
    python -m abstract_values.visualize.check_mriqc_fd
    python -m abstract_values.visualize.check_mriqc_fd --fd-threshold-label "0.2 mm"
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
from matplotlib.backends.backend_pdf import PdfPages

from abstract_values.utils.data import BIDS_FOLDER

DERIV = Path(BIDS_FOLDER) / "derivatives"
MRIQC_TSV = DERIV / "mriqc" / "group_bold.tsv"
DEFAULT_OUT = DERIV / "qa" / "mriqc_fd_qc.pdf"
DEFAULT_SUMMARY_TSV = DERIV / "qa" / "mriqc_fd_summary.tsv"

COLOR_NORMAL = "#3B5BA5"
COLOR_OUTLIER = "#C44E52"
COLOR_PILOT = "#9C9C9C"
COLOR_DOT = "0.25"
OUTLIER_Z = 1.5

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 0,
    "xtick.major.width": 0.8,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(MRIQC_TSV, sep="\t")
    df["subject"] = df["bids_name"].str.extract(r"(sub-[a-zA-Z0-9]+)")
    df["is_pilot"] = df["subject"].str.startswith("sub-pil")

    agg = df.groupby("subject", as_index=False).agg(
        n_runs=("fd_perc", "size"),
        fd_perc_mean=("fd_perc", "mean"),
        fd_perc_median=("fd_perc", "median"),
        fd_perc_max=("fd_perc", "max"),
        fd_mean_mean=("fd_mean", "mean"),
    )
    agg["is_pilot"] = agg["subject"].str.startswith("sub-pil")

    study = agg.loc[~agg["is_pilot"]].copy()
    mu, sd = study["fd_perc_mean"].mean(), study["fd_perc_mean"].std()
    agg["z_fd_perc_mean"] = np.where(agg["is_pilot"], np.nan, (agg["fd_perc_mean"] - mu) / sd)
    pct = pd.Series(study["fd_perc_mean"].rank(pct=True).values * 100, index=study["subject"])
    agg["percentile"] = agg["subject"].map(pct)

    agg.attrs["group_mean"] = mu
    agg.attrs["group_sd"] = sd
    return agg, df


def plot_ranked_bars(pdf: PdfPages, agg: pd.DataFrame, runs: pd.DataFrame, fd_label: str):
    study = agg.loc[~agg["is_pilot"]].sort_values("fd_perc_mean", ascending=True).reset_index(drop=True)
    pilots = agg.loc[agg["is_pilot"]].sort_values("fd_perc_mean", ascending=True).reset_index(drop=True)
    mu, sd = agg.attrs["group_mean"], agg.attrs["group_sd"]

    n_pilot = len(pilots)
    n_study = len(study)
    gap = 1
    y_pilot = np.arange(n_pilot)
    y_study = np.arange(n_study) + n_pilot + gap
    n_total = n_pilot + gap + n_study

    headroom = 1.6
    fig, ax = plt.subplots(figsize=(7.25, 0.235 * (n_total + headroom) + 1.3), constrained_layout=True)

    ax.axvspan(mu - sd, mu + sd, color="0.5", alpha=0.08, zorder=0, lw=0)
    ax.vlines(mu, -0.8, n_total - 0.9, color="0.35", lw=1.0, ls="--", zorder=1)

    colors_study = [
        COLOR_OUTLIER if abs(z) >= OUTLIER_Z else COLOR_NORMAL for z in study["z_fd_perc_mean"]
    ]
    ax.barh(y_study, study["fd_perc_mean"], color=colors_study, height=0.62, zorder=3)
    ax.barh(y_pilot, pilots["fd_perc_mean"], color=COLOR_PILOT, height=0.62, zorder=3)

    runs_s = runs.merge(agg[["subject", "is_pilot"]], on="subject")
    y_lookup = dict(zip(study["subject"], y_study)) | dict(zip(pilots["subject"], y_pilot))
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.22, 0.22, size=len(runs_s))
    ax.scatter(
        runs_s["fd_perc"],
        runs_s["subject"].map(y_lookup) + jitter,
        s=4, color=COLOR_DOT, alpha=0.45, zorder=4, linewidths=0,
    )

    subj_dot_max = runs_s.groupby("subject")["fd_perc"].max()

    outlier_ends = [
        subj_dot_max[subj] for subj, z in zip(study["subject"], study["z_fd_perc_mean"]) if abs(z) >= OUTLIER_Z
    ]
    xmax = max(float(runs["fd_perc"].max()), max(outlier_ends, default=0) + 14)
    xtick_max = int(np.ceil(float(runs["fd_perc"].max()) / 10.0) * 10)
    ax.set_xticks(np.arange(0, xtick_max + 1, 10))
    ax.set_xlim(0, xmax + 2)
    ax.set_xlabel(f"% Frames with FD > {fd_label}")

    yticks = np.concatenate([y_pilot, y_study])
    ylabels = list(pilots["subject"]) + list(study["subject"])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(-0.8, n_total + headroom - 0.2)

    for z, subj in zip(study["z_fd_perc_mean"], study["subject"]):
        if abs(z) >= OUTLIER_Z:
            y = y_lookup[subj]
            ax.text(subj_dot_max[subj] + 1.5, y, f"z = {z:+.1f}", va="center", ha="left",
                     fontsize=7.5, color=COLOR_OUTLIER)

    ax.plot([mu, mu], [n_total - 0.9, n_total - 0.5], color="0.35", lw=1.0, ls="--", zorder=1)
    ax.text(mu, n_total - 0.3, "Group mean", fontsize=7.5, color="0.35", va="bottom", ha="center")

    if n_pilot:
        ax.axhline(n_pilot + gap - 0.5, color="0.75", lw=0.6, ls=":", zorder=2)
        ax.text(xmax + 1.8, (n_pilot - 1) / 2, "Pilots\n(excl. from\ngroup stats)",
                 fontsize=7, color=COLOR_PILOT, va="center", ha="left", style="italic")

    import seaborn as sns
    sns.despine(ax=ax, offset=5, trim=True)

    ax.set_title("MRIQC head-motion QC — % frames exceeding FD threshold, ranked", fontsize=10, pad=10)

    caption = (
        f"Mean % of frames with framewise displacement > {fd_label} per run, averaged across runs "
        f"per subject (dots = individual runs). Study group (n={n_study}): mean {mu:.1f}%, "
        f"SD {sd:.1f}% (shaded band = ±1 SD, dashed line = mean). Red bars: |z| ≥ {OUTLIER_Z:g}. "
        f"sub-18 has 17 runs (extra run from dropped-trigger recovery), sub-26 has 15 (one missing run)."
    )
    fig.text(0.0, -0.02, caption, fontsize=6.5, color="0.3", wrap=True, ha="left", va="top")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_table(pdf: PdfPages, agg: pd.DataFrame, fd_label: str):
    study = agg.loc[~agg["is_pilot"]].sort_values("fd_perc_mean", ascending=False).reset_index(drop=True)
    pilots = agg.loc[agg["is_pilot"]].sort_values("fd_perc_mean", ascending=False).reset_index(drop=True)
    mu, sd = agg.attrs["group_mean"], agg.attrs["group_sd"]

    cols = ["Subject", "Runs", "Mean %FD", "Median %FD", "Max %FD", "FD mean (mm)", "z", "Percentile"]
    xpos = [0.03, 0.24, 0.36, 0.49, 0.62, 0.75, 0.87, 0.95]
    ha = ["left", "center", "center", "center", "center", "center", "center", "center"]

    n_rows = len(study) + len(pilots) + 1  # +1 spacer between blocks
    n_lines = n_rows + 3  # header + summary block + margins
    fig, ax = plt.subplots(figsize=(7.25, 0.205 * n_lines + 0.35))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    dy = 1.0 / (n_lines + 1)
    y = 1.0 - dy * 0.6

    for x, label, alignment in zip(xpos, cols, ha):
        ax.text(x, y, label, fontsize=8, fontweight="bold", ha=alignment, va="center")
    y -= dy * 0.55
    ax.plot([0.0, 1.0], [y, y], color="0.2", lw=0.8, transform=ax.transAxes)
    y -= dy * 0.75

    def row(values, color="black", bg=None, italic=False):
        nonlocal y
        if bg is not None:
            ax.axhspan(y - dy * 0.42, y + dy * 0.42, xmin=0.0, xmax=1.0, color=bg, lw=0, zorder=0)
        for x, val, alignment in zip(xpos, values, ha):
            ax.text(x, y, val, fontsize=8, ha=alignment, va="center", color=color,
                     style="italic" if italic else "normal")
        y -= dy

    for i, r in study.iterrows():
        outlier = abs(r["z_fd_perc_mean"]) >= OUTLIER_Z
        bg = "#F7DEDE" if outlier else ("#F7F7F7" if i % 2 else None)
        color = COLOR_OUTLIER if outlier else "black"
        row([
            r["subject"], f"{r['n_runs']:.0f}", f"{r['fd_perc_mean']:.1f}", f"{r['fd_perc_median']:.1f}",
            f"{r['fd_perc_max']:.1f}", f"{r['fd_mean_mean']:.2f}", f"{r['z_fd_perc_mean']:+.2f}",
            f"{r['percentile']:.0f}",
        ], color=color, bg=bg)

    y -= dy * 0.3
    ax.plot([0.0, 1.0], [y + dy * 0.65, y + dy * 0.65], color="0.7", lw=0.6, ls=":", transform=ax.transAxes)
    for _, r in pilots.iterrows():
        row([
            r["subject"], f"{r['n_runs']:.0f}", f"{r['fd_perc_mean']:.1f}", f"{r['fd_perc_median']:.1f}",
            f"{r['fd_perc_max']:.1f}", f"{r['fd_mean_mean']:.2f}", "—", "—",
        ], color=COLOR_PILOT, italic=True)

    y -= dy * 0.4
    ax.plot([0.0, 1.0], [y + dy * 0.65, y + dy * 0.65], color="0.2", lw=0.8, transform=ax.transAxes)
    study_max = study.loc[study["fd_perc_mean"].idxmax(), "subject"]
    study_min = study.loc[study["fd_perc_mean"].idxmin(), "subject"]
    row([f"Group (study, n={len(study)})", "", f"{mu:.1f}", f"{study['fd_perc_mean'].median():.1f}",
         f"max {study['fd_perc_mean'].max():.1f} ({study_max})", f"SD {sd:.2f}",
         f"min {study['fd_perc_mean'].min():.1f} ({study_min})", ""],
        color="black")

    fig.text(0.01, 0.02,
              f"%FD = percentage of frames with framewise displacement > {fd_label} (MRIQC `fd_perc`). "
              f"z and percentile computed within the study group only (pilots excluded). "
              f"Highlighted rows: |z| ≥ {OUTLIER_Z:g}.",
              fontsize=6.5, color="0.3", ha="left", va="bottom")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def run(out: Path, summary_tsv: Path, fd_label: str):
    agg, runs = load_summary()
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.drop(columns=["is_pilot"]).to_csv(summary_tsv, sep="\t", index=False)

    with PdfPages(out) as pdf:
        plot_ranked_bars(pdf, agg, runs, fd_label)
        plot_table(pdf, agg, fd_label)

    print(f"Wrote {out}")
    print(f"Wrote {summary_tsv}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(DEFAULT_OUT), help=f"Output PDF path (default {DEFAULT_OUT})")
    p.add_argument("--summary-tsv", default=str(DEFAULT_SUMMARY_TSV),
                   help=f"Output summary TSV path (default {DEFAULT_SUMMARY_TSV})")
    p.add_argument("--fd-threshold-label", default="0.2 mm",
                   help="Label for MRIQC's FD threshold, cosmetic only (default '0.2 mm')")
    args = p.parse_args()
    run(Path(args.out), Path(args.summary_tsv), args.fd_threshold_label)


if __name__ == "__main__":
    main()
