"""Head-motion QC across the cohort, from fmriprep's confounds.

Two modes, split along the usual data-size boundary:

    # cluster: walk every subject's confounds and reduce to one row per run
    BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue \
        python -m abstract_values.visualize.check_motion \
            --summary-tsv notes/data/motion_summary.tsv

    # local: read that TSV and draw the cohort figure
    python -m abstract_values.visualize.check_motion \
        --tsv notes/data/motion_summary.tsv --highlight 28 \
        --out notes/figures/motion_summary.pdf

Metrics per run (framewise displacement, Power et al. 2012, as fmriprep
computes it in mm):

    mean_fd / median_fd  central tendency — the "is this subject restless"
                         number. The usual exclusion talk is about mean FD.
    max_fd               single worst jump.
    pct_fd_gt_0p5        % of volumes above 0.5 mm — the fraction of the run a
                         scrubbing pipeline would throw away.
    n_spikes_gt_1mm      volumes above 1 mm; these are the ones that actually
                         break a GLM, since a 1 mm shift is ~half a voxel.
    mean_dvars           intensity-change companion; motion that doesn't show
                         up in FD (e.g. spin-history) often shows up here.
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

from abstract_values.utils.data import BIDS_FOLDER, Subject

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

HIGHLIGHT_COLOUR = "#E76F51"
COHORT_COLOUR = "#3B5BA5"


# ── aggregation (cluster side) ───────────────────────────────────────────────

def discover_subjects(bids_folder):
    """Subject labels with fmriprep output, study subjects before pilots."""
    d = Path(bids_folder) / "derivatives" / "fmriprep"
    subs = [p.name.removeprefix("sub-") for p in sorted(d.glob("sub-*"))
            if p.is_dir() and (p / ".fmriprep_done").exists()]
    return sorted(subs, key=lambda s: (0 if s[0].isdigit() else 1, s))


def summarise(bids_folder, subjects=None):
    bids_folder = Path(bids_folder)
    subjects = subjects or discover_subjects(bids_folder)
    rows = []
    for label in subjects:
        sub = Subject(label, bids_folder=bids_folder)
        for session in sub.get_sessions():
            for run in sub.get_runs(session):
                fn = (bids_folder / "derivatives" / "fmriprep" /
                      f"sub-{label}" / f"ses-{session}" / "func" /
                      f"sub-{label}_ses-{session}_task-abstractvalue_"
                      f"run-{run}_desc-confounds_timeseries.tsv")
                if not fn.exists():
                    print(f"  missing confounds: {fn.name}")
                    continue
                df = pd.read_csv(fn, sep="\t")
                # fmriprep leaves the first volume's FD/DVARS as NaN (no
                # preceding volume to difference against) — drop, don't fill.
                fd = pd.to_numeric(df["framewise_displacement"],
                                   errors="coerce").dropna().to_numpy()
                dvars = pd.to_numeric(df.get("dvars", pd.Series(dtype=float)),
                                      errors="coerce").dropna().to_numpy()
                if fd.size == 0:
                    continue
                rows.append(dict(
                    subject=label, session=session, run=run,
                    n_vols=len(df),
                    mean_fd=fd.mean(), median_fd=np.median(fd), max_fd=fd.max(),
                    pct_fd_gt_0p5=100.0 * (fd > 0.5).mean(),
                    n_spikes_gt_1mm=int((fd > 1.0).sum()),
                    mean_dvars=dvars.mean() if dvars.size else np.nan,
                ))
        print(f"  sub-{label}: {sum(r['subject'] == label for r in rows)} runs")
    return pd.DataFrame(rows)


# ── plotting (local side) ────────────────────────────────────────────────────

METRICS = [
    ("mean_fd", "Mean FD (mm)", None),
    ("pct_fd_gt_0p5", "Volumes > 0.5 mm (%)", None),
    ("n_spikes_gt_1mm", "Volumes > 1 mm (count)", None),
    ("max_fd", "Max FD (mm)", None),
]


def plot(df, highlight, out):
    """One panel per metric: subject means, highlighted subject called out."""
    per_sub = (df.groupby("subject")
                 .agg(**{m: (m, "mean") for m, _, _ in METRICS})
                 .reset_index())
    # study subjects numerically, pilots last
    per_sub["_key"] = per_sub["subject"].map(
        lambda s: (0, int(s)) if s.isdigit() else (1, 0))
    per_sub = per_sub.sort_values("_key").drop(columns="_key")

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(9, 2.1 * len(METRICS)),
                             sharex=True, constrained_layout=True)
    x = np.arange(len(per_sub))
    is_hl = (per_sub["subject"] == str(highlight)).to_numpy()

    for ax, (metric, label, _) in zip(axes, METRICS):
        vals = per_sub[metric].to_numpy(dtype=float)
        colours = np.where(is_hl, HIGHLIGHT_COLOUR, COHORT_COLOUR)
        ax.bar(x, vals, color=colours, width=0.72, linewidth=0)
        # cohort reference excluding the highlighted subject, so the
        # comparison isn't diluted by the subject being judged
        ref = vals[~is_hl]
        ax.axhline(np.nanmedian(ref), color="0.35", lw=0.8, ls="--", zorder=3)
        ax.axhline(np.nanpercentile(ref, 90), color="0.6", lw=0.8, ls=":", zorder=3)
        ax.set_ylabel(label)
        if is_hl.any():
            i = int(np.flatnonzero(is_hl)[0])
            ax.annotate(f"{vals[i]:.2f}", (x[i], vals[i]),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8, color=HIGHLIGHT_COLOUR)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(per_sub["subject"], rotation=90)
    axes[-1].set_xlabel("Subject")
    axes[0].set_title(
        "Head motion per subject (run means)  ·  dashed = cohort median, "
        "dotted = 90th percentile", fontsize=9, color="0.2", loc="left")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return per_sub


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--summary-tsv", default=None,
                   help="aggregate from confounds and write this TSV (cluster)")
    p.add_argument("--tsv", default=None,
                   help="read this TSV instead of aggregating (local)")
    p.add_argument("--highlight", default=None, help="subject label to call out")
    p.add_argument("--out", default=None, help="figure path (implies plotting)")
    args = p.parse_args()

    if args.tsv:
        df = pd.read_csv(args.tsv, sep="\t", dtype={"subject": str})
    else:
        df = summarise(args.bids_folder, args.subjects)
        if args.summary_tsv:
            out = Path(args.summary_tsv)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, sep="\t", index=False)
            print(f"Wrote {out}  ({len(df)} runs)")

    if args.out:
        plot(df, args.highlight, args.out)


if __name__ == "__main__":
    main()
