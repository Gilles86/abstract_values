"""Swarm plots: decoding accuracy vs n_voxels, orientation/value x V1/NPCr.

Companion to ``check_voxel_count_sweep.py`` (which shows the same sweep as
per-subject lines + group mean). Here each n_voxels level is a swarm of
per-subject points plus a median summary marker — easier to read the
per-subject distribution and any outliers at a glance.

2x2 grid:
    rows    = decoder   (gabor → orientation,  value → CHF)
    columns = ROI mask  (BensonV1 "V1",  NPCr)

Metric is circular correlation (Jammalamadaka–Sarma) for gabor decoding,
Pearson r for value decoding — mean within-run, averaged across runs per
subject (same aggregation as check_voxel_count_sweep.metric_per_run).

Reads ``noise-spherical`` decode outputs (the pipeline default — see
project memory: full empirical Omega over-regularizes vs spherical noise).

Usage:
    python -m abstract_values.visualize.decoding_accuracy_swarm
    python -m abstract_values.visualize.decoding_accuracy_swarm --subjects 22 23 24
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize.check_voxel_count_sweep import (
    _nvox_label, _sort_nvox, metric_per_run)

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

DERIV = Path(BIDS_FOLDER) / "derivatives" / "decoding"
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "decoding_accuracy_swarm.pdf"

DECODERS = [("gabor", "Orientation", "Circular correlation")]
DECODERS.append(("value", "Value", "Pearson r"))
ROIS = [("BensonV1", "V1"), ("NPCr", "NPCr")]
ROW_COLOR = {"gabor": "#3B5BA5", "value": "#E76F51"}


def list_nvoxels(decoder: str, subject: str, mask: str, lambd: float) -> list[str]:
    base = DERIV / decoder / f"sub-{subject}" / "func"
    if not base.exists():
        return []
    lam_tag = f"_lambda-{lambd}" if lambd != 0.0 else ""
    pat = re.compile(
        rf"sub-{subject}_mask-{mask}_nvoxels-(\d+)"
        rf"_noise-spherical{re.escape(lam_tag)}_pars\.tsv$")
    return _sort_nvox([m.group(1) for p in base.iterdir()
                       if (m := pat.match(p.name))])


def load_pars(decoder: str, subject: str, mask: str, nv: str,
              lambd: float) -> pd.DataFrame | None:
    lam_tag = f"_lambda-{lambd}" if lambd != 0.0 else ""
    fn = (DERIV / decoder / f"sub-{subject}" / "func"
          / f"sub-{subject}_mask-{mask}_nvoxels-{nv}_noise-spherical"
            f"{lam_tag}_pars.tsv")
    if not fn.exists():
        return None
    return pd.read_csv(fn, sep="\t", index_col=[0, 1, 2])


def discover_subjects() -> list[str]:
    subs = set()
    for d in ("gabor", "value"):
        if not (DERIV / d).exists():
            continue
        for p in (DERIV / d).glob("sub-*"):
            subs.add(p.name.removeprefix("sub-"))
    return sorted(subs)


def collect(subjects: list[str], decoder: str, mask: str,
            lambd: float) -> pd.DataFrame:
    rows = []
    for sub in subjects:
        for nv in list_nvoxels(decoder, sub, mask, lambd):
            df = load_pars(decoder, sub, mask, nv, lambd)
            if df is None:
                continue
            rows.append({"subject": sub, "nvoxels": nv,
                        "metric": metric_per_run(df, decoder)})
    return pd.DataFrame(rows)


def _plot_panel(ax, df: pd.DataFrame, color: str):
    if df.empty or df["metric"].isna().all():
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.5")
        ax.set_axis_off()
        return
    nv_vals = _sort_nvox(list(df["nvoxels"].unique()))
    order = nv_vals

    sns.swarmplot(data=df, x="nvoxels", y="metric", order=order,
                  ax=ax, color=color, alpha=0.45, size=3.2, zorder=2)

    medians = [df.loc[df.nvoxels == nv, "metric"].median() for nv in order]
    ax.scatter(range(len(order)), medians, marker="D", s=55,
              facecolor=color, edgecolor="black", linewidth=1.4, zorder=5)

    ax.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_nvox_label(nv) for nv in order], fontsize=8)
    ax.set_xlabel("n_voxels")


def plot(subjects: list[str], lambd: float) -> tuple[plt.Figure, pd.DataFrame]:
    data = {}
    for decoder, _, _ in DECODERS:
        for mask, _ in ROIS:
            data[(decoder, mask)] = collect(subjects, decoder, mask, lambd)

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.2), constrained_layout=True,
                             sharex="col")
    fig.suptitle(f"Decoding accuracy vs n_voxels  ·  λ={lambd}  ·  "
                f"unsmoothed  ·  n_subjects={len(subjects)}",
                fontsize=10, y=1.03)

    for i, (decoder, decoder_label, metric_label) in enumerate(DECODERS):
        # Shared y-range across the row (same metric, both ROIs)
        row_vals = pd.concat([data[(decoder, m)]["metric"] for m, _ in ROIS])
        row_vals = row_vals.dropna()
        if len(row_vals):
            lo, hi = row_vals.min(), row_vals.max()
            pad = 0.08 * (hi - lo if hi > lo else 1.0)
            ylim = (lo - pad, hi + pad)
        else:
            ylim = None
        for j, (mask, roi_label) in enumerate(ROIS):
            ax = axes[i, j]
            _plot_panel(ax, data[(decoder, mask)], ROW_COLOR[decoder])
            if ylim is not None:
                ax.set_ylim(ylim)
            if j == 0:
                ax.set_ylabel(metric_label)
            else:
                ax.set_ylabel("")
            if i == 0:
                ax.set_title(roi_label, fontsize=9, color="0.2")
            ax.text(0.03, 0.96, decoder_label if j == 0 else "",
                    transform=ax.transAxes, fontsize=8, color=ROW_COLOR[decoder],
                    fontweight="bold", va="top", ha="left")

    sns.despine(fig=fig, offset=5, trim=True)

    long_rows = []
    for (decoder, mask), df in data.items():
        d = df.copy()
        d["decoder"] = decoder
        d["mask"] = mask
        long_rows.append(d)
    return fig, pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()


def run(subjects: list[str] | None, lambd: float, out: Path):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No decode TSVs found.")
    print(f"Subjects: {subjects}")
    fig, agg = plot(subjects, lambd)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    tsv = out.with_suffix(".tsv")
    agg.to_csv(tsv, sep="\t", index=False)
    print(f"Wrote {out}\nSidecar TSV: {tsv}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+",
                   help="Subject labels (default: discover all on disk)")
    p.add_argument("--lambd", type=float, default=0.1,
                   help="Decoder regularisation λ (default 0.1)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, args.lambd, Path(args.out))


if __name__ == "__main__":
    main()
