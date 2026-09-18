"""Matched comparison: linear (no tuning bump) vs Von Mises (tuned bump)
gabor orientation decoder.

Both decoders run leave-one-run-out with the same noise model (spherical),
lambda (0.1), and voxel-selection sweep. Only the encoding family differs:

  Von Mises : 8-function Von Mises basis set (tuned bump) — derivatives/decoding/gabor
  Linear    : signed response on cos(2x)/sin(2x), no tuning bump, closed-form
              fit — derivatives/decoding/gabor-linear

Compared across the full n_voxels sweep {0 (nested cvR²>0), 50, 100, 250,
500} on both BensonV1 ("V1") and NPCr.

One page: per-subject swarm with a group mean ± SEM diamond overlaid, per
ROI, across the n_voxels sweep.

Metric: within-run circular correlation (Jammalamadaka-Sarma, pi-periodic)
between true and posterior-mean decoded orientation, averaged across runs
per subject (check_voxel_count_sweep.metric_per_run).

Usage:
    python -m abstract_values.visualize.compare_linear_vs_vonmises_decoding
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

DERIV = Path(BIDS_FOLDER) / "derivatives" / "decoding"
DEFAULT_OUT = (Path(__file__).resolve().parents[2] / "notes" / "figures"
               / "compare_linear_vs_vonmises_decoding.pdf")

MODELS = [("Von Mises", "gabor"), ("Linear", "gabor-linear")]
MODEL_COLOUR = {"Von Mises": "#3B5BA5", "Linear": "#C97B2E"}
MASKS = [("BensonV1", "V1"), ("NPCr", "NPCr")]
NVOX_GRID = ["0", "50", "100", "250", "500"]


def load_pars(subdir: str, subject: str, mask: str, nv: str, lambd: float
              ) -> pd.DataFrame | None:
    lam_tag = f"_lambda-{lambd}" if lambd != 0.0 else ""
    fn = (DERIV / subdir / f"sub-{subject}" / "func"
          / f"sub-{subject}_mask-{mask}_nvoxels-{nv}_noise-spherical{lam_tag}_pars.tsv")
    if not fn.exists():
        return None
    return pd.read_csv(fn, sep="\t", index_col=[0, 1, 2, 3])


def discover_subjects() -> list[str]:
    subs = set()
    for _, subdir in MODELS:
        d = DERIV / subdir
        if not d.exists():
            continue
        for p in d.glob("sub-*"):
            subs.add(p.name.removeprefix("sub-"))
    return sorted(subs)


def collect(subjects: list[str], lambd: float) -> pd.DataFrame:
    rows = []
    for sub in subjects:
        for model_label, subdir in MODELS:
            for mask, mask_label in MASKS:
                for nv in NVOX_GRID:
                    df = load_pars(subdir, sub, mask, nv, lambd)
                    if df is None:
                        continue
                    df2 = df.reset_index(level=3)  # true_orientation_rad -> column
                    r = metric_per_run(df2, "gabor")
                    rows.append(dict(subject=sub, model=model_label, mask=mask_label,
                                     nvoxels=nv, r=r))
    return pd.DataFrame(rows)


def page_swarm(df: pd.DataFrame, pdf: PdfPages):
    order = _sort_nvox(NVOX_GRID)
    fig, axes = plt.subplots(1, len(MASKS), figsize=(7.25, 3.6),
                             constrained_layout=True, sharey=True)
    rng = np.random.default_rng(0)
    dx = 0.17

    for ax, (_, mask_label) in zip(axes, MASKS):
        for i, nv in enumerate(order):
            for model, off in [("Von Mises", -dx), ("Linear", dx)]:
                vals = df[(df.model == model) & (df["mask"] == mask_label)
                          & (df.nvoxels == nv)]["r"].dropna().to_numpy()
                if len(vals) == 0:
                    continue
                x = i + off + rng.uniform(-0.045, 0.045, size=len(vals))
                ax.scatter(x, vals, s=13, color=MODEL_COLOUR[model], alpha=0.5,
                          linewidth=0, zorder=2)
                mean = np.nanmean(vals)
                sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                ax.errorbar([i + off], [mean], yerr=sem, fmt="D",
                          markersize=6.5, markerfacecolor=MODEL_COLOUR[model],
                          markeredgecolor="black", markeredgewidth=1.3,
                          ecolor=MODEL_COLOUR[model], elinewidth=1.3, capsize=0,
                          zorder=5)
        ax.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([_nvox_label(nv) for nv in order], fontsize=7.5)
        ax.set_xlabel("Voxels selected")
        ax.set_title(mask_label, fontsize=10, color="0.2")

    axes[0].set_ylabel("Decoding accuracy\n(circular correlation)")
    axes[0].text(0.02, 0.97, "Von Mises", transform=axes[0].transAxes,
                color=MODEL_COLOUR["Von Mises"], fontsize=9, fontweight="medium", va="top")
    axes[0].text(0.02, 0.895, "Linear", transform=axes[0].transAxes,
                color=MODEL_COLOUR["Linear"], fontsize=9, fontweight="medium", va="top")
    fig.suptitle("Orientation decoding accuracy vs. voxel-selection criterion  "
                f"(spherical noise, λ=0.1, unsmoothed, n_subjects={df.subject.nunique()})",
                fontsize=9.5, y=1.04)
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def run(subjects: list[str] | None, lambd: float, out: Path):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No decode TSVs found under derivatives/decoding/{gabor,gabor-linear}.")
    print(f"Subjects: {subjects}")

    df = collect(subjects, lambd)
    if df.empty:
        raise SystemExit("No matching (model, mask, nvoxels, subject) combinations.")
    print(df.groupby(["mask", "nvoxels", "model"])["r"]
          .agg(["mean", "median", "count"]).round(3))

    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        page_swarm(df, pdf)
    tsv = out.with_suffix(".tsv")
    df.to_csv(tsv, sep="\t", index=False)
    print(f"\nWrote {out}\nSidecar: {tsv}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+",
                   help="Subject labels (default: discover from decode TSVs)")
    p.add_argument("--lambd", type=float, default=0.1)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, args.lambd, Path(args.out))


if __name__ == "__main__":
    main()
