"""Plot V1 expected uncertainty (decoder SD vs orientation) as a function
of vonmises basis count. One panel per basis count, two conditions
overlaid in each. The question: does increasing the number of basis
functions sharpen the anti-cardinal pattern (dips at 0°/90°/180°) in
the SD curve?

Reads TSVs produced by ``abstract_values.experiments.v1_basis_sweep``
under ``derivatives/experiments/v1_basis_sweep/``.

Usage:
    python -m abstract_values.visualize.v1_basis_sweep
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

ROOT = Path(BIDS_FOLDER) / "derivatives" / "experiments" / "v1_basis_sweep"
DEFAULT_OUT = (Path(BIDS_FOLDER) / "derivatives" / "qa"
               / "v1_basis_sweep.pdf")
COND_COLOUR = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
TRAINED_MIN, TRAINED_MAX = 7.5, 172.5


def _load_all():
    rows = []
    for p in sorted(ROOT.glob("sub-*/ses-*/func/*_pe.tsv")):
        df = pd.read_csv(p, sep="\t")
        df["sd_E"] = np.sqrt(df["var_E"])
        df["orientation_deg"] = np.rad2deg(df["value"])
        df["decoded_deg"]      = np.rad2deg(df["mean_E"])
        df["decoded_sd_deg"]   = np.rad2deg(df["sd_E"])
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _aggregate(df, x_col, y_col, ori_grid):
    per_sub = []
    for _, g in df.groupby("subject"):
        g = g.sort_values(x_col)
        per_sub.append(np.interp(ori_grid, g[x_col].values, g[y_col].values,
                                  left=np.nan, right=np.nan))
    if not per_sub:
        return None, None, 0
    arr = np.asarray(per_sub)
    n_eff = np.maximum(np.sum(~np.isnan(arr), axis=0), 1)
    return (np.nanmean(arr, axis=0),
            np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_eff),
            arr.shape[0])


def page(df, smoothed, out):
    sub_df = df[df["smoothed"].fillna(False) == smoothed] if "smoothed" in df.columns else df
    # Subset: smoothing column may not exist; fall back to all rows.
    basis_counts = sorted(sub_df["n_basis"].unique())
    n_b = len(basis_counts)
    if n_b == 0:
        return
    fig, axes = plt.subplots(1, n_b, figsize=(3.0 * n_b + 0.5, 3.5),
                              constrained_layout=True, sharey=True)
    if n_b == 1:
        axes = [axes]

    ori_grid = np.linspace(TRAINED_MIN, TRAINED_MAX, 60)
    cardinals = (0, 90, 180)
    smooth_lbl = "smoothed" if smoothed else "unsmoothed"
    fig.suptitle(
        f"V1 expected SD vs orientation  ·  fdr05  ·  spherical noise  ·  "
        f"{smooth_lbl}  ·  varying n_basis",
        fontsize=10, y=1.04, color="0.15")

    sd_max = 0.0
    for ax, nb in zip(axes, basis_counts):
        cell = sub_df[sub_df["n_basis"] == nb]
        for cond, sub in cell.groupby("condition"):
            mean, sem, n = _aggregate(sub, "orientation_deg",
                                        "decoded_sd_deg", ori_grid)
            if mean is None: continue
            ax.plot(ori_grid, mean, color=COND_COLOUR[cond], lw=1.8,
                    label=f"{'CDF' if cond=='cdf' else 'InvCDF'}  (n={n})")
            ax.fill_between(ori_grid, mean - sem, mean + sem,
                             color=COND_COLOUR[cond], alpha=0.22,
                             linewidth=0)
            sd_max = max(sd_max, float(np.nanmax(mean + sem)))
        # Cardinal reference lines
        for c in cardinals:
            if TRAINED_MIN <= c <= TRAINED_MAX:
                ax.axvline(c, color="0.7", lw=0.5, ls=":", zorder=0)
        ax.set_xlim(TRAINED_MIN, TRAINED_MAX)
        ax.set_xticks([15, 45, 90, 135, 165])
        ax.set_xlabel("Orientation (deg)")
        ax.set_title(f"n_basis = {nb}", fontsize=9, color="0.2")
        ax.legend(loc="upper right", fontsize=7)
    axes[0].set_ylabel(r"V1 expected SD (deg)")
    for ax in axes:
        ax.set_ylim(0, sd_max * 1.15)
    sns.despine(fig=fig, offset=4)
    out.savefig(fig, bbox_inches="tight"); plt.close(fig)


def run(out):
    df = _load_all()
    if df.empty:
        raise SystemExit(f"No TSVs found under {ROOT}")
    print(f"Loaded {len(df)} rows  "
          f"({df['subject'].nunique()} subjects · "
          f"{df['n_basis'].nunique()} basis counts: "
          f"{sorted(df['n_basis'].unique())})")
    # Add a smoothed column if filenames carry _smoothed; otherwise treat all
    # rows as unsmoothed.
    df["smoothed"] = False
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        page(df, smoothed=False, out=pdf)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(Path(args.out))


if __name__ == "__main__":
    main()
