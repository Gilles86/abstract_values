#!/usr/bin/env python3
"""
Cross-subject surface model comparison: how much cortex each encoding
model explains out-of-sample. Counts fsaverage vertices with cvR2 > thr
(default 0) per (subject, model), from the surface-sampled cvR2 maps that
`sample_r2_to_surface.py --desc cvr2` writes:

    derivatives/encoding_models/<model>/sub-XX/func/
        sub-XX_task-abstractvalue_hemi-{L,R}_space-fsaverage_desc-cvr2{_smoothed}_pe.func.gii

Two modes (aggregate on the cluster where the giis live, plot locally):

  --aggregate : read all giis, write a small TSV
                (subject, model, n_pos, n_total, frac_pos).
  --plot      : bar of n_pos / frac_pos per model (mean +/- SEM across
                subjects, per-subject points overlaid) from the TSV.

Usage
-----
  # cluster
  python -m abstract_values.visualize.cvr2_surface_extent --aggregate \
      --models aprf.cv vonmises.cv aprf-weighted.cv aprf-shift.cv \
      --tsv notes/data/cvr2_surface_extent.tsv
  # local
  python -m abstract_values.visualize.cvr2_surface_extent --plot \
      --tsv notes/data/cvr2_surface_extent.tsv --out notes/figures/cvr2_surface_extent.pdf
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from abstract_values.utils.data import BIDS_FOLDER

DEFAULT_MODELS = ["aprf.cv", "vonmises.cv", "aprf-weighted.cv", "aprf-shift.cv"]


def aggregate(models, thr, smoothed, bids_folder, tsv):
    import nibabel as nib
    sm = "_smoothed" if smoothed else ""
    enc = Path(bids_folder) / "derivatives" / "encoding_models"
    rows = []
    for model in models:
        for sub_dir in sorted((enc / model).glob("sub-*")):
            subject = sub_dir.name.removeprefix("sub-")
            vals = []
            for hemi in ("L", "R"):
                gii = (sub_dir / "func" /
                       f"sub-{subject}_task-abstractvalue_hemi-{hemi}"
                       f"_space-fsaverage_desc-cvr2{sm}_pe.func.gii")
                if gii.exists():
                    vals.append(nib.load(str(gii)).darrays[0].data.ravel())
            if not vals:
                continue
            v = np.concatenate(vals)
            finite = np.isfinite(v) & (v != 0)        # 0 = medial wall / unsampled
            n_total = int(finite.sum())
            n_pos = int((v[finite] > thr).sum())
            rows.append(dict(subject=subject, model=model.replace(".cv", ""),
                             n_pos=n_pos, n_total=n_total,
                             frac_pos=n_pos / max(n_total, 1)))
            print(f"  {model:<20} sub-{subject:<6} "
                  f"n>{thr}={n_pos:>6} / {n_total} ({n_pos/max(n_total,1):.1%})")
    df = pd.DataFrame(rows)
    Path(tsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tsv, sep="\t", index=False)
    print(f"\nwrote {tsv}  ({df['subject'].nunique()} subjects, "
          f"{df['model'].nunique()} models)")


def plot(tsv, out, ycol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = pd.read_csv(tsv, sep="\t")
    order = [m for m in ["aprf", "vonmises", "aprf-weighted", "aprf-shift",
                         "aprf-session-shift"] if m in df["model"].unique()]
    order += [m for m in df["model"].unique() if m not in order]
    lab = {"n_pos": "Vertices with cvR2 > 0",
           "frac_pos": "Fraction of cortex with cvR2 > 0"}[ycol]
    with sns.plotting_context("talk"), sns.axes_style("ticks"):
        fig, ax = plt.subplots(figsize=(1.1 * len(order) + 2, 4),
                               constrained_layout=True)
        sns.barplot(df, x="model", y=ycol, order=order, errorbar="se",
                    color="#4C72B0", ax=ax)
        sns.stripplot(df, x="model", y=ycol, order=order, color="0.25",
                      size=4, alpha=0.6, ax=ax)
        ax.set_xlabel(""); ax.set_ylabel(lab)
        if ycol == "frac_pos":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(f"Out-of-sample cortical extent per model "
                     f"(n={df['subject'].nunique()})", fontsize=11)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        sns.despine(ax=ax, offset=4, trim=True)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--thr", type=float, default=0.0)
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--tsv", default="notes/data/cvr2_surface_extent.tsv")
    p.add_argument("--out", default="notes/figures/cvr2_surface_extent.pdf")
    p.add_argument("--ycol", default="frac_pos", choices=["n_pos", "frac_pos"])
    args = p.parse_args()
    if args.aggregate:
        aggregate(args.models, args.thr, args.smoothed, args.bids_folder, args.tsv)
    if args.plot:
        plot(args.tsv, args.out, args.ycol)
    if not (args.aggregate or args.plot):
        p.error("pass --aggregate (cluster) and/or --plot (local)")


if __name__ == "__main__":
    main()
