"""Value and orientation coding per ROI, left and right, across the cohort.

For every ROI this asks two questions separately rather than making the models
compete:

    orientation gain = cvR²(best vonMises model) − cvR²(null)
    value gain       = cvR²(best aPRF model)    − cvR²(null)

averaged over the ROI's vertices, per subject, per hemisphere. The null is the
subject's own ``aprf-null.cv`` (predict the training mean), which is the right
baseline because a silent vertex scores slightly *negative* cvR², not zero.
Both families are given two models each — one tuned bell and one weighted basis
— so neither space wins on architecture; the family takes the better of its two
at each vertex.

The third panel is their difference, which is the quantity the winner maps
show, but as an effect size rather than a vote.

Averaging over all of an ROI's vertices (not only the ones with signal) keeps
this free of selection circularity: a small, real patch inside a big ROI shows
up as a small positive mean, not as a selected-vertex win.

Run in the ``pycortex2`` env.

Usage
-----
    python -m abstract_values.visualize.value_orientation_by_roi
    python -m abstract_values.visualize.value_orientation_by_roi --rois V1 NPC1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cortex
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize.group_surface_maps import (CX_FSAVERAGE,
                                                          discover_subjects)
from abstract_values.visualize.model_winner_maps import (CANDIDATES, load_stack,
                                                         surface_mask)

# Early visual -> dorsal -> IPS -> the numerosity maps, which is the order the
# argument runs in: orientation should own the first group by construction.
RETINOTOPIC = ["V1", "V2", "V3", "hV4", "LO1", "LO2", "TO1", "TO2",
               "V3b", "V3a", "IPS0", "IPS1", "IPS2", "IPS3", "SPL1", "FEF"]
NUMBER_FIELDS = ["NTO", "NPC1", "NPC2", "NPC3", "NF1", "NF2", "NINS"]

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})

HEMI_COLOUR = {"L": "#3B5BA5", "R": "#C44E52"}


def hemi_masks(deriv, names, n_vertices):
    """{(roi, hemi): boolean mask} from the atlas labels or the pycortex overlay."""
    rois = cortex.utils.get_roi_verts(CX_FSAVERAGE)
    half = n_vertices // 2
    out = {}
    for name in names:
        mask = surface_mask(deriv, name, n_vertices)
        if mask is None:
            mask = np.zeros(n_vertices, bool)
            keys = [k for k in (name, f"{name}_L", f"{name}_R") if k in rois]
            if not keys:
                print(f"  skip {name}: no atlas mask and not in the "
                      f"{CX_FSAVERAGE} overlay")
                continue
            for k in keys:
                mask[rois[k]] = True
        for hemi, sl in (("L", slice(0, half)), ("R", slice(half, n_vertices))):
            m = np.zeros(n_vertices, bool)
            m[sl] = mask[sl]
            if m.sum() < 20:
                print(f"  skip {name} {hemi}: {m.sum()} vertices")
                continue
            out[(name, hemi)] = m
    return out


def collect(deriv, subjects, names, smoothed):
    """Per subject, ROI and hemisphere: the two gains over the null."""
    masks, rows = None, []
    for s in subjects:
        stack, null, kept = load_stack(deriv, s, CANDIDATES, smoothed)
        if stack is None or len(kept) != len(CANDIDATES):
            continue
        if masks is None:
            masks = hemi_masks(deriv, names, stack.shape[1])
        ori = [i for i, l in enumerate(kept) if "vonMises" in l]
        val = [i for i, l in enumerate(kept) if i not in ori]
        best_ori = stack[ori].max(axis=0) - null
        best_val = stack[val].max(axis=0) - null
        for (roi, hemi), m in masks.items():
            rows.append(dict(subject=s, roi=roi, hemi=hemi, n_vertices=int(m.sum()),
                             orientation=float(np.mean(best_ori[m])),
                             value=float(np.mean(best_val[m])),
                             delta=float(np.mean(best_val[m] - best_ori[m]))))
    return pd.DataFrame(rows)


def panel(ax, df, column, names, ylabel, zero_line=False):
    present = [r for r in names if r in set(df.roi)]
    x = np.arange(len(present))
    for hemi, dx in (("L", -0.16), ("R", 0.16)):
        d = df[df.hemi == hemi].pivot(index="subject", columns="roi", values=column)
        mean = np.array([d[r].mean() for r in present])
        sem = np.array([d[r].std(ddof=1) / np.sqrt(d[r].notna().sum()) for r in present])
        ax.errorbar(x + dx, mean, yerr=sem, fmt="o", ms=3.2, lw=0,
                    elinewidth=0.9, color=HEMI_COLOUR[hemi], zorder=3)
    if zero_line:
        ax.axhline(0, color="0.7", lw=0.6, ls=(0, (4, 3)), zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(present) - 0.4)


def figure(df, out_pdf, names, tag, n):
    fig, axes = plt.subplots(3, 1, figsize=(max(7.25, 0.42 * len(names)), 6.4),
                             sharex=True)
    panel(axes[0], df, "orientation", names, "Orientation gain\n(cvR² − null)",
          zero_line=True)
    panel(axes[1], df, "value", names, "Value gain\n(cvR² − null)", zero_line=True)
    panel(axes[2], df, "delta", names, "Value − orientation", zero_line=True)
    # Direct labels instead of a legend, in the data's own colours.
    for hemi, dy in (("L", 0.94), ("R", 0.84)):
        axes[0].text(0.985, dy, f"{hemi} hemisphere", transform=axes[0].transAxes,
                     ha="right", va="top", fontsize=7, color=HEMI_COLOUR[hemi],
                     fontweight="bold")
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.055, 1.02, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="right")
        for side in ("left", "bottom"):
            ax.spines[side].set_position(("outward", 4))
    fig.suptitle(f"Value and orientation coding per ROI — {tag}, n={n}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--rois", nargs="+", default=RETINOTOPIC + NUMBER_FIELDS)
    p.add_argument("--out", default="notes/figures/value_orientation_by_roi.pdf")
    p.add_argument("--smoothing", default="both",
                   choices=["both", "unsmoothed", "smoothed"])
    args = p.parse_args()

    deriv = Path(args.bids_folder) / "derivatives"
    subjects = args.subjects or discover_subjects(deriv)
    smoothing = {"both": (False, True), "unsmoothed": (False,),
                 "smoothed": (True,)}[args.smoothing]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    with PdfPages(str(out)) as pdf:
        for sm in smoothing:
            df = collect(deriv, subjects, args.rois, sm)
            if df.empty:
                continue
            tag = "smoothed" if sm else "unsmoothed"
            n = df.subject.nunique()
            figure(df, pdf, args.rois, tag, n)
            frames.append(df.assign(smoothed=sm))
            print(f"\n  {tag}: n={n}")
            print(f"  {'ROI':7s} {'hemi':4s} {'orientation':>12s} {'value':>10s} "
                  f"{'delta':>10s}  {'vtx':>6s}")
            for (roi, hemi), d in df.groupby(["roi", "hemi"], sort=False):
                print(f"  {roi:7s} {hemi:4s} {d.orientation.mean():12.4f} "
                      f"{d.value.mean():10.4f} {d.delta.mean():+10.4f} "
                      f"{int(d.n_vertices.iloc[0]):6d}")
    tsv = out.with_suffix(".tsv")
    pd.concat(frames).to_csv(tsv, sep="\t", index=False)
    print(f"\nWrote {out}\nWrote {tsv}")


if __name__ == "__main__":
    main()
