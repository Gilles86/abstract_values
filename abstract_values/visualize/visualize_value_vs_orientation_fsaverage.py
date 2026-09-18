"""Where does value beat orientation? Group ΔcvR² map on fsaverage.

Per vertex, the group-mean difference in cross-validated R² between the
shared-value model (``aprf.cv``) and the shared-orientation model
(``vonmises.cv``):

    Δ = cvR²_value − cvR²_orientation

Red = value-preferring cortex, blue = orientation-preferring. Shown only where
at least one model carries clear signal (max(value, orientation) cvR² > 0,
beating the null). Both are the SHARED-across-conditions fits, so this is the
confound-free contrast (a shared model can't launder per-condition orientation
through flexibility; see model_comparison_roi.py).

The question it answers: are value-preferring voxels spatially organised (e.g.
clustered in parietal cortex / IPS) or salt-and-pepper noise? Two real
populations should be anatomically structured.

Usage:
  python -m abstract_values.visualize.visualize_value_vs_orientation_fsaverage \
      --static-png notes/figures/value_vs_orientation_dcvr2_fsaverage.png

Run in the pycortex2 env.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cortex
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from scipy.stats import norm

from abstract_values.utils.data import BIDS_FOLDER, Subject

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

PYCORTEX_FSAVG_SUBJECT = "fsaverage"
DCMAP = "RdBu_r"                       # diverging: blue = orientation, red = value
DLABEL = "Δ cvR²  (value − orientation)"


def soft_alpha(values, thr, sigma):
    return norm.cdf(values, loc=thr, scale=sigma).astype(np.float32)


def build_delta(subjects, value_model, ori_model, bids_folder, smoothed=False):
    """Group-mean Δ cvR² and clear-signal prevalence per vertex."""
    deltas, masks, used = [], [], []
    for s in subjects:
        sub = Subject(s, bids_folder=bids_folder)
        val = sub.get_encoding_surface_bilateral(value_model, "cvr2",
                                                 smoothed=smoothed)
        ori = sub.get_encoding_surface_bilateral(ori_model, "cvr2",
                                                 smoothed=smoothed)
        if val is None or ori is None or val.shape != ori.shape:
            continue
        clear = np.isfinite(val) & np.isfinite(ori) & (np.maximum(val, ori) > 0)
        deltas.append(np.where(clear, val - ori, np.nan))
        masks.append(clear)
        used.append(str(s))
    if not deltas:
        return None, None, []
    D = np.vstack(deltas)
    n = len(used)
    count = np.sum([np.isfinite(d) for d in deltas], axis=0)
    with np.errstate(invalid="ignore"):
        mean_delta = np.where(count > 0, np.nanmean(D, axis=0), np.nan)
    prop = (count / n).astype(np.float32)
    return mean_delta.astype(np.float32), prop, used


def _save_png(vtx, out, vlim):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig = cortex.quickflat.make_figure(vtx, with_curvature=True,
                                       with_colorbar=False, with_rois=False,
                                       with_labels=False)
    cax = fig.add_axes([0.36, 0.07, 0.28, 0.020])
    cb = ColorbarBase(cax, cmap=plt.get_cmap(DCMAP),
                      norm=Normalize(vmin=-vlim, vmax=vlim),
                      orientation="horizontal")
    cb.set_label(DLABEL, fontsize=9, labelpad=3)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=8, width=0.5, length=2.5)
    cb.set_ticks([-vlim, 0, vlim])
    cb.set_ticklabels([f"−{vlim:g}\norientation", "0", f"+{vlim:g}\nvalue"])
    fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved flatmap PNG → {out}")


def main(subjects, value_model, ori_model, bids_folder, min_prevalence, sigma,
         vlim, smoothed, static_png=None):
    mean_delta, prop, used = build_delta(subjects, value_model, ori_model,
                                         bids_folder, smoothed=smoothed)
    if mean_delta is None:
        raise SystemExit("No subjects with both value + orientation cvr2 surfaces.")

    sm = "smoothed" if smoothed else "unsmoothed"
    shown = mean_delta[(prop >= min_prevalence) & np.isfinite(mean_delta)]
    print(f"n={len(used)} [{', '.join(used)}]  ({sm})")
    print(f"vertices ≥{min_prevalence:.0%} clear: {float((prop >= min_prevalence).mean())*100:.2f}% of cortex")
    if shown.size:
        print(f"Δ where shown: {float((shown > 0).mean())*100:.1f}% value-preferring, "
              f"median {np.median(shown):+.3f}")

    v = cortex.Vertex(np.nan_to_num(mean_delta).astype(np.float32),
                      PYCORTEX_FSAVG_SUBJECT, vmin=-vlim, vmax=vlim, cmap=DCMAP)
    vtx = v.blend_curvature(soft_alpha(prop, min_prevalence, sigma))
    label = (f"Δ cvR² value({value_model}) − orientation({ori_model}) | "
             f"{sm} | ≥{min_prevalence:.0%} of {len(used)} subj clear")

    if static_png:
        _save_png(vtx, static_png, vlim)
    else:
        print("Launching pycortex webgl viewer...")
        cortex.webgl.show({label: vtx})


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+",
                   default=["03", "04", "05", "06", "07", "08", "09", "10",
                            "13", "14"])
    p.add_argument("--value-model", default="aprf.cv",
                   help="Shared value model cvR² (default aprf.cv).")
    p.add_argument("--ori-model", default="vonmises.cv",
                   help="Shared orientation model cvR² (default vonmises.cv).")
    p.add_argument("--min-prevalence", type=float, default=0.4)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--vlim", type=float, default=0.04,
                   help="Symmetric Δ colour limit (default 0.04).")
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--static-png", default=None)
    args = p.parse_args()
    main(args.subjects, args.value_model, args.ori_model, Path(args.bids_folder),
         args.min_prevalence, args.sigma, args.vlim, args.smoothed,
         static_png=args.static_png)
