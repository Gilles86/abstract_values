"""Which model tends to win where? Group winner map on fsaverage.

A per-vertex, cross-validated *winner-take-all* across three nested models:

    null  (aprf-null.cv)        — "predict the training mean"
    orientation (vonmises.cv)   — gabor orientation tuning
    value (aprf-fwhm-shift.cv)  — abstract-value aPRF, mode + FWHM shift per session

For every subject and vertex we take the argmax of the three cvR² maps. A
vertex is "signal" for that subject when orientation or value beats the null
(see ``project_cvr2_null_baseline`` — the baseline is the per-vertex null
cvR², not a flat 0). Aggregated over subjects:

  - colour = which real model wins more often, orientation (blue) ↔ value (red),
             i.e. the value share of the real winners
  - alpha  = fraction of subjects where *some* real model beats the null
             (fades in at --min-prevalence)

So grey curvature = "null tends to win" (no reliable tuning), blue = orientation
cortex, red = value cortex. This is the categorical companion to
``visualize_value_vs_orientation_fsaverage.py`` (continuous ΔcvR²).

Note: the value model only has unsmoothed cvR² on disk, so this is unsmoothed.

Usage
-----
  # interactive webgl viewer
  python -m abstract_values.visualize.visualize_model_winner_fsaverage

  # static flatmap PNG (no server)
  python -m abstract_values.visualize.visualize_model_winner_fsaverage \
      --static-png notes/figures/model_winner_fsaverage.png

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
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

PYCORTEX_FSAVG_SUBJECT = "fsaverage"
CMAP = "RdBu_r"   # diverging: blue = orientation wins, red = value wins
LABEL = "Value share of winners  (blue: orientation, red: value)"


def soft_alpha(values, thr, sigma):
    return norm.cdf(values, loc=thr, scale=sigma).astype(np.float32)


def build_winner(subjects, null_model, ori_model, value_model, bids_folder,
                 smoothed=False):
    """Per-vertex value-share of real winners and signal prevalence.

    Returns ``(value_share, signal, used)``:
      value_share — frac_value / (frac_value + frac_orientation), 0.5 where tied
                    (NaN where no subject has signal)
      signal      — fraction of subjects where orientation or value beats null
    """
    ori_wins = val_wins = n_finite = None
    used = []
    for s in subjects:
        sub = Subject(s, bids_folder=bids_folder)
        nul = sub.get_encoding_surface_bilateral(null_model, "cvr2",
                                                 smoothed=smoothed)
        ori = sub.get_encoding_surface_bilateral(ori_model, "cvr2",
                                                 smoothed=smoothed)
        val = sub.get_encoding_surface_bilateral(value_model, "cvr2",
                                                 smoothed=smoothed)
        if nul is None or ori is None or val is None:
            continue
        if not (nul.shape == ori.shape == val.shape):
            continue
        finite = np.isfinite(nul) & np.isfinite(ori) & np.isfinite(val)
        stack = np.vstack([nul, ori, val])          # 0=null, 1=ori, 2=value
        winner = np.argmax(stack, axis=0)
        ow = finite & (winner == 1)
        vw = finite & (winner == 2)
        if ori_wins is None:
            ori_wins = np.zeros_like(ori, dtype=np.float64)
            val_wins = np.zeros_like(ori, dtype=np.float64)
            n_finite = np.zeros_like(ori, dtype=np.float64)
        ori_wins += ow
        val_wins += vw
        n_finite += finite
        used.append(str(s))
    if not used:
        return None, None, []
    n = len(used)
    real = ori_wins + val_wins
    with np.errstate(invalid="ignore", divide="ignore"):
        value_share = np.where(real > 0, val_wins / real, np.nan)
    signal = (real / n).astype(np.float32)          # frac beating null
    return value_share.astype(np.float32), signal, used


def _save_png(vtx, out):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig = cortex.quickflat.make_figure(vtx, with_curvature=True,
                                       with_colorbar=False, with_rois=False,
                                       with_labels=False)
    cax = fig.add_axes([0.36, 0.07, 0.28, 0.020])
    cb = ColorbarBase(cax, cmap=plt.get_cmap(CMAP),
                      norm=Normalize(vmin=0, vmax=1),
                      orientation="horizontal")
    cb.set_label(LABEL, fontsize=9, labelpad=3)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=8, width=0.5, length=2.5)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Orientation", "Tie", "Value"])
    fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved flatmap PNG → {out}")


def main(subjects, null_model, ori_model, value_model, bids_folder,
         min_prevalence, sigma, smoothed, static_png=None, static_html=None):
    value_share, signal, used = build_winner(
        subjects, null_model, ori_model, value_model, bids_folder,
        smoothed=smoothed)
    if value_share is None:
        raise SystemExit("No subjects with all three cvr2 surfaces.")

    sm = "smoothed" if smoothed else "unsmoothed"
    shown_mask = (signal >= min_prevalence) & np.isfinite(value_share)
    shown = value_share[shown_mask]
    print(f"n={len(used)} [{', '.join(used)}]  ({sm})")
    print(f"models: null={null_model}  orientation={ori_model}  value={value_model}")
    print(f"vertices ≥{min_prevalence:.0%} subj beating null: "
          f"{float((signal >= min_prevalence).mean())*100:.2f}% of cortex")
    if shown.size:
        print(f"of those: {float((shown > 0.5).mean())*100:.1f}% value-dominant, "
              f"{float((shown < 0.5).mean())*100:.1f}% orientation-dominant, "
              f"median value-share {np.nanmedian(shown):.2f}")

    v = cortex.Vertex(np.nan_to_num(value_share, nan=0.5).astype(np.float32),
                      PYCORTEX_FSAVG_SUBJECT, vmin=0.0, vmax=1.0, cmap=CMAP)
    vtx = v.blend_curvature(soft_alpha(signal, min_prevalence, sigma))
    label = (f"Model winner | null={null_model} ori={ori_model} "
             f"val={value_model} | {sm} | ≥{min_prevalence:.0%} of {len(used)} subj")

    if static_png:
        _save_png(vtx, static_png)
    elif static_html:
        out = Path(static_html); out.mkdir(parents=True, exist_ok=True)
        cortex.webgl.make_static(str(out), {label: vtx}, title="Model winner")
        print(f"Saved static webgl bundle → {out}  (open {out}/index.html)")
    else:
        print("Launching pycortex webgl viewer...")
        cortex.webgl.show({label: vtx})


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+",
                   default=["03", "04", "05", "06", "07", "08", "09", "10",
                            "13", "14"])
    p.add_argument("--null-model", default="aprf-null.cv")
    p.add_argument("--ori-model", default="vonmises.cv")
    p.add_argument("--value-model", default="aprf-fwhm-shift.cv")
    p.add_argument("--min-prevalence", type=float, default=0.25,
                   help="Fade-in threshold on fraction of subjects beating null.")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--static-png", default=None,
                   help="Write a flatmap PNG instead of launching the viewer.")
    p.add_argument("--static-html", default=None,
                   help="Write a self-contained webgl bundle to this dir "
                        "(open index.html) instead of launching a live server.")
    args = p.parse_args()
    main(args.subjects, args.null_model, args.ori_model, args.value_model,
         Path(args.bids_folder), args.min_prevalence, args.sigma, args.smoothed,
         static_png=args.static_png, static_html=args.static_html)
