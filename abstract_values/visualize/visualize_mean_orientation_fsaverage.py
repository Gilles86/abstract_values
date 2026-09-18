"""Group preferred-ORIENTATION map on fsaverage cortex via pycortex.

Per vertex, the circular-mean preferred orientation (0–180°) of the von Mises
encoding model across subjects — shown only where the model reliably beats the
null (cvR²-vs-null prevalence; see ``project_cvr2_null_baseline``).

Orientation is π-periodic, so everything is done on the doubled-angle unit
vector (cos2θ, sin2θ) written by
``encoding_models/compute_preferred_orientation.py``:

  - per subject, vertex: (cos2θ, sin2θ), masked to where the model beats null
  - group circular mean: C̄ = mean cos2θ, S̄ = mean sin2θ over winning subjects
  - preferred orientation θ = ½·atan2(S̄, C̄)  (mapped to [0, 180°))
  - alpha fades in with the win-prevalence

** Important caveat (radial bias). ** In early visual cortex a preferred-
orientation map is expected to reflect the *radial bias* — voxels prefer
orientations radial to fixation — so a V1 map largely recapitulates the
retinotopic polar-angle map rather than an abstract orientation code. Read
this as a retinotopy/stimulus reference, not a novel result; compare against a
Benson polar-angle map to quantify it.

A cyclic colormap (default 'hsv') is used because orientation wraps.

Usage
-----
  python -m abstract_values.visualize.visualize_mean_orientation_fsaverage
  python -m abstract_values.visualize.visualize_mean_orientation_fsaverage \
      --static-png notes/figures/group_vonMises-preferred-orientation_fsaverage.png

Run in the pycortex2 env.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import cortex
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from scipy.stats import norm

from abstract_values.utils.data import (BIDS_FOLDER, Subject, cvr2_signal,
                                        DEFAULT_NULL_MODEL)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

PYCORTEX_FSAVG_SUBJECT = "fsaverage"
ORI_CMAP = "hsv"                       # cyclic — orientation wraps at 180°
ORI_LABEL = "Preferred orientation (°)"


def soft_alpha(values, thr, sigma):
    return norm.cdf(values, loc=thr, scale=sigma).astype(np.float32)


def build_group_orientation(subjects, ori_model, cv_model, baseline_model,
                            bids_folder, smoothed=False):
    """Circular-mean preferred orientation (degrees, 0–180) and win-prevalence.

    Returns ``(theta_deg, prop, R, used)`` where R is the cross-subject
    resultant length (orientation consistency, 0–1)."""
    cos_stack, sin_stack, used = [], [], []
    for s in subjects:
        sub = Subject(s, bids_folder=bids_folder)
        c = sub.get_encoding_surface_bilateral(ori_model, "orient_cos",
                                               smoothed=smoothed)
        si = sub.get_encoding_surface_bilateral(ori_model, "orient_sin",
                                                smoothed=smoothed)
        if c is None or si is None:
            continue
        sig, _ = cvr2_signal(s, cv_model, baseline_model=baseline_model,
                             smoothed=smoothed, bids_folder=bids_folder)
        if sig is None or sig.shape != c.shape:
            continue
        m = sig & np.isfinite(c) & np.isfinite(si)
        cos_stack.append(np.where(m, c, np.nan))
        sin_stack.append(np.where(m, si, np.nan))
        used.append(str(s))

    if not cos_stack:
        return None, None, None, []

    # Vertices never sampled in any subject are all-NaN → nanmean warns
    # ("Mean of empty slice"); the result (NaN) is handled downstream.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        C = np.nanmean(np.vstack(cos_stack), axis=0)   # circular mean components
        S = np.nanmean(np.vstack(sin_stack), axis=0)
    n = len(used)
    count = np.sum([np.isfinite(a) for a in cos_stack], axis=0)
    prop = (count / n).astype(np.float32)
    R = np.sqrt(np.nan_to_num(C) ** 2 + np.nan_to_num(S) ** 2).astype(np.float32)
    theta = 0.5 * np.arctan2(np.nan_to_num(S), np.nan_to_num(C))   # radians, half-angle
    theta_deg = (np.rad2deg(theta) % 180.0).astype(np.float32)
    return theta_deg, prop, R, used


def _save_png(vtx, out, *, cmap=ORI_CMAP, label=ORI_LABEL):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = cortex.quickflat.make_figure(vtx, with_curvature=True,
                                       with_colorbar=False, with_rois=False,
                                       with_labels=False)
    cax = fig.add_axes([0.36, 0.07, 0.28, 0.020])
    cb = ColorbarBase(cax, cmap=plt.get_cmap(cmap),
                      norm=Normalize(vmin=0, vmax=180),
                      orientation="horizontal")
    cb.set_label(label, fontsize=9, labelpad=3)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=8, width=0.5, length=2.5)
    cb.set_ticks([0, 45, 90, 135, 180])
    fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved flatmap PNG → {out}")


def main(subjects, ori_model, cv_model, baseline_model, bids_folder,
         min_prevalence, sigma, smoothed, cmap=ORI_CMAP, static_png=None):
    theta_deg, prop, R, used = build_group_orientation(
        subjects, ori_model, cv_model, baseline_model, bids_folder,
        smoothed=smoothed)
    if theta_deg is None:
        raise SystemExit("No subjects with both orient_cos/sin + cvr2 surfaces. "
                         "Run compute_preferred_orientation.py + "
                         "sample_r2_to_surface.py --desc orient_cos/orient_sin first.")

    sm = "smoothed" if smoothed else "unsmoothed"
    surviving = float((prop >= min_prevalence).mean()) * 100
    print(f"n={len(used)} [{', '.join(used)}]  ({sm})")
    print(f"vertices ≥{min_prevalence:.0%} prevalence: {surviving:.2f}% of cortex")

    v = cortex.Vertex(np.nan_to_num(theta_deg).astype(np.float32),
                      PYCORTEX_FSAVG_SUBJECT, vmin=0, vmax=180, cmap=cmap)
    vtx = v.blend_curvature(soft_alpha(prop, min_prevalence, sigma))
    label = (f"{ori_model} preferred orientation (°) — group circular mean | "
             f"{sm} | ≥{min_prevalence:.0%} of {len(used)} subj beat null")

    if static_png:
        _save_png(vtx, static_png, cmap=cmap)
    else:
        print("Launching pycortex webgl viewer...")
        cortex.webgl.show({label: vtx})


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+",
                   default=["03", "04", "05", "06", "07", "08", "09", "10",
                            "13", "14"])
    p.add_argument("--ori-model", default="vonmises",
                   help="Model dir with desc-orient_cos/sin (default vonmises).")
    p.add_argument("--cv-model", default="vonmises.cv",
                   help="Model dir with cvR² for the signal mask (default vonmises.cv).")
    p.add_argument("--baseline-model", default=DEFAULT_NULL_MODEL,
                   help=f"Per-vertex cvR² null reference (default {DEFAULT_NULL_MODEL!r}; "
                        "'none' → cvR² > 0).")
    p.add_argument("--min-prevalence", type=float, default=0.4)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--cmap", default=ORI_CMAP,
                   help=f"Cyclic colormap for orientation (default {ORI_CMAP!r}; "
                        "try 'twilight' or 'twilight_shifted').")
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--static-png", default=None)
    args = p.parse_args()

    baseline = (None if str(args.baseline_model).lower() == "none"
                else args.baseline_model)
    main(args.subjects, args.ori_model, args.cv_model, baseline,
         Path(args.bids_folder), args.min_prevalence, args.sigma,
         args.smoothed, cmap=args.cmap, static_png=args.static_png)
