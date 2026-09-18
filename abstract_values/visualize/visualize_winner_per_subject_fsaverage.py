"""Flip through participants: per-subject value-vs-orientation map on fsaverage.

One fsaverage surface per subject, all loaded into a single pycortex webgl
viewer so you can flip between participants with the dataset dropdown. Each
surface shows, per vertex, the cross-validated contrast

    Δ = cvR²_value − cvR²_orientation        (value = aprf-fwhm-shift.cv,
                                              orientation = vonmises.cv)

red = value-preferring, blue = orientation-preferring, shown (alpha) where the
better of the two beats the per-vertex null (aprf-null.cv; see
``project_cvr2_null_baseline``). This is the per-subject companion to the group
maps ``visualize_value_vs_orientation_fsaverage.py`` /
``visualize_model_winner_fsaverage.py`` — use it to eyeball how consistent the
orientation-posterior / value-parietal split is across people.

Note: the value model only has unsmoothed cvR² on disk, so this is unsmoothed.

Usage
-----
  # interactive webgl viewer (live server, flip through subjects in the browser)
  python -m abstract_values.visualize.visualize_winner_per_subject_fsaverage

  # self-contained static webgl bundle (no server; open index.html)
  python -m abstract_values.visualize.visualize_winner_per_subject_fsaverage \
      --static-html notes/figures/winner_per_subject_webgl

Run in the pycortex2 env.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cortex
import numpy as np
from scipy.stats import norm

from abstract_values.utils.data import BIDS_FOLDER, Subject

PYCORTEX_FSAVG_SUBJECT = "fsaverage"
CMAP = "RdBu_r"   # red = value-preferring, blue = orientation-preferring


def soft_alpha(values, thr, sigma):
    return norm.cdf(values, loc=thr, scale=sigma).astype(np.float32)


def build_subject_vertex(subject, null_model, ori_model, value_model,
                         bids_folder, vlim, sigma, smoothed=False):
    """Per-subject Δ cvR² Vertex (value − orientation), faded by beating null.

    Returns a ``cortex.Vertex`` or ``None`` if any surface is missing.
    """
    sub = Subject(subject, bids_folder=bids_folder)
    nul = sub.get_encoding_surface_bilateral(null_model, "cvr2", smoothed=smoothed)
    ori = sub.get_encoding_surface_bilateral(ori_model, "cvr2", smoothed=smoothed)
    val = sub.get_encoding_surface_bilateral(value_model, "cvr2", smoothed=smoothed)
    if nul is None or ori is None or val is None:
        return None
    if not (nul.shape == ori.shape == val.shape):
        return None
    finite = np.isfinite(nul) & np.isfinite(ori) & np.isfinite(val)
    delta = np.where(finite, val - ori, 0.0).astype(np.float32)
    # alpha = how much the better real model beats the per-vertex null
    margin = np.where(finite, np.maximum(ori, val) - nul, -np.inf)
    alpha = soft_alpha(margin, thr=0.0, sigma=sigma)
    v = cortex.Vertex(delta, PYCORTEX_FSAVG_SUBJECT,
                      vmin=-vlim, vmax=vlim, cmap=CMAP)
    return v.blend_curvature(alpha)


def main(subjects, null_model, ori_model, value_model, bids_folder,
         vlim, sigma, smoothed, static_html=None):
    sm = "smoothed" if smoothed else "unsmoothed"
    viewer = {}
    for s in subjects:
        vtx = build_subject_vertex(s, null_model, ori_model, value_model,
                                   bids_folder, vlim, sigma, smoothed=smoothed)
        if vtx is None:
            print(f"sub-{s}: missing a cvr2 surface — skipping")
            continue
        viewer[f"sub-{s}"] = vtx
        print(f"sub-{s}: added")
    if not viewer:
        raise SystemExit("No subjects with all three cvr2 surfaces.")
    print(f"n={len(viewer)} subjects ({sm}); "
          f"Δ = value({value_model}) − orientation({ori_model}); ±{vlim:g}")

    if static_html:
        out = Path(static_html); out.mkdir(parents=True, exist_ok=True)
        cortex.webgl.make_static(str(out), viewer,
                                 title="Value vs orientation — per subject")
        print(f"Saved static webgl bundle → {out}  (open {out}/index.html)")
    else:
        print("Launching pycortex webgl viewer (flip subjects via the dropdown)...")
        cortex.webgl.show(viewer)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+",
                   default=["03", "04", "05", "06", "07", "08", "09", "10",
                            "13", "14"])
    p.add_argument("--null-model", default="aprf-null.cv")
    p.add_argument("--ori-model", default="vonmises.cv")
    p.add_argument("--value-model", default="aprf-fwhm-shift.cv")
    p.add_argument("--vlim", type=float, default=0.05,
                   help="Symmetric Δ cvR² colour limit (default 0.05).")
    p.add_argument("--sigma", type=float, default=0.01,
                   help="Soft-alpha width on the beats-null margin (cvR² units).")
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--static-html", default=None,
                   help="Write a self-contained webgl bundle to this dir "
                        "(open index.html) instead of launching a live server.")
    args = p.parse_args()
    main(args.subjects, args.null_model, args.ori_model, args.value_model,
         Path(args.bids_folder), args.vlim, args.sigma, args.smoothed,
         static_html=args.static_html)
