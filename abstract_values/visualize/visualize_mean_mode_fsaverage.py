"""Group preferred-value (aPRF mode) map on fsaverage cortex via pycortex.

What this shows
---------------
Per vertex, the **mean preferred value (CHF)** of the aPRF across subjects —
but only averaged over, and only displayed where, the encoding model
*reliably beats the null*. "Reliably" = the cvR²-vs-null prevalence built in
``utils.data.cvr2_prevalence`` / ``cvr2_signal`` (a voxel "wins" for a subject
when its cross-validated R² exceeds the per-vertex null-model cvR²; see
``project_cvr2_null_baseline``). So:

  - colour  = mean aPRF mode (CHF), averaged over the subjects that win there
  - alpha   = fraction of subjects that win (fades in at --min-prevalence)

This couples the two halves of the analysis: the preferred-value map is shown
exactly where there is consistent cross-validated signal, not smeared across
noise.

Usage
-----
  # interactive webgl viewer
  python -m abstract_values.visualize.visualize_mean_mode_fsaverage

  # static flatmap PNG (no server)
  python -m abstract_values.visualize.visualize_mean_mode_fsaverage \
      --static-png notes/figures/group_mode_fsaverage.png

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

from abstract_values.utils.data import (BIDS_FOLDER, Subject, cvr2_signal,
                                        DEFAULT_NULL_MODEL)

# Restrained house style for the colorbar text.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

PYCORTEX_FSAVG_SUBJECT = "fsaverage"

# Presented objective value range (CHF) — colour scale bounds.
MODE_VMIN, MODE_VMAX = 2.5, 41.5
MODE_CMAP = "nipy_spectral"
VALUE_LABEL = "Preferred value (CHF)"
# Plausible-mode band for the averaging filter: drop pathological fits whose
# mode lands far outside the stimulus range, which would corrupt the mean.
PLAUSIBLE_LO, PLAUSIBLE_HI = 0.0, 60.0


def soft_alpha(values, thr, sigma):
    """Gaussian-CDF centred on thr — smooth alpha transition."""
    return norm.cdf(values, loc=thr, scale=sigma).astype(np.float32)


def smooth_label(smoothed):
    return "smoothed" if smoothed else "unsmoothed"


def load_subject_maps(subjects, mode_model, cv_model, baseline_model,
                      bids_folder, smoothed=False):
    """Load per-subject (mode, win-mask) on fsaverage.

    Returns ``(modes, masks, used)`` as lists aligned by subject. ``modes`` are
    the full-fit aPRF preferred values (CHF); ``masks`` are boolean per-vertex
    "this model beats the null" (see :func:`cvr2_signal`).
    """
    modes, masks, used = [], [], []
    no_mode, no_sig = [], []
    for s in subjects:
        sub = Subject(s, bids_folder=bids_folder)
        mode = sub.get_encoding_surface_bilateral(mode_model, "mode",
                                                  smoothed=smoothed)
        if mode is None:
            no_mode.append(str(s))
            continue
        sig, _ = cvr2_signal(s, cv_model, baseline_model=baseline_model,
                             smoothed=smoothed, bids_folder=bids_folder)
        if sig is None or sig.shape != mode.shape:
            no_sig.append(str(s))
            continue
        modes.append(mode)
        masks.append(sig)
        used.append(str(s))

    sm = smooth_label(smoothed)
    if no_mode:
        print(f"  [{sm}] missing {mode_model} fsaverage mode surface for: "
              f"{', '.join(no_mode)}  "
              f"(run: sample_aprf_to_surface.py <sub> --session 1"
              f"{' --smoothed' if smoothed else ''})")
    if no_sig:
        print(f"  [{sm}] missing {cv_model}/{baseline_model} fsaverage cvr2 for: "
              f"{', '.join(no_sig)}")
    return modes, masks, used


def build_group_mode(subjects, mode_model, cv_model, baseline_model, bids_folder,
                     smoothed=False):
    """Per-vertex signal-weighted mean aPRF mode and win-prevalence.

    Returns ``(mean_mode, prop, used)``:
      mean_mode : (n_vert,) mean mode over winning subjects (NaN where none)
      prop      : (n_vert,) fraction of subjects that win there
      used      : list of subject labels with both maps present
    """
    modes, masks, used = load_subject_maps(
        subjects, mode_model, cv_model, baseline_model, bids_folder,
        smoothed=smoothed)
    if not modes:
        return None, None, []

    modes = np.vstack(modes)                          # (n, V)
    masks = np.vstack(masks)                           # (n, V) bool
    valid = (masks & np.isfinite(modes)
             & (modes >= PLAUSIBLE_LO) & (modes <= PLAUSIBLE_HI))
    count = valid.sum(axis=0)
    summ = np.where(valid, modes, 0.0).sum(axis=0)
    mean_mode = np.where(count > 0, summ / np.maximum(count, 1), np.nan)
    prop = (count / masks.shape[0]).astype(np.float32)
    return mean_mode.astype(np.float32), prop, used


def make_vertex(values, alpha, cmap=MODE_CMAP):
    v = cortex.Vertex(np.nan_to_num(values).astype(np.float32),
                      PYCORTEX_FSAVG_SUBJECT,
                      vmin=MODE_VMIN, vmax=MODE_VMAX, cmap=cmap)
    return v.blend_curvature(alpha)


def _save_png(vtx, out, *, cmap=MODE_CMAP, vmin=MODE_VMIN, vmax=MODE_VMAX,
              label=VALUE_LABEL):
    """Render the flatmap and draw a colorbar that actually matches the cmap.

    pycortex's ``with_colorbar=True`` cannot be trusted here: ``blend_curvature``
    returns an RGB vertex with no scalar cmap/vmin/vmax, so the auto colorbar is
    meaningless. We render with ``with_colorbar=False`` and add our own
    ``ColorbarBase`` keyed to the real ``cmap`` / ``vmin`` / ``vmax``.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = cortex.quickflat.make_figure(vtx, with_curvature=True,
                                       with_colorbar=False, with_rois=False,
                                       with_labels=False)
    # Horizontal colorbar, bottom-centre, thin and unobtrusive.
    cax = fig.add_axes([0.36, 0.07, 0.28, 0.020])
    cb = ColorbarBase(cax, cmap=plt.get_cmap(cmap),
                      norm=Normalize(vmin=vmin, vmax=vmax),
                      orientation="horizontal")
    cb.set_label(label, fontsize=9, labelpad=3)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=8, width=0.5, length=2.5)
    cb.set_ticks(np.linspace(vmin, vmax, 5))
    fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved flatmap PNG → {out}")


def run_group(subjects, mode_model, cv_model, baseline_model, bids_folder,
              min_prevalence, sigma, smoothed, static_png=None, cmap=MODE_CMAP):
    mean_mode, prop, used = build_group_mode(
        subjects, mode_model, cv_model, baseline_model, bids_folder,
        smoothed=smoothed)
    if mean_mode is None:
        raise SystemExit("No subjects with both mode + cvr2 surfaces.")

    sm = smooth_label(smoothed)
    surviving = float((prop >= min_prevalence).mean()) * 100
    shown = mean_mode[prop >= min_prevalence]
    shown = shown[np.isfinite(shown)]
    print(f"n={len(used)} [{', '.join(used)}]  ({sm})")
    print(f"vertices ≥{min_prevalence:.0%} prevalence: {surviving:.2f}% of cortex")
    if shown.size:
        print(f"preferred value where shown: median {np.median(shown):.1f} CHF, "
              f"range [{np.percentile(shown,5):.1f}, {np.percentile(shown,95):.1f}] CHF")

    vtx = make_vertex(mean_mode, soft_alpha(prop, min_prevalence, sigma), cmap=cmap)
    # Explicit dataset name: model, what it is, smoothing, threshold, n.
    label = (f"{mode_model} PRF preferred-value (mode, CHF) — group mean | "
             f"{sm} | ≥{min_prevalence:.0%} of {len(used)} subj beat null")

    if static_png:
        _save_png(vtx, static_png, cmap=cmap)
    else:
        print("Launching pycortex webgl viewer...")
        cortex.webgl.show({label: vtx})


def run_per_subject(subjects, mode_model, cv_model, baseline_model, bids_folder,
                    sigma, smoothed, static_png=None, cmap=MODE_CMAP):
    """One pycortex surface per participant: that subject's own aPRF preferred
    value on fsaverage, alpha-masked by where its model beats the null."""
    modes, masks, used = load_subject_maps(
        subjects, mode_model, cv_model, baseline_model, bids_folder,
        smoothed=smoothed)
    if not modes:
        raise SystemExit("No subjects with both mode + cvr2 surfaces.")

    sm = smooth_label(smoothed)
    ds = {}
    for s, mode, mask in zip(used, modes, masks):
        valid = (mask & np.isfinite(mode)
                 & (mode >= PLAUSIBLE_LO) & (mode <= PLAUSIBLE_HI))
        disp = np.where(valid, mode, np.nan)
        # Soft alpha off the boolean win-mask (0/1) for a clean edge.
        vtx = make_vertex(disp, soft_alpha(valid.astype(np.float32), 0.5, sigma),
                          cmap=cmap)
        # Explicit per-subject name.
        label = f"sub-{s} | {mode_model} PRF preferred-value (mode, CHF) | {sm}"
        ds[label] = vtx
        if static_png:
            outdir = Path(static_png)
            _save_png(vtx, outdir / f"sub-{s}_{mode_model}-PRF-mode_"
                                    f"fsaverage_{sm}.png", cmap=cmap)
    print(f"per-subject ({sm}): n={len(used)} [{', '.join(used)}]")

    if not static_png:
        print("Launching pycortex webgl viewer (one surface per participant)...")
        cortex.webgl.show(ds)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+",
                   default=["03", "04", "05", "06", "07", "08", "09", "10",
                            "13", "14"],
                   help="Subject labels (default: the 10 study subjects).")
    p.add_argument("--mode-model", default="aprf",
                   help="Model dir holding the full-fit desc-mode map (default aprf).")
    p.add_argument("--cv-model", default="aprf.cv",
                   help="Model dir holding the cvR² map for the signal mask "
                        "(default aprf.cv).")
    p.add_argument("--baseline-model", default=DEFAULT_NULL_MODEL,
                   help=f"Per-vertex cvR² null reference (default {DEFAULT_NULL_MODEL!r}; "
                        "'none' → compare cvR² > 0).")
    p.add_argument("--min-prevalence", type=float, default=0.4,
                   help="Fraction of subjects that must win for a vertex to show "
                        "(default 0.4).")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Gaussian-CDF alpha transition width on the prevalence "
                        "scale (default 0.1).")
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--cmap", default=MODE_CMAP,
                   help=f"Colormap for preferred value (default {MODE_CMAP!r}; "
                        "try 'turbo' or 'viridis').")
    p.add_argument("--per-subject", action="store_true",
                   help="One surface per participant (each subject's own aPRF "
                        "preferred value on fsaverage) instead of the group "
                        "mean. With --static-png, treats it as a directory and "
                        "writes one PNG per subject.")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--static-png", default=None,
                   help="If set, save static flatmap PNG(s) instead of the "
                        "webgl viewer. For --per-subject, this is a directory.")
    args = p.parse_args()

    baseline = (None if str(args.baseline_model).lower() == "none"
                else args.baseline_model)
    bids_folder = Path(args.bids_folder)

    if args.per_subject:
        run_per_subject(args.subjects, args.mode_model, args.cv_model, baseline,
                        bids_folder, args.sigma, args.smoothed,
                        static_png=args.static_png, cmap=args.cmap)
    else:
        run_group(args.subjects, args.mode_model, args.cv_model, baseline,
                  bids_folder, args.min_prevalence, args.sigma, args.smoothed,
                  static_png=args.static_png, cmap=args.cmap)
