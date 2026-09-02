"""Group-level aPRF surface maps on fsaverage, plus an all-subjects contact sheet.

Two outputs, both in group (fsaverage) space:

1. ``--static-html`` — a webgl bundle, the group counterpart of
   ``webshow_surface_maps``, holding:

       Subjects beating null (count)   per vertex, how many subjects have
                                       cvR2(aPRF) > cvR2(aprf-null.cv)
       Mean aPRF full-fit R2           straight cross-subject mean
       Mean orientation von Mises R2   same, for the gabor model

   The count map is the honest headline. Averaging cvR2 across subjects is
   a bad summary: cvR2 is unbounded below, so a handful of very negative
   noise vertices drag the mean around and the resulting map mostly reflects
   where the *worst* subject was worst. Binarising per subject first — does
   this subject's aPRF beat its own null here — throws that tail away and
   asks a question with a defensible answer.

2. ``--contact-sheet`` — one PDF page, one flatmap panel per subject, so a
   whole cohort's map can be eyeballed at once for outliers or coverage
   holes.

Everything is rendered on pycortex's built-in ``fsaverage`` subject, which
ships with a flat map. Per-subject native flatmaps would need an
``autoflatten`` run (~35 min) and a pycortex import each, and would not be
comparable panel-to-panel anyway.

Run from the ``pycortex2`` env.

Usage
-----
    python -m abstract_values.visualize.group_surface_maps --static-html
    python -m abstract_values.visualize.group_surface_maps \\
        --contact-sheet notes/figures/group_aprf_vs_null.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cortex
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize.webshow_surface_maps import (
    DEFAULT_WEBGL_ROOT, blended, inject_legend, live, save_colorbar_pdf,
    serve_directory, write_root_index)

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 8, "axes.titlesize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

CX_FSAVERAGE = "fsaverage"


def load_fsaverage(deriv, model, subject, desc, smoothed=False):
    """L+R fsaverage surface for one subject/model/desc, or None if missing."""
    tag = "_smoothed" if smoothed else ""
    out = []
    for hemi in ("L", "R"):
        fn = (deriv / "encoding_models" / model / f"sub-{subject}" / "func" /
              f"sub-{subject}_task-abstractvalue_hemi-{hemi}"
              f"_space-fsaverage_desc-{desc}{tag}_pe.func.gii")
        if not fn.exists():
            return None
        out.append(nib.load(str(fn)).darrays[0].data.astype(np.float32))
    return np.concatenate(out)


def discover_subjects(deriv, model="aprf.cv", desc="cvr2"):
    """Subjects with the fsaverage surface for `model`, study first, pilots last."""
    d = deriv / "encoding_models" / model
    subs = []
    for p in sorted(d.glob("sub-*")):
        label = p.name.removeprefix("sub-")
        if load_fsaverage(deriv, model, label, desc) is not None:
            subs.append(label)
    return sorted(subs, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))


def null_beaten_count(deriv, subjects, smoothed=False):
    """Per-vertex count of subjects whose aPRF cvR2 beats their own null.

    Returns ``(count, used)``. Subjects missing either surface are skipped and
    reported, so the denominator is never silently wrong.
    """
    stack, used, missing = [], [], []
    for s in subjects:
        cv = load_fsaverage(deriv, "aprf.cv", s, "cvr2", smoothed)
        null = load_fsaverage(deriv, "aprf-null.cv", s, "cvr2", smoothed)
        if cv is None or null is None:
            missing.append(s)
            continue
        delta = cv - null
        stack.append(np.isfinite(delta) & (delta > 0))
        used.append(s)
    if missing:
        print(f"  missing cv/null surfaces, excluded: {' '.join(missing)}")
    if not stack:
        return None, []
    return np.vstack(stack).sum(axis=0).astype(np.float32), used


def mean_map(deriv, subjects, model, desc, smoothed=False):
    """Cross-subject mean of a per-subject fsaverage map."""
    stack = [m for s in subjects
             if (m := load_fsaverage(deriv, model, s, desc, smoothed)) is not None]
    if not stack:
        return None, 0
    return np.nanmean(np.vstack(stack), axis=0), len(stack)


# ── group webgl bundle ───────────────────────────────────────────────────────

def build_group_datasets(deriv, subjects, smoothed_variants=(False, True),
                         min_count=None, colorbars="baked"):
    ds, cbars = {}, []

    def make(values, alpha, vmin, vmax, cmap):
        if colorbars == "live":
            return live(values, alpha, CX_FSAVERAGE, vmin, vmax, cmap,
                        nonce=len(ds))
        return blended(values, alpha, CX_FSAVERAGE, vmin, vmax, cmap)

    for sm in smoothed_variants:
        sm_tag = "smoothed" if sm else "unsmoothed"
        print(f"group  smoothed={sm}")

        count, used = null_beaten_count(deriv, subjects, sm)
        if count is not None:
            n = len(used)
            thr = min_count if min_count is not None else n / 2.0
            # Alpha is a hard gate here, not a ramp: the quantity is a count,
            # so "at least half the cohort" is a statement about the data, not
            # a display parameter to soften.
            alpha = (count >= thr).astype(np.float32)
            name = f"Subjects beating null n={n} ({sm_tag})"
            ds[name] = make(count, alpha, thr, float(n), "hot")
            cbars.append((f"Subjects with cvR2 > null (of {n})", "hot", thr, n))
            print(f"  count: max {int(count.max())}/{n}, "
                  f"{100 * (count >= thr).mean():.1f}% of vertices >= {thr:.0f}")

        for model, desc, label in [("aprf", "r2", "Mean aPRF full-fit R2"),
                                   ("aprf", "gabor-r2",
                                    "Mean orientation von Mises R2")]:
            m, n = mean_map(deriv, subjects, model, desc, sm)
            if m is None:
                print(f"  skip {label}: no surfaces")
                continue
            vmax = float(np.nanpercentile(m[m > 0], 99)) if (m > 0).any() else 0.1
            thr = vmax / 4.0
            alpha = np.clip((m - thr) / max(vmax - thr, 1e-6), 0, 1).astype(np.float32)
            name = f"{label} n={n} ({sm_tag})"
            ds[name] = make(m, alpha, thr, vmax, "hot")
            cbars.append((f"{label} (n={n})", "hot", thr, vmax))
            print(f"  {label}: n={n}, range [{np.nanmin(m):.3g}, {np.nanmax(m):.3g}]")
    return ds, cbars



# ── per-subject R² browser ───────────────────────────────────────────────────

BROWSE_MODELS = [("aprf", "aPRF"), ("vonmises", "vonMises")]


NULL_MODEL = "aprf-null.cv"


def _cv_dir(model):
    """The cross-validated dir for a model dir (aprf -> aprf.cv)."""
    return model if model.endswith(".cv") else f"{model}.cv"


def beats_null(deriv, model, subject, smoothed=False):
    """Boolean per-vertex mask: this model's cvR2 beats the null model's.

    ``aprf-null.cv`` predicts the training mean, so it is a valid baseline for
    any encoding model fitted to the same betas — vonMises included. Returns
    None if either surface is missing, so the caller can say so rather than
    silently showing an unmasked map.
    """
    cv = load_fsaverage(deriv, _cv_dir(model), subject, "cvr2", smoothed)
    null = load_fsaverage(deriv, NULL_MODEL, subject, "cvr2", smoothed)
    if cv is None or null is None:
        return None
    return np.isfinite(cv - null) & ((cv - null) > 0)


def build_browse_datasets(deriv, subjects, desc="r2", smoothed=False,
                          models=BROWSE_MODELS, colorbars="baked",
                          floor_pct=25.0, mask="cvr2-null",
                          mean_prevalence=0.25):
    """One dataset per (model, subject) on fsaverage, plus the group mean.

    Every subject in a model shares one colour scale — floor to the 99th
    percentile of that model's pooled values. Per-subject autoscaling would
    make a weak subject look as strong as a good one, which defeats the whole
    point of paging through them.

    With ``mask="cvr2-null"`` a vertex is drawn only where that subject's
    cross-validated R2 beats their own null model. That is the project's
    per-voxel signal test, and it is what stops the whole brain washing red:
    99.5% of cortex carries full-fit R2 > 0, because a full fit is not
    held-out and will always explain some noise.

    `floor_pct` then sets where opacity starts within what survives, as a
    percentile of the masked values, so strong vertices still read louder than
    marginal ones.

    The group mean is built on the same scale and named to sort first, so
    paging through a model starts from the cohort picture and then walks the
    subjects that make it up.
    """
    ds, cbars = {}, []
    tag = " smoothed" if smoothed else ""
    for model, pretty in models:
        maps = {}
        for sub in subjects:
            m = load_fsaverage(deriv, model, sub, desc, smoothed)
            if m is not None:
                maps[sub] = m
        if not maps:
            print(f"  skip {pretty}: no fsaverage {desc} surfaces")
            continue

        gates, no_gate = {}, []
        if mask == "cvr2-null":
            for sub in maps:
                g = beats_null(deriv, model, sub, smoothed)
                if g is None:
                    no_gate.append(sub)
                    g = np.ones_like(maps[sub], dtype=bool)
                gates[sub] = g
            if no_gate:
                print(f"    no cv/null surfaces, shown unmasked: "
                      f"{' '.join(no_gate)}")
        else:
            gates = {sub: np.ones_like(m, dtype=bool) for sub, m in maps.items()}

        # Scale from what actually survives the gate, not from all of cortex.
        kept = np.concatenate([maps[s][gates[s]] for s in maps])
        kept = kept[np.isfinite(kept)]
        vmax = float(np.percentile(kept, 99)) if kept.size else 0.1
        lo = float(np.percentile(kept, floor_pct)) if kept.size else 0.0
        span = max(vmax - lo, 1e-9)
        gated_frac = np.mean([g.mean() for g in gates.values()])
        print(f"  {pretty} {desc}: n={len(maps)}, scale {lo:.4g}–{vmax:.4g}; "
              f"{100 * gated_frac:.1f}% of vertices beat the null on average")

        def emit(name, values, label, gate=None, scale=None):
            v0, v1 = scale if scale else (lo, vmax)
            alpha = np.clip((np.nan_to_num(values) - v0) / max(v1 - v0, 1e-9),
                            0, 1).astype(np.float32)
            if gate is not None:
                alpha = alpha * gate.astype(np.float32)
            if colorbars == "live":
                ds[name] = live(values, alpha, CX_FSAVERAGE, v0, v1, "hot",
                                nonce=len(ds))
            else:
                ds[name] = blended(values, alpha, CX_FSAVERAGE, v0, v1, "hot")
            cbars.append((label, "hot", v0, v1))

        # Group mean: gate on the majority of the cohort beating the null,
        # matching the count map in the group bundle.
        mean = np.nanmean(np.vstack(list(maps.values())), axis=0)
        count = np.sum(np.vstack([gates[s] for s in maps]), axis=0)
        mean_gate = count >= max(mean_prevalence * len(maps), 1)
        # The mean needs its own scale. Averaging 29 noisy maps compresses the
        # range hard, so re-using the per-subject scale leaves almost every
        # mean value below the opacity floor and the map renders blank.
        kept_mean = mean[mean_gate]
        kept_mean = kept_mean[np.isfinite(kept_mean)]
        if kept_mean.size:
            m_scale = (float(np.percentile(kept_mean, floor_pct)),
                       float(np.percentile(kept_mean, 99)))
        else:
            m_scale = (lo, vmax)
        print(f"    mean: gate >= {mean_prevalence:.0%} of subjects "
              f"({100 * mean_gate.mean():.2f}% of vertices), "
              f"scale {m_scale[0]:.4g}–{m_scale[1]:.4g}")
        emit(f"{pretty} {desc.upper()}{tag} MEAN n={len(maps)}", mean,
             f"{pretty} {desc.upper()} group mean", mean_gate, m_scale)
        for sub, m in maps.items():
            emit(f"{pretty} {desc.upper()}{tag} sub-{sub}", m,
                 f"{pretty} {desc.upper()} sub-{sub}", gates[sub])
    return ds, cbars


# ── all-subjects contact sheet ───────────────────────────────────────────────

def contact_sheet(deriv, subjects, out_pdf, smoothed=False, quantity="vs-null",
                  ncol=6):
    """One page, one flatmap panel per subject, rendered on fsaverage.

    `quantity` is 'vs-null' (cvR2 aPRF minus its own null, the signal map),
    'cvr2', or 'r2'.
    """
    panels = []
    for s in subjects:
        if quantity == "vs-null":
            cv = load_fsaverage(deriv, "aprf.cv", s, "cvr2", smoothed)
            null = load_fsaverage(deriv, "aprf-null.cv", s, "cvr2", smoothed)
            vals = None if (cv is None or null is None) else cv - null
        elif quantity == "cvr2":
            vals = load_fsaverage(deriv, "aprf.cv", s, "cvr2", smoothed)
        else:
            vals = load_fsaverage(deriv, "aprf", s, "r2", smoothed)
        if vals is None:
            print(f"  skip sub-{s}: no surface")
            continue
        panels.append((s, vals))

    if not panels:
        raise SystemExit("No subjects had the requested surfaces.")

    # One shared colour scale across panels, or the sheet compares nothing.
    allv = np.concatenate([v for _, v in panels])
    if quantity == "r2":
        vmin, vmax, cmap = 0.0, float(np.nanpercentile(allv, 99.5)), "hot"
    else:
        vmax = float(np.nanpercentile(np.abs(allv), 99.5))
        vmin, cmap = 0.0, "hot"
    print(f"  shared scale [{vmin:.3g}, {vmax:.3g}] over {len(panels)} subjects")

    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 1.7 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (s, vals) in zip(axes, panels):
        alpha = np.clip((vals - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        vtx = blended(vals, alpha, CX_FSAVERAGE, vmin, vmax, cmap)
        im = cortex.quickflat.composite.make_flatmap_image(
            vtx, height=440)[0]
        ax.imshow(im, origin="lower", interpolation="nearest")
        ax.set_title(f"sub-{s}", pad=2)
        ax.set_axis_off()
    for ax in axes[len(panels):]:
        ax.set_axis_off()

    qlabel = {"vs-null": "cvR² − null", "cvr2": "cvR²", "r2": "R²"}[quantity]
    fig.suptitle(f"aPRF {qlabel} on fsaverage · "
                 f"{'smoothed' if smoothed else 'unsmoothed'} · "
                 f"n = {len(panels)} · shared scale {vmin:.3g}–{vmax:.3g}",
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}  ({len(panels)} panels)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Subject labels (default: everything on disk)")
    p.add_argument("--out-root", default=str(DEFAULT_WEBGL_ROOT))
    p.add_argument("--static-html", nargs="?", const="", default=None,
                   help="Build the group webgl bundle (default destination "
                        "<out-root>/group)")
    p.add_argument("--serve", type=int, nargs="?", const=8000, default=None)
    p.add_argument("--min-count", type=int, default=None,
                   help="Display threshold for the count map (default: half "
                        "the subjects with data)")
    p.add_argument("--browse", nargs="?", const="", default=None,
                   help="Build a per-subject R² browser on fsaverage: one "
                        "dataset per (model, subject) so you can page through "
                        "the whole cohort in the viewer. Default destination "
                        "<out-root>/r2-browser.")
    p.add_argument("--browse-floor", type=float, default=25.0,
                   help="Percentile (of the values surviving the mask) at "
                        "which opacity starts. Default 25.")
    p.add_argument("--browse-mean-prevalence", type=float, default=0.25,
                   help="Fraction of subjects that must beat the null at a "
                        "vertex for it to show in the MEAN map (default 0.25). "
                        "A majority gate leaves almost nothing: fsaverage "
                        "registration scatter means the exact vertex carrying "
                        "signal moves between subjects.")
    p.add_argument("--browse-mask", default="cvr2-null",
                   choices=["cvr2-null", "none"],
                   help="'cvr2-null' (default): draw a vertex only where that "
                        "subject's cvR2 beats their own aprf-null.cv. 'none': "
                        "show the raw R2 everywhere.")
    p.add_argument("--browse-desc", default="r2", choices=["r2", "cvr2"],
                   help="Which map the browser shows (default r2; cvr2 reads "
                        "the .cv model dirs).")
    p.add_argument("--contact-sheet", default=None,
                   help="Write a one-page all-subjects flatmap PDF here")
    p.add_argument("--quantity", default="vs-null",
                   choices=["vs-null", "cvr2", "r2"],
                   help="What the contact sheet shows (default: vs-null)")
    p.add_argument("--smoothed", action="store_true",
                   help="Contact sheet: use the smoothed variant")
    p.add_argument("--ncol", type=int, default=6)
    p.add_argument("--colorbars", default="baked", choices=["baked", "live"],
                   help="'baked' (default): blend_curvature plus an injected "
                        "legend panel. 'live': Vertex2D with pycortex's own "
                        "colorbar, but washed-out shader blending.")
    args = p.parse_args()

    deriv = Path(args.bids_folder) / "derivatives"
    subjects = args.subjects or discover_subjects(deriv)
    print(f"{len(subjects)} subjects: {' '.join(subjects)}\n")

    if args.contact_sheet:
        contact_sheet(deriv, subjects, args.contact_sheet,
                      smoothed=args.smoothed, quantity=args.quantity,
                      ncol=args.ncol)

    if args.browse is not None:
        dest = Path(args.browse) if args.browse else \
            Path(args.out_root) / "r2-browser"
        models = ([(f"{m}.cv", p) for m, p in BROWSE_MODELS]
                  if args.browse_desc == "cvr2" else BROWSE_MODELS)
        ds, cbars = build_browse_datasets(deriv, subjects, desc=args.browse_desc,
                                          smoothed=args.smoothed, models=models,
                                          colorbars=args.colorbars,
                                          floor_pct=args.browse_floor,
                                          mask=args.browse_mask,
                                          mean_prevalence=args.browse_mean_prevalence)
        if not ds:
            raise SystemExit("No surfaces found for the browser.")
        dest.mkdir(parents=True, exist_ok=True)
        for stale in (dest / "data").glob("*"):   # see write_static_html
            stale.unlink()
        print(f"\nBuilding browser ({len(ds)} maps) in {dest} ...")
        cortex.webgl.make_static(str(dest), ds, types=("inflated",),
                                 title=f"Per-subject {args.browse_desc.upper()} "
                                       f"browser (n={len(subjects)})",
                                 recache=False,
                                 curvature_brightness=0.62,
                                 curvature_contrast=0.28,
                                 curvature_smoothness=2.0)
        # Every subject in a model shares one scale, so the PDF needs one
        # bar per model, not per subject.
        seen, uniq = set(), []
        for label, cmap, vmin, vmax in cbars:
            key = (cmap, round(vmin, 6), round(vmax, 6))
            if key in seen:
                continue
            seen.add(key)
            uniq.append((label.rsplit(" sub-", 1)[0], cmap, vmin, vmax))
        save_colorbar_pdf(uniq, dest / "colorbars.pdf")
        inject_legend(dest / "index.html", list(ds.keys()), cbars)
        write_root_index(dest.parent)
        print(f"Wrote browser → {dest / 'index.html'}")
        if args.serve is not None:
            serve_directory(dest.parent, args.serve)

    if args.static_html is not None:
        dest = Path(args.static_html) if args.static_html else \
            Path(args.out_root) / "group"
        ds, cbars = build_group_datasets(deriv, subjects,
                                         min_count=args.min_count,
                                         colorbars=args.colorbars)
        if not ds:
            raise SystemExit("No group data built.")
        dest.mkdir(parents=True, exist_ok=True)
        for stale in (dest / "data").glob("*"):   # see write_static_html
            stale.unlink()
        print(f"\nBuilding group bundle in {dest} ...")
        cortex.webgl.make_static(str(dest), ds, types=("inflated",),
                                 title=f"Group aPRF maps (n={len(subjects)})",
                                 recache=False,
                                 curvature_brightness=0.62,
                                 curvature_contrast=0.28,
                                 curvature_smoothness=2.0)
        save_colorbar_pdf(cbars, dest / "colorbars.pdf")
        inject_legend(dest / "index.html", list(ds.keys()), cbars)
        write_root_index(dest.parent)
        print(f"Wrote group bundle → {dest / 'index.html'}")
        if args.serve is not None:
            serve_directory(dest.parent, args.serve)

    if (args.contact_sheet is None and args.static_html is None
            and args.browse is None):
        raise SystemExit("Nothing to do — pass --static-html, --browse "
                         "and/or --contact-sheet.")


if __name__ == "__main__":
    main()
