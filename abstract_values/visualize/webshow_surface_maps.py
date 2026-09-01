"""All the relevant aPRF maps for one subject, on that subject's own surface.

Individual (fsnative) space — no fsaverage resampling, so nothing is blurred
across the cortical-folding mismatch that makes single-subject group-space
maps look speckled. Opens one pycortex webgl viewer holding every map worth
looking at when judging a subject's encoding fit:

    mode            preferred value (CHF) — the aPRF's actual tuning
    fwhm            tuning width (CHF)
    amplitude       signed response gain
    aprf_r2         aPRF full-fit R²
    gabor_r2        von Mises orientation-model R² (the sanity channel: V1
                    should light up here even where value tuning does not)
    aprf_cvr2       cross-validated R² — can be legitimately negative
    aprf_vs_null    cvR²(aPRF) − cvR²(aprf-null.cv), the honest "is there
                    signal here" contrast (see the project note: > 0 against
                    a real null, not > 0 against zero)
    linear_vs_aprf  cvR²(aprf-linear) − cvR²(aPRF): positive where a monotonic
                    ramp in value beats a tuned bump

Maps whose surface files are missing are skipped with a warning rather than
failing the whole viewer.

Run from the ``pycortex2`` env (see the pycortex skill: keep anything that
touches ``cortex.*`` out of the heavy analysis env).

Usage
-----
    # interactive viewer (stays alive until you Ctrl-C)
    python -m abstract_values.visualize.webshow_surface_maps 29

    # smoothed variant
    python -m abstract_values.visualize.webshow_surface_maps 29 --smoothed

    # static PNGs instead — the only way to actually verify what rendered
    python -m abstract_values.visualize.webshow_surface_maps 29 \\
        --static-png notes/figures/sub-29_surface
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cortex
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from scipy.stats import norm

from abstract_values.utils.data import BIDS_FOLDER

MODE_VMIN, MODE_VMAX = 2.0, 42.0


def _load_hemi(deriv, model, subject, hemi, desc, smoothed):
    tag = "_smoothed" if smoothed else ""
    fn = (deriv / "encoding_models" / model / f"sub-{subject}" / "func" /
          f"sub-{subject}_task-abstractvalue_hemi-{hemi}"
          f"_space-fsnative_desc-{desc}{tag}_pe.func.gii")
    if not fn.exists():
        return None
    return nib.load(str(fn)).darrays[0].data.astype(np.float32)


def load_bilateral(deriv, model, subject, desc, smoothed):
    """L+R concatenated in pycortex order, or None if either hemi is missing."""
    lh = _load_hemi(deriv, model, subject, "L", desc, smoothed)
    rh = _load_hemi(deriv, model, subject, "R", desc, smoothed)
    if lh is None or rh is None:
        return None
    return np.concatenate([lh, rh])


def r2_alpha(r2, thr, sigma):
    """Smooth alpha ramp via a Gaussian CDF centred on ``thr``.

    NaNs are forced to 0 alpha: ``blend_curvature`` clips but does not guard
    NaN, and a NaN alpha silently renders the whole dataset as an identical
    all-black image — which then collides with any other such dataset on
    pycortex's content hash and dies far away with a byte-indexing TypeError
    (see the pycortex skill).
    """
    a = norm.cdf(np.nan_to_num(r2, nan=-np.inf), loc=thr, scale=sigma)
    return np.nan_to_num(a, nan=0.0).astype(np.float32)


def blended(values, alpha, cx_subject, vmin, vmax, cmap):
    v = cortex.Vertex(np.nan_to_num(values).astype(np.float32), cx_subject,
                      vmin=vmin, vmax=vmax, cmap=cmap)
    return v.blend_curvature(np.clip(alpha, 0, 1))


def build_datasets(subject, bids_folder=BIDS_FOLDER, smoothed=False,
                   cx_subject=None, r2_thr=0.05, r2_sigma=0.01,
                   cv_sigma=0.01):
    """Returns (datasets dict, colorbar specs list)."""
    deriv = Path(bids_folder) / "derivatives"
    cx_subject = cx_subject or f"abstractvalue.sub-{subject}"
    tag = "_smoothed" if smoothed else ""

    def L(model, desc):
        return load_bilateral(deriv, model, subject, desc, smoothed)

    ds, cbars = {}, []

    def add(key, values, alpha, vmin, vmax, cmap, label):
        if values is None:
            print(f"  skip {key}: surface files missing")
            return
        ds[f"{subject}{tag}.{key}"] = blended(values, alpha, cx_subject,
                                              vmin, vmax, cmap)
        cbars.append((label, cmap, vmin, vmax))
        print(f"  {key}: range [{np.nanmin(values):.3g}, {np.nanmax(values):.3g}]")

    r2 = L("aprf", "r2")
    alpha_r2 = r2_alpha(r2, r2_thr, r2_sigma) if r2 is not None else None

    mode = L("aprf", "mode")
    if mode is not None and alpha_r2 is not None:
        # Modes pinned at the edge of the fitted range are unidentified, not
        # a real preference — drop them rather than paint the cortex red.
        in_range = ((mode >= MODE_VMIN) & (mode <= MODE_VMAX)).astype(np.float32)
        add("mode", mode, alpha_r2 * in_range, MODE_VMIN, MODE_VMAX,
            "nipy_spectral", "Preferred value (CHF)")

    fwhm = L("aprf", "fwhm")
    if fwhm is not None and alpha_r2 is not None:
        add("fwhm", fwhm, alpha_r2, 0.0, MODE_VMAX - MODE_VMIN,
            "viridis", "Tuning FWHM (CHF)")

    amp = L("aprf", "amplitude")
    if amp is not None and alpha_r2 is not None:
        lim = float(np.nanpercentile(np.abs(amp), 99)) or 1.0
        add("amplitude", amp, alpha_r2, -lim, lim, "RdBu_r", "Amplitude (a.u.)")

    if r2 is not None:
        vmax = float(np.nanpercentile(r2[r2 > 0], 99.9)) if (r2 > 0).any() else 0.3
        add("aprf_r2", r2, alpha_r2, r2_thr, vmax, "hot", "aPRF R²")

    gabor = L("aprf", "gabor-r2")
    if gabor is not None:
        vmax = (float(np.nanpercentile(gabor[gabor > 0], 99.9))
                if (gabor > 0).any() else 0.3)
        add("gabor_r2", gabor, r2_alpha(gabor, r2_thr, r2_sigma),
            r2_thr, vmax, "hot", "Gabor (von Mises) R²")

    # ── cross-validated maps ────────────────────────────────────────────────
    cv = L("aprf.cv", "cvr2")
    null = L("aprf-null.cv", "cvr2")
    lin = L("aprf-linear.cv", "cvr2")

    if cv is not None:
        lim = float(np.nanpercentile(np.abs(cv), 99)) or 0.05
        add("aprf_cvr2", cv, r2_alpha(cv, 0.0, cv_sigma), -lim, lim,
            "RdBu_r", "aPRF cvR²")

    if cv is not None and null is not None:
        delta = cv - null
        lim = float(np.nanpercentile(np.abs(delta), 99)) or 0.05
        add("aprf_vs_null", delta, r2_alpha(delta, 0.0, cv_sigma), -lim, lim,
            "RdBu_r", "cvR²: aPRF − null")
    elif cv is not None:
        print("  skip aprf_vs_null: aprf-null.cv has no fsnative surface")

    if cv is not None and lin is not None:
        delta = lin - cv
        lim = float(np.nanpercentile(np.abs(delta), 99)) or 0.05
        add("linear_vs_aprf", delta, r2_alpha(np.abs(delta), 0.0, cv_sigma),
            -lim, lim, "PuOr_r", "cvR²: linear − aPRF")

    return ds, cbars


def drop_duplicate_datasets(ds, cbars):
    """Remove datasets whose blended RGB is bit-identical to an earlier one.

    Pycortex keys datasets inside ``Package`` by a content hash of the RGB
    array, but ``Package.reorder()`` is not dedup-aware: it reprocesses the
    shared slot twice and the second pass hits already-serialised bytes,
    dying with ``TypeError: byte indices must be integers or slices, not
    tuple`` — an error that says nothing about the real cause. Duplicates are
    always a bug upstream (an all-NaN alpha, or two descs pointing at the
    same source volume), so warn loudly rather than dropping silently.
    """
    seen, keep_ds, keep_cbars = {}, {}, []
    for (name, vtx), cbar in zip(ds.items(), cbars):
        arr = np.stack([vtx.red.data, vtx.green.data,
                        vtx.blue.data]).astype(np.uint8)
        key = hash(arr.tobytes())
        if key in seen:
            print(f"  WARNING: {name} is bit-identical to {seen[key]} — "
                  f"dropping it (pycortex cannot pack duplicates). "
                  f"Check the source files for these two descs.")
            continue
        seen[key] = name
        keep_ds[name] = vtx
        keep_cbars.append(cbar)
    return keep_ds, keep_cbars


def save_colorbar_pdf(cbars, out_path):
    fig, axes = plt.subplots(len(cbars), 1, figsize=(5, 1.15 * len(cbars)))
    axes = np.atleast_1d(axes)
    for ax, (label, cmap, vmin, vmax) in zip(axes, cbars):
        cb = ColorbarBase(ax, cmap=plt.get_cmap(cmap),
                          norm=Normalize(vmin=vmin, vmax=vmax),
                          orientation="horizontal")
        cb.set_label(label)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    plt.close(fig)
    print(f"Wrote {out_path}")


def save_static(ds, cbars, out_dir):
    """One flatmap PNG per dataset, plus one colorbar PDF.

    ``with_colorbar`` is deliberately off: these are pre-blended RGB images,
    so pycortex has no vmin/vmax/cmap left to introspect and would draw a
    meaningless 0-255 swatch.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, vtx in ds.items():
        fn = out_dir / f"{name.replace('.', '_')}_flatmap.png"
        # with_rois/with_sulci render the subject's overlay SVG, which
        # pycortex rasterises by shelling out to Inkscape — not installed
        # here, and a freshly imported subject has no drawn ROIs anyway.
        fig = cortex.quickflat.make_figure(vtx, with_curvature=False,
                                           with_colorbar=False,
                                           with_rois=False, with_sulci=False,
                                           with_labels=False)
        fig.savefig(str(fn), dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {fn}")
    save_colorbar_pdf(cbars, out_dir / "colorbars.pdf")


def has_flatmap(cx_subject):
    """True if the pycortex subject has flat surfaces.

    A subject imported straight from FreeSurfer has none — flat maps need
    either manual Freeview cuts or an autoflatten run — and asking the
    viewer for a 'flat' type it does not have makes it fail at load.
    """
    try:
        cortex.db.get_surf(cx_subject, "flat", merge=True, nudge=True)
        return True
    except Exception:
        return False


def write_static_html(ds, cbars, out_dir, subject, cx_subject=None):
    """Write a self-contained webgl bundle servable by any static file server.

    Unlike ``cortex.webgl.show``, nothing here depends on a live Python
    process: the output is plain HTML/JS/binary that survives the script
    exiting, can be rsynced elsewhere, and can be reopened later without
    rebuilding the datasets.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cx_subject = cx_subject or f"abstractvalue.sub-{subject}"

    # `types` lists MORPH TARGETS only. Flat must never go in here: pycortex
    # picks the flat surface up on its own (BrainCTM loads it and stores it as
    # UV coordinates, which is what drives the viewer's flatten slider), while
    # addSurf("flat") would renormalise it into the fiducial bounding box —
    # and the flat surface's z axis is constant, so that divides by zero and
    # OpenCTM rejects the mesh with a bare CTM_INVALID_MESH.
    types = ("inflated",)
    if has_flatmap(cx_subject):
        print(f"  {cx_subject} has flat surfaces — flatten view enabled")
    else:
        print(f"  note: {cx_subject} has no flat surfaces — inflated only "
              f"(run autoflatten + cortex.freesurfer.import_flat to add them)")

    print(f"Building static webgl bundle in {out_dir} ...")
    cortex.webgl.make_static(str(out_dir), ds, types=types,
                             title=f"sub-{subject} aPRF surface maps",
                             recache=False)
    save_colorbar_pdf(cbars, out_dir / "colorbars.pdf")
    print(f"Wrote static bundle → {out_dir / 'index.html'}")
    return out_dir


def serve_directory(directory, port):
    """Serve `directory` over HTTP until interrupted, and open a browser."""
    import functools
    import http.server
    import socketserver

    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(directory))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://localhost:{port}/index.html"
        print(f"\n=== SERVING ===\nOpen this URL:  {url}\n"
              f"Serving {directory}\n===============\n", flush=True)
        subprocess.run(["open", url])
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subject", help="Subject label without 'sub-', e.g. 29")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--smoothed", action="store_true",
                   help="Load the _smoothed BOLD variant")
    p.add_argument("--both-smoothing", action="store_true",
                   help="Load unsmoothed AND smoothed into one viewer")
    p.add_argument("--cx-subject", default=None,
                   help="Pycortex subject (default: abstractvalue.sub-<subject>)")
    p.add_argument("--r2-thr", type=float, default=0.05,
                   help="R² alpha threshold, fraction scale (default 0.05)")
    p.add_argument("--r2-sigma", type=float, default=0.01,
                   help="Gaussian-CDF width of the R² alpha ramp")
    p.add_argument("--static-png", default=None,
                   help="Write static flatmap PNGs to this directory instead "
                        "of launching the viewer")
    p.add_argument("--static-html", default=None,
                   help="Write a self-contained webgl bundle to this directory "
                        "(cortex.webgl.make_static) instead of launching the "
                        "live viewer. Serve it with any static file server; "
                        "--serve does that for you.")
    p.add_argument("--serve", type=int, nargs="?", const=8000, default=None,
                   help="After --static-html, serve that directory over HTTP "
                        "on this port (default 8000) and open a browser.")
    args = p.parse_args()

    variants = ([False, True] if args.both_smoothing else [args.smoothed])
    ds, cbars = {}, []
    for sm in variants:
        print(f"sub-{args.subject}  smoothed={sm}")
        d, c = build_datasets(args.subject, args.bids_folder, smoothed=sm,
                              cx_subject=args.cx_subject,
                              r2_thr=args.r2_thr, r2_sigma=args.r2_sigma)
        ds.update(d)
        cbars.extend(c)

    if not ds:
        raise SystemExit("No surface data found — run sample_aprf_to_surface.py first.")

    ds, cbars = drop_duplicate_datasets(ds, cbars)

    if args.static_png:
        save_static(ds, cbars, args.static_png)
        return

    if args.static_html:
        out = write_static_html(ds, cbars, args.static_html, args.subject,
                                args.cx_subject)
        if args.serve is not None:
            serve_directory(out, args.serve)
        return

    # cortex.webgl.show()'s server thread is daemon=True and dies the moment
    # this process returns; open_browser=True also builds the URL from the
    # machine hostname, which usually will not resolve. Keep the process alive
    # and open a plain localhost URL ourselves (see the pycortex skill).
    print(f"\nLaunching pycortex viewer with {len(ds)} dataset(s)...")
    server = cortex.webgl.show(ds, open_browser=False, autoclose=False)
    url = f"http://localhost:{server.port}/mixer.html"
    print(f"\n=== WEBSHOW READY ===\nOpen this URL:  {url}\n=====================\n",
          flush=True)
    subprocess.run(["open", url])
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
