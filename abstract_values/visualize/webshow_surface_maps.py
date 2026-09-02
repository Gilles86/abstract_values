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
import json
import subprocess
import time
from datetime import datetime
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

# Where per-subject bundles are written, one directory each, so a single
# server at the root can browse every processed subject.
DEFAULT_WEBGL_ROOT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "webgl"


def dataset_name(label, smoothed):
    """Viewer-facing name for one map.

    Pycortex lists these verbatim, so they have to say what the map IS
    without the reader holding a legend in their head. Leading with the
    quantity keeps the smoothed/unsmoothed pair adjacent when sorted.
    Kept to plain ASCII: the names become JSON keys and DOM ids.
    """
    return f"{label} ({'smoothed' if smoothed else 'unsmoothed'})"


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


def ensure_alpha_cmap(name):
    """Return the name of a 2D "<name>_alpha" colormap, generating it if needed.

    A ``blend_curvature`` dataset is pre-blended RGB, so the viewer has no
    vmin/vmax/cmap left to introspect and cannot draw a real colorbar. A
    ``Vertex2D`` keeps them live: dim1 carries the value, dim2 indexes a 2D
    colormap whose alpha channel ramps 0 -> 1, so thresholding still reads as
    transparency AND the viewer shows the colormap with working sliders.

    Pycortex ships a handful of ``*_alpha`` maps but not one per matplotlib
    colormap, so build any missing one: 256x256 RGBA, dim1 across (the
    colormap), dim2 down (alpha, opaque at the top to match the shipped maps).
    """
    cmapdir = Path(cortex.options.config.get("webgl", "colormaps"))
    target = f"{name}_alpha"
    out = cmapdir / f"{target}.png"
    if out.exists():
        return target
    import matplotlib.pyplot as _plt
    from matplotlib.image import imsave
    n = 256
    rgb = _plt.get_cmap(name)(np.linspace(0, 1, n))[:, :3]          # (n, 3)
    img = np.repeat(rgb[None, :, :], n, axis=0)                     # (n, n, 3)
    alpha = np.linspace(1.0, 0.0, n)[:, None]                       # opaque on top
    img = np.dstack([img, np.repeat(alpha, n, axis=1)])
    cmapdir.mkdir(parents=True, exist_ok=True)
    imsave(str(out), img.astype(np.float32))
    print(f"  generated 2D colormap {out}")
    return target


def live(values, alpha, cx_subject, vmin, vmax, cmap, vmin2=0.0, vmax2=1.0,
         nonce=0):
    """Vertex2D carrying its own colorbar, as an alternative to `blended`.

    `nonce` works around a pycortex packing limitation. A BrainData's name is
    a read-only hash of its array, and Package.reorder is not dedup-aware: if
    two datasets hand it bit-identical arrays it serialises that one slot
    twice, and the second pass finds bytes where it expects an ndarray and
    dies with "byte indices must be integers or slices, not tuple". The
    parameter maps legitimately share one alpha mask (the same
    cvR2-beats-null gate), so scale each copy by 1 - nonce*1e-6 to keep the
    arrays distinct. The colormap quantises alpha to 256 levels, so a
    perturbation this small cannot change a rendered pixel.
    """
    alpha = np.nan_to_num(alpha).astype(np.float32) * (1.0 - nonce * 1e-6)
    return cortex.Vertex2D(np.nan_to_num(values).astype(np.float32), alpha,
                           cx_subject, vmin=vmin, vmax=vmax,
                           vmin2=vmin2, vmax2=vmax2,
                           cmap=ensure_alpha_cmap(cmap))


def signal_alpha(delta, pct=95.0):
    """Opacity for a "beats the null" map: gate on sign, scale by magnitude.

    A pure ``delta > 0`` gate makes a vertex that beats its null by 1e-6 as
    loud as one that beats it by 0.15, which is what turns these maps into
    confetti. The sign is still the test — nothing negative is ever shown —
    but opacity then ramps linearly to a robust high percentile of the
    positive margins, so the eye is drawn to effect size instead of to the
    sheer number of vertices that scraped past zero.
    """
    delta = np.nan_to_num(delta, nan=-np.inf)
    pos = delta[delta > 0]
    if pos.size == 0:
        return np.zeros_like(delta, dtype=np.float32)
    scale = float(np.percentile(pos, pct)) or float(pos.max())
    return np.clip(delta / max(scale, 1e-9), 0.0, 1.0).astype(np.float32)


def blended(values, alpha, cx_subject, vmin, vmax, cmap):
    v = cortex.Vertex(np.nan_to_num(values).astype(np.float32), cx_subject,
                      vmin=vmin, vmax=vmax, cmap=cmap)
    return v.blend_curvature(np.clip(alpha, 0, 1))


def build_datasets(subject, bids_folder=BIDS_FOLDER, smoothed=False,
                   cx_subject=None, r2_thr=0.05, r2_sigma=0.01,
                   cv_sigma=0.01, alpha_source="cvr2-null",
                   colorbars="baked"):
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
        if colorbars == "live":
            vtx = live(values, alpha, cx_subject, vmin, vmax, cmap,
                       nonce=len(ds))
        else:
            vtx = blended(values, alpha, cx_subject, vmin, vmax, cmap)
        ds[dataset_name(label, smoothed)] = vtx
        cbars.append((label, cmap, vmin, vmax))
        print(f"  {key}: range [{np.nanmin(values):.3g}, {np.nanmax(values):.3g}]")

    r2 = L("aprf", "r2")
    alpha_r2 = r2_alpha(r2, r2_thr, r2_sigma) if r2 is not None else None

    cv = L("aprf.cv", "cvr2")
    null = L("aprf-null.cv", "cvr2")
    lin = L("aprf-linear.cv", "cvr2")

    # Which mask decides "there is value signal in this vertex".
    #
    # Full-fit R² is not held-out, so its absolute scale does not mean what a
    # fixed cut like 0.05 implies, and the cut is arbitrary besides. The
    # project's per-voxel test is cvR²(model) > cvR²(null): held-out, and fair
    # across models with different parameter counts.
    #
    # Empirically on sub-29 the two are nested rather than in conflict —
    # R² >= 0.05 keeps 1.7% of vertices (5.0% smoothed) and is almost a strict
    # subset of the 14.2% (25.5%) that beat the null. So the R² cut was not
    # letting overfit vertices through so much as hiding ~8x more real,
    # modest-effect cortex than it showed.
    if alpha_source == "cvr2-null" and cv is not None and null is not None:
        alpha_signal = signal_alpha(cv - null)
        print("  alpha: cvR²(aPRF) > cvR²(aprf-null.cv), opacity ∝ margin")
    elif alpha_source == "cvr2-null":
        alpha_signal = alpha_r2
        print("  WARNING: alpha falls back to full-fit R² — aprf.cv / "
              "aprf-null.cv fsnative surfaces missing. Parameter maps will "
              "show overfit vertices.")
    else:
        alpha_signal = alpha_r2
        print(f"  alpha: full-fit R² >= {r2_thr}")

    mode = L("aprf", "mode")
    if mode is not None and alpha_signal is not None:
        # Modes pinned at the edge of the fitted range are unidentified, not
        # a real preference — drop them rather than paint the cortex red.
        in_range = ((mode >= MODE_VMIN) & (mode <= MODE_VMAX)).astype(np.float32)
        add("mode", mode, alpha_signal * in_range, MODE_VMIN, MODE_VMAX,
            "turbo", "Preferred value CHF")

    fwhm = L("aprf", "fwhm")
    if fwhm is not None and alpha_signal is not None:
        add("fwhm", fwhm, alpha_signal, 0.0, MODE_VMAX - MODE_VMIN,
            "viridis", "Tuning width FWHM CHF")

    amp = L("aprf", "amplitude")
    if amp is not None and alpha_signal is not None:
        lim = float(np.nanpercentile(np.abs(amp), 99)) or 1.0
        add("amplitude", amp, alpha_signal, -lim, lim, "RdBu_r",
            "Response amplitude")

    if r2 is not None:
        vmax = float(np.nanpercentile(r2[r2 > 0], 99.9)) if (r2 > 0).any() else 0.3
        add("aprf_r2", r2, alpha_r2, r2_thr, vmax, "hot", "aPRF full-fit R2")

    gabor = L("aprf", "gabor-r2")
    if gabor is not None:
        vmax = (float(np.nanpercentile(gabor[gabor > 0], 99.9))
                if (gabor > 0).any() else 0.3)
        add("gabor_r2", gabor, r2_alpha(gabor, r2_thr, r2_sigma),
            r2_thr, vmax, "hot", "Orientation von Mises R2")

    # ── cross-validated maps ────────────────────────────────────────────────
    # These two are one-sided in practice: negative cvR2 just means "no signal
    # here", so the alpha gate hides it. Painting that with a diverging
    # colormap advertises a blue half that never appears — use a sequential
    # ramp anchored at 0 and say what is actually shown.
    if cv is not None:
        vmax = float(np.nanpercentile(cv[cv > 0], 99)) if (cv > 0).any() else 0.05
        add("aprf_cvr2", cv, signal_alpha(cv), 0.0, vmax, "hot",
            "aPRF cross-validated cvR2")

    if cv is not None and null is not None:
        delta = cv - null
        vmax = (float(np.nanpercentile(delta[delta > 0], 99))
                if (delta > 0).any() else 0.05)
        add("aprf_vs_null", delta, signal_alpha(delta), 0.0, vmax, "hot",
            "Signal cvR2 aPRF minus null")
    elif cv is not None:
        print("  skip aprf_vs_null: aprf-null.cv has no fsnative surface")

    # This one is genuinely two-sided — positive means a monotonic ramp in
    # value beats a tuned bump, negative the reverse — so it keeps a diverging
    # map, with opacity on the magnitude of the difference either way.
    if cv is not None and lin is not None:
        delta = lin - cv
        lim = float(np.nanpercentile(np.abs(delta), 99)) or 0.05
        # Gate on there being value signal at all before asking which shape
        # fits it better. |linear - aPRF| is non-zero essentially everywhere,
        # so without the gate the whole cortex lights up with a comparison
        # between two models that both explain nothing at that vertex.
        if null is not None:
            gate = (np.nan_to_num(cv - null, nan=-np.inf) > 0).astype(np.float32)
        else:
            gate = (np.nan_to_num(cv, nan=-np.inf) > 0).astype(np.float32)
        add("linear_vs_aprf", delta, gate * signal_alpha(np.abs(delta)),
            -lim, lim, "PuOr_r", "Ramp vs bump cvR2 linear minus aPRF")

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
        if hasattr(vtx, "red"):                      # VertexRGB (baked)
            arr = np.stack([vtx.red.data, vtx.green.data,
                            vtx.blue.data]).astype(np.uint8)
        else:                                        # Vertex2D (live)
            arr = np.stack([np.nan_to_num(vtx.dim1.data),
                            np.nan_to_num(vtx.dim2.data)]).astype(np.float32)
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


def _gradient_css(cmap, n=24):
    """CSS linear-gradient approximating a matplotlib colormap."""
    import matplotlib as _mpl
    import matplotlib.pyplot as _plt
    # Callers may pass a name or an already-built Colormap (the categorical
    # winner map builds a ListedColormap on the fly).
    cm = cmap if isinstance(cmap, _mpl.colors.Colormap) else _plt.get_cmap(cmap)
    cols = cm(np.linspace(0, 1, n))[:, :3]
    stops = ", ".join(
        "rgb(%d,%d,%d)" % tuple(int(round(255 * c)) for c in row) for row in cols)
    return f"linear-gradient(to right, {stops})"


def inject_legend(index_html, names, cbars):
    """Add a colorbar panel to a make_static page that follows the active map.

    blend_curvature bakes data and curvature into one RGB image, which is what
    makes these maps read well against the anatomy — but it leaves pycortex no
    vmin/vmax/cmap to build a colorbar from. So draw the legend ourselves from
    the ranges we already hold, and show only the map currently on screen:
    with both smoothing variants there are 16 of them, and a wall of scales is
    no more use than none.

    Tracking is done by wrapping ``mriview.Viewer.prototype.setData`` — the
    single funnel every dataset switch goes through, whether it came from the
    dropdown, a keypress or the URL. If that global is ever missing the panel
    falls back to listing everything rather than showing nothing.
    """
    index_html = Path(index_html)
    html = index_html.read_text()
    if 'id="aprf-legend"' in html:
        return
    entries = {
        name: {"cmap": _gradient_css(cmap), "label": label,
               "vmin": f"{vmin:.3g}", "vmax": f"{vmax:.3g}"}
        for name, (label, cmap, vmin, vmax) in zip(names, cbars)}

    panel = """
<style>
#aprf-legend { position: fixed; left: 20px; bottom: 20px; z-index: 10000;
  font: 16px/1.45 -apple-system, system-ui, sans-serif; color: #f2f2f2;
  background: rgba(18,18,18,.92); border: 1px solid #4a4a4a; border-radius: 10px;
  padding: 16px 20px 18px; width: 30vw; min-width: 380px; max-width: 620px;
  box-shadow: 0 6px 24px rgba(0,0,0,.45); }
#aprf-legend .cb-name { font-size: 17px; font-weight: 600; color: #fff;
  margin-bottom: 10px; line-height: 1.25; }
#aprf-legend .cb-bar { height: 30px; border-radius: 4px; border: 1px solid #666; }
#aprf-legend .cb-lim { display: flex; justify-content: space-between;
  align-items: baseline; font-size: 14px; color: #b6b6b6;
  font-variant-numeric: tabular-nums; margin-top: 7px; gap: 14px; }
#aprf-legend .cb-lim b { color: #f2f2f2; font-weight: 500; font-size: 13px;
  text-align: center; }
#aprf-legend.all { max-height: 72vh; overflow-y: auto; }
#aprf-legend.all .cb { margin-bottom: 18px; }
#aprf-legend.all .cb-name { font-size: 14px; margin-bottom: 6px; }
#aprf-legend.all .cb-bar { height: 20px; }
#aprf-toggle { float: right; cursor: pointer; color: #cfcfcf; font-size: 13px;
  border: 1px solid #666; border-radius: 5px; padding: 3px 10px;
  margin: -4px -6px 0 10px; user-select: none; }
#aprf-toggle:hover { color: #fff; border-color: #999; background: rgba(255,255,255,.08); }
</style>
<div id="aprf-legend"><span id="aprf-toggle">all</span><div id="aprf-body"></div></div>
<script>
(function () {
  var CB = __ENTRIES__;
  var showAll = false, current = null;
  function row(name, e) {
    return '<div class="cb"><div class="cb-name">' + name + '</div>' +
           '<div class="cb-bar" style="background:' + e.cmap + '"></div>' +
           '<div class="cb-lim"><span>' + e.vmin + '</span><b>' + e.label +
           '</b><span>' + e.vmax + '</span></div></div>';
  }
  function render() {
    var body = document.getElementById('aprf-body');
    var panel = document.getElementById('aprf-legend');
    if (!body) return;
    if (showAll) {
      panel.className = 'all';
      body.innerHTML = Object.keys(CB).map(function (k) { return row(k, CB[k]); }).join('');
    } else {
      panel.className = '';
      var k = (current && CB[current]) ? current : Object.keys(CB)[0];
      body.innerHTML = CB[k] ? row(k, CB[k])
        : '<div class="cb-name">No map selected</div>';
    }
  }
  function setCurrent(name) {
    if (name instanceof Array) name = name[0];
    current = name;
    if (!showAll) render();
  }
  // pycortex selects the first dataset during load, which happens before the
  // hook below is installed — so read the active view directly rather than
  // showing nothing until the user switches. `figure` is an implicit global
  // (assigned without var in the generated page).
  function probe() {
    try {
      var roots = [typeof figure !== 'undefined' ? figure : null];
      for (var i = 0; i < roots.length; i++) {
        var r = roots[i];
        if (!r) continue;
        if (r.active && r.active.name) return r.active.name;
        for (var k in r) {
          var c = r[k];
          if (c && c.active && c.active.name) return c.active.name;
        }
      }
    } catch (err) {}
    return null;
  }
  document.addEventListener('DOMContentLoaded', function () {
    var t = document.getElementById('aprf-toggle');
    if (t) t.onclick = function () { showAll = !showAll; t.textContent = showAll ? 'one' : 'all'; render(); };
    render();
  });
  // Wrap the single funnel every dataset switch goes through.
  var tries = 0;
  var iv = setInterval(function () {
    if (typeof mriview !== 'undefined' && mriview.Viewer && mriview.Viewer.prototype.setData) {
      clearInterval(iv);
      var orig = mriview.Viewer.prototype.setData;
      mriview.Viewer.prototype.setData = function (name) {
        var r = orig.apply(this, arguments);
        try { setCurrent(this.active ? this.active.name : name); } catch (err) { setCurrent(name); }
        return r;
      };
      var found = probe();
      if (found) setCurrent(found);
    } else if (++tries > 100) {   // ~10 s; viewer never appeared
      clearInterval(iv);
      showAll = true;
      var t = document.getElementById('aprf-toggle');
      if (t) t.textContent = 'one';
      render();
    }
  }, 100);
})();
</script>
"""
    panel = panel.replace("__ENTRIES__", json.dumps(entries))
    marker = "</body>"
    html = (html.replace(marker, panel + marker) if marker in html
            else html + panel)
    index_html.write_text(html)
    print(f"  injected a tracking legend for {len(entries)} maps "
          f"into {index_html.name}")


def save_colorbar_pdf(cbars, out_path):
    fig, axes = plt.subplots(len(cbars), 1, figsize=(5, 1.15 * len(cbars)))
    axes = np.atleast_1d(axes)
    for ax, (label, cmap, vmin, vmax) in zip(axes, cbars):
        import matplotlib as _mpl
        cm = cmap if isinstance(cmap, _mpl.colors.Colormap) else plt.get_cmap(cmap)
        cb = ColorbarBase(ax, cmap=cm,
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

    # make_static never removes files it did not write this run, so a rebuild
    # with different datasets leaves the old ones behind: unreferenced, but
    # they accumulate (a browser bundle reached 235 MB across three builds).
    for stale in (out_dir / "data").glob("*"):
        stale.unlink()

    print(f"Building static webgl bundle in {out_dir} ...")
    # Default curvature is near-binary dark/light grey, which fights the data
    # for attention. Flatten it into a soft background: lower contrast, a
    # little smoothing so the gyral/sulcal pattern still orients you.
    cortex.webgl.make_static(str(out_dir), ds, types=types,
                             title=f"sub-{subject} aPRF surface maps",
                             recache=False,
                             curvature_brightness=0.62,
                             curvature_contrast=0.28,
                             curvature_smoothness=2.0)
    save_colorbar_pdf(cbars, out_dir / "colorbars.pdf")
    inject_legend(out_dir / "index.html", list(ds.keys()), cbars)
    print(f"Wrote static bundle → {out_dir / 'index.html'}")
    return out_dir


def write_root_index(root):
    """Generate a landing page listing every subject bundle under `root`.

    http.server's own directory listing works, but it shows the raw folder
    names and gives no hint which bundles are stale. This lists subjects in
    the project's usual order (numeric first, pilots last) with the date each
    bundle was built.
    """
    root = Path(root)
    rows = []
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        idx = d / "index.html"
        if not idx.exists():
            continue
        label = d.name.removeprefix("sub-")
        built = datetime.fromtimestamp(idx.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        rows.append((label, d.name, built, size / 1e6))

    # cohort-level bundles first, then study subjects numerically, pilots last
    titles = {"group": "Group (all subjects)",
              "r2-browser": "R² browser (every subject, both models)"}
    order = {"group": -2, "r2-browser": -1}

    def sort_key(r):
        if r[1] in order:
            return (order[r[1]], 0)
        return (0, int(r[0])) if r[0].isdigit() else (1, 0)

    rows.sort(key=sort_key)

    items = "\n".join(
        f'      <li><a href="{name}/index.html">'
        f'{titles.get(name, f"sub-{label}")}</a>'
        f'<span>{built} &middot; {mb:.0f} MB</span></li>'
        for label, name, built, mb in rows)
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>aPRF surface maps</title>
<style>
  body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 3rem auto;
         max-width: 40rem; color: #222; }}
  h1 {{ font-size: 1.25rem; font-weight: 600; }}
  p.sub {{ color: #666; margin-top: -0.5rem; }}
  ul {{ list-style: none; padding: 0; }}
  li {{ display: flex; justify-content: space-between; align-items: baseline;
        padding: 0.5rem 0; border-bottom: 1px solid #eee; }}
  a {{ text-decoration: none; color: #1a5fb4; font-weight: 500; }}
  a:hover {{ text-decoration: underline; }}
  span {{ color: #888; font-size: 0.85em; font-variant-numeric: tabular-nums; }}
</style>
<h1>aPRF surface maps</h1>
<p class="sub">{len(rows)} subject bundle(s), individual (fsnative) space.</p>
<ul>
{items}
</ul>
"""
    (root / "index.html").write_text(html)
    print(f"Wrote root index ({len(rows)} subjects) -> {root / 'index.html'}")
    return root / "index.html"


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
    p.add_argument("subject", nargs="?", default=None,
                   help="Subject label without 'sub-', e.g. 29. Omit when "
                        "using --serve-all.")
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
    p.add_argument("--colorbars", default="baked", choices=["baked", "live"],
                   help="'baked' (default): blend_curvature, which composites "
                        "data onto curvature in Python and reads far better "
                        "against the anatomy. Its ranges are drawn as an "
                        "injected legend panel and written to colorbars.pdf. "
                        "'live': Vertex2D against a 2D *_alpha colormap — "
                        "gives pycortex's own colorbar and working sliders, "
                        "but the shader-side alpha blending looks washed out.")
    p.add_argument("--alpha-source", default="cvr2-null",
                   choices=["cvr2-null", "r2"],
                   help="What masks the parameter maps (mode/fwhm/amplitude). "
                        "'cvr2-null' (default): show a vertex only where "
                        "cvR²(aPRF) beats cvR²(aprf-null.cv) — the project's "
                        "per-voxel signal test. 'r2': legacy full-fit R² >= "
                        "--r2-thr, which is not held-out and empirically ~8x "
                        "stricter, hiding most modest-effect cortex.")
    p.add_argument("--static-png", default=None,
                   help="Write static flatmap PNGs to this directory instead "
                        "of launching the viewer")
    p.add_argument("--static-html", default=None,
                   help="Explicit bundle destination, overriding the default "
                        "<out-root>/sub-<subject>.")
    p.add_argument("--serve", type=int, nargs="?", const=8000, default=None,
                   help="After --static-html, serve that directory over HTTP "
                        "on this port (default 8000) and open a browser.")
    p.add_argument("--out-root", default=str(DEFAULT_WEBGL_ROOT),
                   help=f"Root holding one bundle per subject "
                        f"(default {DEFAULT_WEBGL_ROOT}). A build with no "
                        f"--static-html writes to <out-root>/sub-<subject>.")
    p.add_argument("--live", action="store_true",
                   help="Launch the in-process cortex.webgl viewer instead of "
                        "writing a bundle. Dies when this process exits; the "
                        "bundle does not.")
    p.add_argument("--serve-all", type=int, nargs="?", const=8000, default=None,
                   help="Serve every subject bundle under --out-root on this "
                        "port (default 8000), behind a generated index. "
                        "Takes no subject argument.")
    args = p.parse_args()

    if args.serve_all is not None:
        root = Path(args.out_root)
        if not root.exists():
            raise SystemExit(f"No bundle root at {root} — build a subject first.")
        write_root_index(root)
        serve_directory(root, args.serve_all)
        return

    if args.subject is None:
        raise SystemExit("A subject is required unless --serve-all is given.")

    variants = ([False, True] if args.both_smoothing else [args.smoothed])
    # Static PNGs always use the baked form: pre-blending onto curvature reads
    # better on paper, and the PDF carries the real ranges anyway.
    colorbars = "baked" if args.static_png else args.colorbars
    ds, cbars = {}, []
    for sm in variants:
        print(f"sub-{args.subject}  smoothed={sm}")
        d, c = build_datasets(args.subject, args.bids_folder, smoothed=sm,
                              cx_subject=args.cx_subject,
                              r2_thr=args.r2_thr, r2_sigma=args.r2_sigma,
                              alpha_source=args.alpha_source,
                              colorbars=colorbars)
        ds.update(d)
        cbars.extend(c)

    if not ds:
        raise SystemExit("No surface data found — run sample_aprf_to_surface.py first.")

    ds, cbars = drop_duplicate_datasets(ds, cbars)

    if args.static_png:
        save_static(ds, cbars, args.static_png)
        return

    if not args.live:
        # Default: write a bundle into <out-root>/sub-XX. It outlives this
        # process, unlike cortex.webgl.show, whose server thread is a daemon
        # and dies the moment the interpreter exits.
        dest = args.static_html or (Path(args.out_root) / f"sub-{args.subject}")
        out = write_static_html(ds, cbars, dest, args.subject, args.cx_subject)
        write_root_index(out.parent)
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
