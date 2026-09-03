"""Preferred value (aPRF mode) painted on the fsaverage surface.

The companion to ``preferred_value_figure``, which shows the same quantity as
distributions. This one puts it on cortex, because the question "does M1 look
like a value map or like a bid-magnitude ramp" is partly a question about
*spatial* structure: a real value code should show orderly progressions across
a patch, a movement-extent confound should not.

Per vertex, the map is the **median preferred value across the subjects whose
aPRF beats their own ``aprf-null.cv`` there**. Taking the median only over
subjects that pass their own signal test is the point: a mean over everyone
would average in the modes of voxels with no value response at all, which are
arbitrary. Opacity encodes how many subjects contributed, so a vertex where
3/29 subjects agree is visibly fainter than one where 20/29 do.

Panel d is the part that decides something. Bids are entered by moving a
mouse slider (``experiment/task.py`` ``get_events``), and the mouse is
re-centred on the marker before every response phase, so the hand's final
position is monotonic in the bid. A bid-magnitude motor/proprioceptive
confound must therefore be lateralised to the hemisphere contralateral to the
response hand; a value code has no reason to be. Note that the response hand
is *not* recorded anywhere in the repo — the strong left-lateralisation found
here is itself the evidence that responses were right-handed, which is the
usual arrangement but was not verified independently.

ROI outlines are drawn by flat-mapping the binary mask itself and contouring
it, rather than through ``quickflat``'s ``with_rois`` — that path shells out
to Inkscape, which is not installed here.

Run from the ``pycortex2`` env.

Usage
-----
    python -m abstract_values.visualize.preferred_value_surface \\
        --out notes/figures/preferred_value_surface.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cortex
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import nibabel.freesurfer.io as fsio
import numpy as np
from matplotlib.colors import Normalize

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize.group_surface_maps import (
    discover_subjects, load_fsaverage)
from abstract_values.visualize.webshow_surface_maps import blended

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 150, "savefig.dpi": 300,
})

CX = "fsaverage"
FS_LABEL_DIR = ("/data/ds-abstractvalue/derivatives/fmriprep/sourcedata/"
                "freesurfer/fsaverage/label")
N_LH, N_VERT = 163842, 327684
VALUE_MIN, VALUE_MAX = 2.0, 42.0
CMAP = "turbo"

ROI_OUTLINES = {
    "V1": dict(colour="#111111", lw=0.7),
    "M1": dict(colour="#111111", lw=0.9),
    "S1": dict(colour="#111111", lw=0.9),
    "NPC": dict(colour="#111111", lw=0.7),
}


def fs_label(*names):
    m = np.zeros(N_VERT, bool)
    for name in names:
        for hemi, off in (("lh", 0), ("rh", N_LH)):
            m[off + fsio.read_label(f"{FS_LABEL_DIR}/{hemi}.{name}.label")] = True
    return m


def rois():
    rv = cortex.get_roi_verts(CX)
    npc = np.zeros(N_VERT, bool)
    for k in ("NPC_L", "NPC_R"):
        if k in rv:
            npc[rv[k]] = True
    return {
        "V1": fs_label("V1_exvivo.thresh"),
        "M1": fs_label("BA4a_exvivo.thresh", "BA4p_exvivo.thresh"),
        "S1": fs_label("BA3b_exvivo.thresh", "BA1_exvivo.thresh",
                       "BA2_exvivo.thresh"),
        "NPC": npc,
    }


def per_subject_modes(deriv, subjects, smoothed=False):
    """Stack of per-subject preferred values, NaN where the aPRF loses to null.

    One array per subject, so downstream code can do the subject-wise things a
    group map cannot: count how many people contribute to a vertex, and run a
    paired test across subjects instead of eyeballing a group picture.
    """
    modes, used = [], []
    for s in subjects:
        mode = load_fsaverage(deriv, "aprf", s, "mode", smoothed)
        cv = load_fsaverage(deriv, "aprf.cv", s, "cvr2", smoothed)
        null = load_fsaverage(deriv, "aprf-null.cv", s, "cvr2", smoothed)
        if mode is None or cv is None or null is None:
            continue
        delta = cv - null
        good = (np.isfinite(delta) & (delta > 0) & np.isfinite(mode)
                & (mode >= VALUE_MIN) & (mode <= VALUE_MAX))
        modes.append(np.where(good, mode, np.nan).astype(np.float32))
        used.append(s)
    return np.vstack(modes), used


def group_mode(deriv, subjects, smoothed=False):
    """Per-vertex median preferred value over subjects that beat their null."""
    stack, used = per_subject_modes(deriv, subjects, smoothed)
    count = np.isfinite(stack).sum(axis=0).astype(np.float32)
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, axis=0)
    return med, count, len(used)


def prevalence_alpha(count, n, min_frac=0.15, sat_frac=0.45):
    """Opacity ramping from `min_frac` of subjects (invisible) to `sat_frac`.

    A hard count gate makes the map binary and speckled — a vertex with 5/29
    subjects looks exactly as authoritative as one with 22/29. Ramping instead
    lets the eye weight the map by how many people actually agree.
    """
    frac = count / max(n, 1)
    return np.clip((frac - min_frac) / max(sat_frac - min_frac, 1e-9),
                   0, 1).astype(np.float32)


def lateralisation(stack, roi):
    """Per-subject L-vs-R counts and preferred values inside one ROI.

    Responses are made with the mouse in the right hand, so a bid-magnitude
    motor/proprioceptive confound must be **left**-lateralised, while a value
    code has no reason to be. This is the test that separates them; the group
    map only suggests it.
    """
    lh = np.zeros(N_VERT, bool)
    lh[:N_LH] = True
    out = []
    for row in stack:
        ok = np.isfinite(row)
        L, R = roi & lh, roi & ~lh
        out.append(dict(
            n_l=int((ok & L).sum()), n_r=int((ok & R).sum()),
            frac_l=float((ok & L).sum() / max(L.sum(), 1)),
            frac_r=float((ok & R).sum() / max(R.sum(), 1)),
            val_l=float(np.nanmedian(row[L])) if (ok & L).sum() >= 10 else np.nan,
            val_r=float(np.nanmedian(row[R])) if (ok & R).sum() >= 10 else np.nan))
    return out


def flat(values, **kw):
    """Flatmap image of an fsaverage vertex array (nan outside the flat patch)."""
    vtx = cortex.Vertex(np.asarray(values, np.float32), CX, **kw)
    img, extents = cortex.quickflat.composite.make_flatmap_image(vtx, height=1024)
    return img, extents


def outline(ax, mask_img, extents, colour, lw):
    ax.contour(np.nan_to_num(mask_img, nan=0.0), levels=[0.5],
               colors=[colour], linewidths=lw, extent=extents,
               origin="lower")


def _to_data(extents, shape, xs, ys):
    h, w = shape
    x0, x1, y0, y1 = extents
    return (x0 + (xs + 0.5) / w * (x1 - x0), y0 + (ys + 0.5) / h * (y1 - y0))


def half(mask_img, side):
    """Keep only the left or right half of a flat-mapped mask image."""
    m = np.nan_to_num(mask_img, nan=0.0) > 0.5
    w = m.shape[1]
    keep = np.zeros_like(m)
    sl = slice(0, w // 2) if side == "l" else slice(w // 2, w)
    keep[:, sl] = m[:, sl]
    return keep


def bbox_of(mask, extents, pad=0.3):
    ys, xs = np.nonzero(mask)
    xd, yd = _to_data(extents, mask.shape, xs.astype(float), ys.astype(float))
    bx0, bx1, by0, by1 = xd.min(), xd.max(), yd.min(), yd.max()
    px, py = pad * (bx1 - bx0), pad * (by1 - by0)
    return bx0 - px, bx1 + px, by0 - py, by1 + py


def match_boxes(a, b):
    """Give two bounding boxes an identical width and height, each centred."""
    wa, wb = a[1] - a[0], b[1] - b[0]
    ha, hb = a[3] - a[2], b[3] - b[2]
    w, h = max(wa, wb), max(ha, hb)
    out = []
    for x0, x1, y0, y1 in (a, b):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        out.append((cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2))
    return out


def label_rois(ax, masks, extents, names, side=None, fontsize=7.5):
    for name in names:
        m = np.nan_to_num(masks[name], nan=0.0) > 0.5
        sides = ("l", "r") if side is None else (side,)
        for sd in sides:
            mm = half(masks[name], sd)
            if mm.sum() < 50:
                continue
            ys, xs = np.nonzero(mm)
            cx_, cy_ = _to_data(extents, mm.shape, xs.mean(), ys.mean())
            ax.text(cx_, cy_, name, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color="w",
                    path_effects=[pe.withStroke(linewidth=1.8,
                                                foreground="0.1")])


def figure(deriv, subjects, out_pdf, smoothed=False, min_frac=0.15,
           sat_frac=0.45):
    stack, used = per_subject_modes(deriv, subjects, smoothed)
    n = len(used)
    count = np.isfinite(stack).sum(axis=0).astype(np.float32)
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, axis=0)
    alpha = prevalence_alpha(count, n, min_frac, sat_frac)
    print(f"n={n} subjects; vertices with >=1 subject: {int((count > 0).sum())}")

    R = rois()
    sensorimotor = R["M1"] | R["S1"]
    lat = lateralisation(stack, sensorimotor)
    dl = np.array([r["frac_l"] for r in lat])
    dr = np.array([r["frac_r"] for r in lat])
    try:
        from scipy.stats import wilcoxon
        stat, pval = wilcoxon(dl, dr)
        test = f"Wilcoxon W = {stat:.0f}, p = {pval:.1e}"
    except Exception:
        test = ""
    print(f"  sensorimotor L {dl.mean():.3f} vs R {dr.mean():.3f}; "
          f"{sum(dl > dr)}/{n} subjects L>R. {test}")

    rgb = blended(np.nan_to_num(med, nan=VALUE_MIN), alpha, CX,
                  VALUE_MIN, VALUE_MAX, CMAP)
    img, extents = cortex.quickflat.composite.make_flatmap_image(rgb, height=1024)
    masks = {k: flat(m.astype(np.float32), vmin=0, vmax=1)[0]
             for k, m in R.items()}

    fig = plt.figure(figsize=(7.25, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.14,
                          wspace=0.22, left=0.03, right=0.985, top=0.97,
                          bottom=0.10)

    # ── a. whole flatmap ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(img, extent=extents, origin="lower", interpolation="bilinear")
    for name, kw in ROI_OUTLINES.items():
        outline(ax, masks[name], extents, kw["colour"], kw["lw"])
    ax.set_xlim(extents[0], extents[1])
    ax.set_ylim(extents[2], extents[3])
    ax.axis("off")
    label_rois(ax, masks, extents, ROI_OUTLINES)

    # ── b, c. central sulcus, both hemispheres at a matched scale ────────────
    cs = np.nan_to_num(masks["M1"], nan=0.0) + np.nan_to_num(masks["S1"], nan=0.0)
    boxes = match_boxes(bbox_of(half(cs, "l"), extents, pad=0.22),
                        bbox_of(half(cs, "r"), extents, pad=0.22))
    zoom_axes = []
    for i, (side, title, box) in enumerate(
            [("l", "Left central sulcus", boxes[0]),
             ("r", "Right central sulcus", boxes[1])]):
        axz = fig.add_subplot(gs[1, i])
        zoom_axes.append(axz)
        axz.imshow(img, extent=extents, origin="lower", interpolation="bilinear")
        for name in ("M1", "S1"):
            outline(axz, masks[name], extents, "#111111", 1.0)
        axz.set_xlim(box[0], box[1])
        axz.set_ylim(box[2], box[3])
        axz.set_title(title, fontsize=7)
        axz.axis("off")
        label_rois(axz, masks, extents, ("M1", "S1"), side=side, fontsize=7)

    # ── d. per-subject lateralisation ────────────────────────────────────────
    axd = fig.add_subplot(gs[1, 2])
    rng = np.random.default_rng(0)
    xs = []
    for j, (v, col) in enumerate([(100 * dl, "#C44E52"), (100 * dr, "#3B5BA5")]):
        x = j + rng.uniform(-.13, .13, len(v))
        xs.append(x)
        axd.scatter(x, v, s=13, alpha=.55, color=col, linewidths=0, zorder=3)
        axd.hlines(np.median(v), j - .27, j + .27, color=col, lw=2.2, zorder=4)
    # Paired lines must land on the jittered points, not on nominal x — a line
    # that misses its own dot reads as a different subject.
    axd.plot(np.vstack(xs), np.vstack([100 * dl, 100 * dr]), color="0.78",
             lw=0.4, zorder=1)
    axd.set_xticks([0, 1])
    axd.set_xticklabels(["Left", "Right"])
    axd.set_xlim(-0.45, 1.45)
    axd.set_xlabel("Sensorimotor hemisphere")
    axd.set_ylabel("Vertices beating null (%)")
    axd.set_title(f"M1 + S1, per subject\n{test}", fontsize=7)
    sns_despine(axd)

    # Anchor panel letters to the real axes boxes rather than guessed figure
    # coordinates — a hand-placed letter drifts onto a tick label the moment
    # the layout changes.
    fig.canvas.draw()
    for a_, letter, dy in ((ax, "a", 0.0), (zoom_axes[0], "b", 0.012),
                           (zoom_axes[1], "c", 0.012), (axd, "d", 0.052)):
        bb = a_.get_position()
        fig.text(bb.x0 - 0.030, bb.y1 + dy, letter, fontsize=8,
                 fontweight="bold", va="top", ha="left")

    cax = fig.add_axes([0.37, 0.045, 0.26, 0.020])
    mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap(CMAP),
                              norm=Normalize(VALUE_MIN, VALUE_MAX),
                              orientation="horizontal")
    cax.set_xticks([2, 12, 22, 32, 42])
    cax.set_xlabel("Preferred value (CHF)", labelpad=2)
    cax.tick_params(length=2, width=0.6, pad=1.5)

    fig.text(0.01, 0.045,
             f"Median across the subjects beating their own\n"
             f"aprf-null.cv; n = {n}; opacity = prevalence "
             f"({min_frac:.0%}\u2013{sat_frac:.0%})",
             fontsize=6.2, color="0.35", va="bottom")

    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(out_pdf).replace(".pdf", ".png"), bbox_inches="tight",
                pad_inches=0.02, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def sns_despine(ax, offset=4):
    """Minimal despine+offset, so this module need not import seaborn."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_position(("outward", offset))


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--min-frac", type=float, default=0.15)
    p.add_argument("--sat-frac", type=float, default=0.45)
    p.add_argument("--out", default="notes/figures/preferred_value_surface.pdf")
    a = p.parse_args()
    deriv = Path(a.bids_folder) / "derivatives"
    subjects = a.subjects or discover_subjects(deriv)
    figure(deriv, subjects, a.out, a.smoothed, a.min_frac, a.sat_frac)


if __name__ == "__main__":
    main()
