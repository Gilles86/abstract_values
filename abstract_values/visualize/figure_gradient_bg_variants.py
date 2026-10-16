"""Design sheet: ways to show *where there is a code at all* under the ROI weights.

Same panel (value-precision weight on value-code strength, v2) drawn six
ways; the background is the group-mean value cvR² over the null.

    python -m abstract_values.visualize.figure_gradient_bg_variants   # pycortex2 env
    -> notes/figures/fig_gradient_bg_variants.pdf
"""
from __future__ import annotations

import cortex
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from abstract_values.visualize import figure_gradient_versions as fv
from abstract_values.visualize.figure_gradient import paint

LIM = 0.5


def curv_rgb(brightness=0.82, contrast=0.12):
    zero = np.zeros(sum(cortex.db.get_surf("fsaverage", "flat", h)[0].shape[0]
                        for h in ("lh", "rh")), np.float32)
    c = cortex.Vertex(zero, "fsaverage").blend_curvature(zero, brightness=brightness,
                                                         contrast=contrast)
    return np.stack([c.red.data, c.green.data, c.blue.data], 1) / 255.0


def to_vtx(rgb):
    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return cortex.VertexRGB(u8[:, 0], u8[:, 1], u8[:, 2], "fsaverage")


def ramp(bg, lo_q=75, hi_q=95):
    lo, hi = np.nanpercentile(bg, [lo_q, hi_q])
    return np.clip((np.nan_to_num(bg) - lo) / (hi - lo), 0, 1)


def mix(rgb, col, a):
    return rgb * (1 - a[:, None]) + np.asarray(to_rgb(col))[None, :] * a[:, None]


def roi_layer(rgb, m, col, a_in):
    roi = fv.stage_cmap(col)(np.clip(np.nan_to_num(m), 0, LIM) / LIM)
    return rgb * (1 - a_in[:, None]) + roi[:, :3] * a_in[:, None]


def variant(name, m, bg, col):
    inside = np.isfinite(m)
    r = ramp(bg)
    if name == "A":                       # current: dark grey outside, ROI opacity
        rgb = mix(curv_rgb(), (0.42, 0.42, 0.46), 0.75 * r * ~inside)
        return roi_layer(rgb, m, col, np.where(inside, 0.3 + 0.7 * r, 0))
    if name == "B":                       # ROI opacity only
        return roi_layer(curv_rgb(), m, col, np.where(inside, 0.25 + 0.75 * r, 0))
    if name == "C":                       # outside tinted in the stage hue
        rgb = mix(curv_rgb(), fv.LIGHT[col] if False else col, 0.45 * r * ~inside)
        return roi_layer(rgb, m, col, np.where(inside, 0.3 + 0.7 * r, 0))
    if name == "D":                       # spotlight: non-coding cortex washed out
        base = curv_rgb(0.7, 0.35)
        rgb = mix(base, (1, 1, 1), 0.8 * (1 - r))
        return roi_layer(rgb, m, col, np.where(inside, 0.3 + 0.7 * r, 0))
    if name == "E":                       # contours only, solid ROIs
        return roi_layer(curv_rgb(), m, col, inside.astype(float))
    if name == "F":                       # light blue-grey outside, solid ROIs
        rgb = mix(curv_rgb(), (0.55, 0.62, 0.72), 0.7 * r * ~inside)
        return roi_layer(rgb, m, col, np.where(inside, 0.45 + 0.55 * r, 0))
    raise ValueError(name)


TITLES = {"A": "A  Current: grey outside, ROI opacity = signal",
          "B": "B  ROI opacity = signal, nothing outside",
          "C": "C  Outside tinted in the stage colour",
          "D": "D  Spotlight: cortex without signal fades to white",
          "E": "E  Contours of signal over solid ROIs",
          "F": "F  Pale blue-grey outside, ROI opacity"}


def main():
    tab = fv.subject_table(fv.PARAMS)
    val, _, verts, maps = fv.strength_measures()
    est = fv.profile(val, tab, ["perc_prec", "val_prec"])
    fig = plt.figure(figsize=(7.25, 6.6))
    for i, name in enumerate(TITLES):
        col = fv.REP_C
        m = paint({r: est[r][1] if r in est else np.nan for r in verts}, verts)
        ax = fig.add_axes([0.5 * (i % 2), 0.67 - 0.33 * (i // 2), 0.5, 0.29])
        fv.brain_panel(ax, {r: est[r][1] if r in est else np.nan for r in verts}, verts,
                       col)                                 # sets labels + crop
        xl, yl = ax.get_xlim(), ax.get_ylim()
        data_im = ax.images[0]                           # replace only the data layer
        z, origin, ext = data_im.get_zorder(), data_im.origin, data_im.get_extent()
        data_im.remove()
        rgb = variant(name, m, maps["val"], col)
        img, _ = cortex.quickflat.make_flatmap_image(to_vtx(rgb), height=2048)
        ax.imshow(img, extent=ext, origin=origin, zorder=z, interpolation="bilinear")
        if name == "E":
            bimg, _ = cortex.quickflat.make_flatmap_image(
                cortex.Vertex(np.nan_to_num(maps["val"]).astype(np.float32), "fsaverage"),
                height=1024)
            lv = np.nanpercentile(maps["val"], [85, 95])
            xs = np.linspace(ext[0], ext[1], bimg.shape[1])
            ys = np.linspace(ext[2], ext[3], bimg.shape[0])
            if origin == "upper":
                ys = ys[::-1]
            ax.contour(xs, ys, np.nan_to_num(bimg), levels=lv, colors=["0.45", "0.2"],
                       linewidths=[0.5, 0.8])
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        fig.text(0.5 * (i % 2) + 0.01, 0.975 - 0.33 * (i // 2), TITLES[name], fontsize=7)
    out = fv.FIG / "fig_gradient_bg_variants.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
