"""Anatomical ROI outlines (IPS, LO, M1) for the pycortex bundles.

The bundles show maps but no landmarks, so there is nothing to say *where* a
blob is. Pycortex's own ROI machinery expects an ``overlays.svg`` traced by
hand in Inkscape, which no subject here has. These ROIs come instead from the
FreeSurfer annotations fmriprep already produces, so they need no manual step
and exist for every subject:

    IPS  S_intrapariet_and_P_trans   (Destrieux, aparc.a2009s)
    LO   lateraloccipital            (Desikan, aparc)
    M1   BA4a + BA4p                 (BA_exvivo.thresh)

They are anatomical, not functional: IPS here is the sulcus, not the project's
NPC map, and LO is the gyral parcel rather than LO1/LO2. Use them to orient,
not to define analysis ROIs -- ``derivatives/masks`` is for that.

Drawn as outlines rather than filled patches: a filled ROI hides the map
underneath, which defeats the purpose of drawing it on top of one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# (label, annot file, entries to union, colour)
ROI_SPECS = [
    ("IPS", "aparc.a2009s",     ["S_intrapariet_and_P_trans"], "#3B5BA5"),
    ("LO",  "aparc",            ["lateraloccipital"],          "#2A9D8F"),
    ("M1",  "BA_exvivo.thresh", ["BA4a_exvivo", "BA4p_exvivo"], "#C4442B"),
]

FS_REL = "derivatives/fmriprep/sourcedata/freesurfer"


def _fs_subject_dir(bids_folder, subject):
    """fmriprep names the FreeSurfer subject sub-XX_ses-1, not sub-XX."""
    root = Path(bids_folder) / FS_REL
    for name in (f"sub-{subject}_ses-1", f"sub-{subject}"):
        if (root / name / "label").is_dir():
            return root / name
    raise SystemExit(
        f"No FreeSurfer label dir for sub-{subject} under {root}. "
        f"build_surface_bundle.py syncs it; run that first.")


def annot_masks(subject, bids_folder, specs=ROI_SPECS):
    """{label: boolean mask over L+R fsnative vertices} for each ROI spec."""
    import nibabel.freesurfer.io as fsio
    fs = _fs_subject_dir(bids_folder, subject)
    out = {}
    for label, annot, entries, _ in specs:
        parts = []
        for hemi in ("lh", "rh"):
            lab, _, names = fsio.read_annot(str(fs / "label" / f"{hemi}.{annot}.annot"))
            names = [n.decode() for n in names]
            idx = [names.index(e) for e in entries if e in names]
            if not idx:
                raise SystemExit(f"{entries} not in {hemi}.{annot}.annot "
                                 f"(has {len(names)} labels)")
            parts.append(np.isin(lab, idx))
        out[label] = np.concatenate(parts)
    return out


def _adjacency(cx_subject):
    """Neighbour lists over the L+R vertex concatenation, from the wm surface."""
    import cortex
    surfs = cortex.db.get_surf(cx_subject, "wm")
    neighbours, offset = {}, 0
    for pts, polys in surfs:
        for a, b, c in polys:
            for u, v in ((a, b), (b, c), (c, a)):
                neighbours.setdefault(u + offset, set()).add(v + offset)
                neighbours.setdefault(v + offset, set()).add(u + offset)
        offset += len(pts)
    return neighbours, offset


def outline(mask, neighbours, width=1):
    """Vertices on the inside edge of ``mask``, dilated ``width`` rings.

    One vertex is too thin to see once the surface is rendered at bundle
    resolution, hence the dilation.
    """
    edge = np.zeros_like(mask)
    for v in np.flatnonzero(mask):
        if any(not mask[n] for n in neighbours.get(v, ())):
            edge[v] = True
    for _ in range(max(0, width - 1)):
        grow = edge.copy()
        for v in np.flatnonzero(edge):
            for n in neighbours.get(v, ()):
                if mask[n]:
                    grow[n] = True
        edge = grow
    return edge


def roi_outline_dataset(subject, cx_subject, bids_folder, width=2,
                        specs=ROI_SPECS):
    """A categorical Vertex: 1..N on each ROI's outline, transparent elsewhere.

    Returns (vertex, colours, labels) so the caller can build a legend.
    """
    import cortex
    from matplotlib.colors import ListedColormap
    masks = annot_masks(subject, bids_folder, specs)
    neighbours, n_vert = _adjacency(cx_subject)

    values = np.zeros(n_vert, dtype=np.float32)
    alpha = np.zeros(n_vert, dtype=np.float32)
    for i, (label, _, _, _) in enumerate(specs, start=1):
        m = masks[label]
        if len(m) != n_vert:
            raise SystemExit(f"{label}: annot has {len(m)} vertices, surface "
                             f"has {n_vert} — mismatched FreeSurfer subject?")
        edge = outline(m, neighbours, width=width)
        values[edge] = i
        alpha[edge] = 1.0
        print(f"  {label}: {int(m.sum())} vertices, {int(edge.sum())} on the outline")

    # Register under a name: the legend builder resolves colormaps by name
    # through matplotlib, and hands a bare ListedColormap straight to
    # plt.get_cmap, which does not accept one on every matplotlib version.
    import matplotlib
    cmap_name = "roi_" + "_".join(s[0].lower() for s in specs)
    cmap = ListedColormap([c for _, _, _, c in specs], name=cmap_name)
    if cmap_name not in matplotlib.colormaps:
        matplotlib.colormaps.register(cmap, name=cmap_name)

    from abstract_values.visualize.webshow_surface_maps import blended
    vtx = blended(values, alpha, cx_subject, 0.5, len(specs) + 0.5, cmap)
    return vtx, cmap_name, [s[0] for s in specs]


# ── native pycortex overlays (vector paths in overlays.svg) ─────────────────
#
# The outline dataset above is a data layer: it can only be looked at one map
# at a time, like any other. A real pycortex ROI lives as a path in the
# subject's overlays.svg, which is what `with_rois=True` draws on top of
# whatever is displayed, in the mixer and in every flatmap. Those paths are
# normally traced by hand in Inkscape; these are traced by contouring the
# annotation mask in flatmap space, which is the same thing without the mouse.
#
# Needs flat surfaces: overlays.svg lives in flatmap coordinates, so a subject
# whose autoflatten has not finished cannot get one.

SVG_NS = "http://www.w3.org/2000/svg"
INK_NS = "http://www.inkscape.org/namespaces/inkscape"


def _svg_shape(svgfile):
    """(width, height) straight from the SVG header, as pycortex reads it."""
    import xml.etree.ElementTree as ET
    root = ET.parse(svgfile).getroot()
    return float(root.get("width")), float(root.get("height"))


def _flat_to_svg(cx_subject, svgshape):
    """Flat vertex coordinates in the SVG's pixel space, as pycortex maps them."""
    import cortex
    pts, _ = cortex.db.get_surf(cx_subject, "flat", merge=True, nudge=True)
    c = pts[:, :2].astype(float).copy()
    c -= c.min(0)
    c /= c.max(0)
    return c * np.asarray(svgshape)


def _contour_paths(mask, svg_xy, svgshape, grid=900, flip_y=True, smooth=2.5):
    """Closed SVG paths around ``mask``, contoured in flatmap space.

    ``smooth`` is a Gaussian blur (in grid cells) applied to the rasterised
    mask before contouring. Without it the outline follows the staircase of
    the nearest-neighbour rasterisation and looks visibly pixelated; a couple
    of cells of blur rounds it off without moving the boundary anywhere it
    matters, since the 0.5 level of a blurred binary mask stays put.
    """
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    w, h = svgshape
    nx = grid
    ny = max(8, int(round(grid * h / w)))
    xs = np.linspace(0, w, nx)
    ys = np.linspace(0, h, ny)
    gx, gy = np.meshgrid(xs, ys)

    # Nearest surface vertex for every grid cell; a cell is inside when its
    # nearest vertex is. Nearest-neighbour rather than interpolation so the
    # boundary stays where the parcel boundary is.
    tree = cKDTree(svg_xy)
    _, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]))
    z = mask[idx].reshape(gy.shape).astype(float)
    if smooth:
        from scipy.ndimage import gaussian_filter
        z = gaussian_filter(z, smooth)

    fig = plt.figure()
    cs = plt.contour(gx, gy, z, levels=[0.5])
    plt.close(fig)

    paths = []
    for seg in cs.allsegs[0]:
        if len(seg) < 8:                      # specks, not parcels
            continue
        if flip_y:
            seg = np.column_stack([seg[:, 0], h - seg[:, 1]])
        d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in seg) + " Z"
        paths.append(d)
    return paths


def write_roi_overlay(subject, cx_subject, bids_folder, specs=ROI_SPECS,
                      grid=900, flip_y=True, smooth=2.5, dry_run=False):
    """Write IPS/LO/M1 as real pycortex ROIs into the subject's overlays.svg."""
    import xml.etree.ElementTree as ET
    import cortex

    # Deliberately NOT cortex.db.get_overlay(): loading an overlay can rewrite
    # overlays.svg from pycortex's own in-memory tree (and with default args it
    # prompts "overwrite overlays.svg?" on stdin, which hangs a script). Both
    # lose the paths we are adding. Everything needed is in the file itself.
    svgfile = Path(cortex.database.default_filestore) / cx_subject / "overlays.svg"
    svgshape = _svg_shape(svgfile)
    svg_xy = _flat_to_svg(cx_subject, svgshape)
    masks = annot_masks(subject, bids_folder, specs)

    ET.register_namespace("", SVG_NS)
    ET.register_namespace("inkscape", INK_NS)
    tree = ET.parse(svgfile)
    root = tree.getroot()

    def _sub_layer(parent, name):
        for g in parent.findall(f"{{{SVG_NS}}}g"):
            if g.get(f"{{{INK_NS}}}label") == name:
                return g
        return None

    # pycortex reads rois > shapes > one <g inkscape:label=NAME> per ROI, whose
    # <path> children are the outline. A labelled <path> dropped straight into
    # the rois layer is valid XML, loads without error, and yields zero ROIs.
    rois = _sub_layer(root, "rois")
    if rois is None:
        raise SystemExit("no 'rois' layer in overlays.svg")
    shapes = _sub_layer(rois, "shapes")
    if shapes is None:
        shapes = ET.SubElement(rois, f"{{{SVG_NS}}}g")
        shapes.set(f"{{{INK_NS}}}label", "shapes")
        shapes.set(f"{{{INK_NS}}}groupmode", "layer")

    # Clear anything an earlier run left directly in the rois layer.
    for child in list(rois):
        if child.tag == f"{{{SVG_NS}}}path":
            rois.remove(child)

    for label, _, _, colour in specs:
        for child in list(shapes):
            if child.get(f"{{{INK_NS}}}label") == label:
                shapes.remove(child)
        group = ET.SubElement(shapes, f"{{{SVG_NS}}}g")
        group.set(f"{{{INK_NS}}}label", label)
        group.set("id", f"roi_{label}")
        paths = _contour_paths(masks[label], svg_xy, svgshape, grid,
                               flip_y, smooth)
        for i, d in enumerate(paths):
            el = ET.SubElement(group, f"{{{SVG_NS}}}path")
            el.set("d", d)
            el.set("id", f"roi_{label}_{i}")
            el.set("style", f"fill:none;stroke:{colour};stroke-width:2")
        print(f"  {label}: {len(paths)} path(s)")

    if dry_run:
        print("  (dry run — overlays.svg not written)")
        return svgfile
    tree.write(svgfile, encoding="utf-8", xml_declaration=True)
    print(f"  wrote {svgfile}")
    return svgfile


def burn_outlines_into(ds, subject, cx_subject, bids_folder, width=2,
                       specs=ROI_SPECS):
    """Paint ROI outlines into every dataset's RGB, in place.

    The belt-and-braces option. overlays.svg ROIs are the *right* way to do
    this -- toggleable, labelled, drawn over whatever is displayed -- but their
    rendering depends on the viewer honouring overlays_visible and on the
    stroke colour (pycortex forces white, which disappears over a bright map).
    Burning the outline into the RGB cannot fail to show, at the cost of not
    being switchable and of hiding the few vertices it covers.
    """
    import numpy as np
    masks = annot_masks(subject, bids_folder, specs)
    neighbours, n_vert = _adjacency(cx_subject)
    edges = []
    for label, _, _, colour in specs:
        edge = outline(masks[label], neighbours, width=width)
        rgb = tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))
        edges.append((edge, rgb))
        print(f"  burning {label}: {int(edge.sum())} vertices")

    for name, vtx in ds.items():
        for edge, (r, g, b) in edges:
            # VertexRGB.red/green/blue are Vertex objects, not arrays.
            vtx.red.data[edge] = r
            vtx.green.data[edge] = g
            vtx.blue.data[edge] = b
    return ds
