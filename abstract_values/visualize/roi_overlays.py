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
