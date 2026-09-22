"""Name the value-family patches: numbered clusters on an atlas-labelled flatmap.

The group winner maps (``model_winner_maps.py``) show, per vertex, whether the
value family (either aPRF) or the orientation family (either vonMises) wins
more often across subjects. Eyeballing that in the webgl viewer gives "there is
a red patch just lateral of IPS" — this turns such an impression into a list:
every contiguous value-family cluster above a size, where it sits in the
Benson-14 / Wang-15 / Desikan atlases, and how far it is from the retinotopic
IPS band.

The flatmap panels draw the same value-vs-orientation map with IPS0-3 (green),
NPC1-3 (magenta) and the Benson areas (grey) outlined, and each cluster
numbered to match the printed table.

Run in the ``pycortex2`` env.

Usage
-----
    python -m abstract_values.visualize.value_family_clusters \\
        --out notes/figures/value_family_clusters.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cortex
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.tri import Triangulation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from abstract_values.surface.make_fsaverage_atlas_masks import find_atlas_dir
from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize.group_surface_maps import (CX_FSAVERAGE,
                                                          discover_subjects)
from abstract_values.visualize.model_winner_maps import (
    CANDIDATES, blended, family_vote, prevalence_gate, surface_mask,
    winner_per_subject)

BENSON = {1: "V1", 2: "V2", 3: "V3", 4: "hV4", 5: "VO1", 6: "VO2", 7: "LO1",
          8: "LO2", 9: "TO1", 10: "TO2", 11: "V3b", 12: "V3a"}
OUTLINE_ATLAS = ["V1", "hV4", "LO", "TO1", "TO2", "V3a", "V3b"]
OUTLINE_IPS = ["IPS0", "IPS1", "IPS2", "IPS3"]


def fsaverage_graph(polys, n):
    e = np.vstack([polys[:, [0, 1]], polys[:, [1, 2]], polys[:, [2, 0]]])
    adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    return adj + adj.T


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--out", default="notes/figures/value_family_clusters.pdf")
    p.add_argument("--smoothed", action="store_true", default=True)
    p.add_argument("--unsmoothed", dest="smoothed", action="store_false")
    p.add_argument("--min-size", type=int, default=150,
                   help="Smallest cluster to report, in fsaverage vertices.")
    p.add_argument("--min-share", type=float, default=0.6,
                   help="Fraction of subjects favouring the value family.")
    p.add_argument("--max-distance", type=float, default=30.0,
                   help="Only report clusters whose nearest vertex is within "
                        "this many mm of the Wang IPS band (0 = no limit).")
    args = p.parse_args()

    deriv = Path(args.bids_folder) / "derivatives"
    subjects = discover_subjects(deriv)
    fpts, fpolys = cortex.db.get_surf(CX_FSAVERAGE, "flat", merge=True)
    gpts, gpolys = cortex.db.get_surf(CX_FSAVERAGE, "fiducial", merge=True)
    n = fpts.shape[0]
    half = n // 2
    tri = Triangulation(fpts[:, 0], fpts[:, 1], fpolys)
    adj = fsaverage_graph(gpolys, n)

    rois = cortex.utils.get_roi_verts(CX_FSAVERAGE)
    masks = {r: surface_mask(deriv, r, n) for r in OUTLINE_ATLAS + OUTLINE_IPS}
    masks["IPS"] = surface_mask(deriv, "IPS", n)
    # V1/hV4 have no atlas mask on disk — they are drawn in the fsaverage
    # overlay that ships with the pycortex store, like NPC.
    for name, m in list(masks.items()):
        if m is None and name in rois:
            m = np.zeros(n, bool)
            m[rois[name]] = True
            masks[name] = m
    for k in ("NPC1_L", "NPC2_L", "NPC3_L", "NPC1_R", "NPC2_R", "NPC3_R"):
        m = np.zeros(n, bool)
        m[rois[k]] = True
        masks[k] = m
    npc = np.logical_or.reduce([masks[k] for k in masks if k.startswith("NPC")])
    ips_tree = cKDTree(gpts[masks["IPS"]])

    atlas_dir = find_atlas_dir()
    ben = np.concatenate([
        np.asarray(nib.load(str(sorted(atlas_dir.glob(
            f"{hemi}.benson14_varea.*.mgz"))[0])).dataobj).squeeze()
        for hemi in ("lh", "rh")])
    ap, ap_names = [], None
    lab_dir = deriv / "fmriprep" / "sourcedata" / "freesurfer" / "fsaverage" / "label"
    for hemi in ("lh", "rh"):
        lab, _, names = nib.freesurfer.read_annot(str(lab_dir / f"{hemi}.aparc.annot"))
        ap.append(lab)
        ap_names = [x.decode() for x in names]
    ap = np.concatenate(ap)

    wins, signal, labels, used = winner_per_subject(deriv, subjects, CANDIDATES,
                                                    args.smoothed)
    gate = prevalence_gate(signal.mean(axis=0), 0.4, 0.25)
    ori = [i for i, l in enumerate(labels) if "vonMises" in l]
    val = [i for i, l in enumerate(labels) if i not in ori]
    fvote, fsig = family_vote(deriv, used, CANDIDATES, ori, val, args.smoothed)
    den = np.maximum(fsig.sum(axis=0), 1)
    val_share = ((fvote == 1) & fsig).sum(axis=0) / den

    cmap = mpl.colors.ListedColormap(["#25457F", "#B33A22"])
    vtx = blended((val_share > 0.5).astype(np.float32),
                  gate * np.clip(np.abs(val_share - 0.5) * 2, 0, 1),
                  CX_FSAVERAGE, -0.5, 1.5, cmap)
    im, extents = cortex.quickflat.make_flatmap_image(vtx, height=1600)

    cand = (val_share > args.min_share) & (gate > 0.5)
    idx = np.where(cand)[0]
    _, comp = connected_components(adj[cand][:, cand], directed=False)
    sizes = np.bincount(comp)
    clusters = []
    for c in np.argsort(sizes)[::-1]:
        if sizes[c] < args.min_size:
            break
        v = idx[comp == c]
        d = ips_tree.query(gpts[v])[0]
        if args.max_distance and d.min() > args.max_distance:
            continue
        if masks["IPS"][v].mean() > 0.5:
            continue
        clusters.append((v, d))

    tag = "smoothed" if args.smoothed else "unsmoothed"
    print(f"{len(clusters)} value-family clusters (>= {args.min_size} vtx, "
          f"share > {args.min_share}, outside IPS, within "
          f"{args.max_distance:.0f} mm of it) — {tag}, n={len(used)}\n")
    for rank, (v, d) in enumerate(clusters, 1):
        hemi = "L" if v.mean() < half else "R"
        bb = {BENSON[k]: int((ben[v] == k).sum()) for k in BENSON
              if (ben[v] == k).sum()}
        aa = {ap_names[k]: int((ap[v] == k).sum()) for k in np.unique(ap[v])
              if k >= 0}
        top_b = sorted(bb.items(), key=lambda kv: -kv[1])[:3]
        top_a = sorted(aa.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {rank}: {len(v):5d} vtx  {hemi}  value-share "
              f"{val_share[v].mean():.2f}  {d.min():.0f} mm from IPS")
        print("      benson: " + (", ".join(f"{k} {100 * x / len(v):.0f}%"
                                            for k, x in top_b) or "outside")
              + "   aparc: " + ", ".join(f"{k} {100 * x / len(v):.0f}%"
                                         for k, x in top_a))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
    for ax, (lo, hi, side) in zip(axes, [(0, half, "Left"), (half, n, "Right")]):
        ax.imshow(im, extent=extents, origin="lower")
        outlines = ([(r, "#0B7A3B", 1.0) for r in OUTLINE_IPS] +
                    [(r, "#3A3A3A", 0.7) for r in OUTLINE_ATLAS] +
                    [(f"NPC{i}_{side[0]}", "#B5179E", 1.0) for i in (1, 2, 3)])
        for name, colour, lw in outlines:
            m = masks.get(name)
            if m is None:
                continue
            m = m.copy()
            m[:lo] = False
            m[hi:] = False
            if not m.any():
                continue
            ax.tricontour(tri, m.astype(float), levels=[0.5], colors=colour,
                          linewidths=lw)
            c = fpts[m, :2].mean(axis=0)
            ax.text(c[0], c[1], name.split("_")[0], color=colour, fontsize=5.5,
                    fontweight="bold", ha="center", va="center")
        for rank, (v, _) in enumerate(clusters, 1):
            vv = v[(v >= lo) & (v < hi)]
            if vv.size < 60:
                continue
            m = np.zeros(n, bool)
            m[vv] = True
            ax.tricontour(tri, m.astype(float), levels=[0.5], colors="#111111",
                          linewidths=1.4)
            c = fpts[vv, :2].mean(axis=0)
            ax.text(c[0], c[1], str(rank), fontsize=8, fontweight="bold",
                    ha="center", va="center", color="#111111",
                    bbox=dict(boxstyle="circle,pad=0.12", fc="white",
                              ec="#111111", lw=0.9))
        # Frame on the occipito-parietal quarter of this hemisphere only.
        keep = np.zeros(n, bool)
        keep[lo:hi] = True
        sel = fpts[keep & (masks["IPS"] | npc | masks["hV4"] |
                           masks["TO2"] | masks["V3a"])][:, :2]
        ax.set_xlim(sel[:, 0].min() - 18, sel[:, 0].max() + 18)
        ax.set_ylim(sel[:, 1].min() - 18, sel[:, 1].max() + 18)
        ax.set_title(f"{side} hemisphere", fontsize=8)
        ax.set_axis_off()
    fig.suptitle(f"Value (red) vs orientation (blue) family — {tag}, "
                 f"n={len(used)}", fontsize=9)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=130, facecolor="white",
                bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
