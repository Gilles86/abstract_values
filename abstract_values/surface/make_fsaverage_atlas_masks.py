"""Write fsaverage-space ROI masks for the Wang-15 and Benson-14 atlases.

The project already keeps hand-drawn fsaverage labels for NPC (and its NPC1-3
subdivisions) in ``derivatives/surface_masks``. This adds the atlas ROIs that
are missing there and that the group surface maps keep needing:

* **IPS0-IPS5, SPL1** from the **Wang-15** maximum-probability atlas. Benson-14
  stops at V3a/V3b and never reaches parietal cortex, so anything asking "is
  this cluster inside retinotopic IPS, and which band" has to come from Wang.
* **LO1, LO2** (and their union ``LO``) plus **TO1, TO2, V3a, V3b** from the
  **Benson-14** ``varea`` template, which is the atlas the rest of the project
  already uses for V1-hV4.

Both ship with neuropythy as fsaverage-space ``.mgz`` on the standard
163842-vertex mesh, so no per-subject inference and no resampling is involved —
this is the group template itself, written out in the project's own label
naming so ``Subject.get_roi_mask``-style lookups and the surface scripts can
find it.

Note the two atlases disagree about LO1/LO2 (Wang's LO1 overlaps Benson's LO2
about as much as its own namesake). Benson is used here because V1-hV4 in this
project are Benson-derived; a Wang LO can be written with ``--wang-lo``.

Usage
-----
    python -m abstract_values.surface.make_fsaverage_atlas_masks
    python -m abstract_values.surface.make_fsaverage_atlas_masks --dry-run

Reads the atlases straight out of the installed neuropythy package (any conda
env that has one), so it can run from the light ``pycortex2`` env.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import nibabel as nib
import numpy as np

from abstract_values.utils.data import BIDS_FOLDER

# Wang et al. (2015) ProbAtlas_v4 label order. Verified against the atlases
# themselves rather than trusted: labels 1-7 reproduce the pycortex fsaverage
# V1/V2/V3/hV4 overlay (Dice 0.42-0.69, ordinary atlas-to-atlas agreement) and
# 13/15 land on Benson's LO2/LO1.
WANG_LABELS = {
    1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d", 7: "hV4",
    8: "VO1", 9: "VO2", 10: "PHC1", 11: "PHC2", 12: "TO2", 13: "TO1",
    14: "LO2", 15: "LO1", 16: "V3B", 17: "V3A", 18: "IPS0", 19: "IPS1",
    20: "IPS2", 21: "IPS3", 22: "IPS4", 23: "IPS5", 24: "SPL1", 25: "FEF",
}
# Benson-14 varea label order (neuropythy.vision.visual_area_names).
BENSON_LABELS = {1: "V1", 2: "V2", 3: "V3", 4: "hV4", 5: "VO1", 6: "VO2",
                 7: "LO1", 8: "LO2", 9: "TO1", 10: "TO2", 11: "V3b",
                 12: "V3a"}

WANG_WANTED = ["IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5", "SPL1"]
BENSON_WANTED = ["LO1", "LO2", "TO1", "TO2", "V3a", "V3b"]
# Unions worth having as one mask: IPS5 is ~20 vertices per hemisphere, far too
# small to carry a per-vertex win vote on its own.
UNIONS = {"LO": ("benson", ["LO1", "LO2"]),
          "IPS": ("wang", ["IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5"]),
          "IPSpost": ("wang", ["IPS0", "IPS1"]),
          "IPSant": ("wang", ["IPS2", "IPS3", "IPS4", "IPS5"])}


def find_atlas_dir(explicit=None):
    """neuropythy's bundled fsaverage surf directory, from any conda env."""
    if explicit:
        return Path(explicit)
    try:
        import neuropythy  # noqa: F401 — only used for its install location
        return (Path(neuropythy.__file__).parent / "lib" / "data" /
                "fsaverage" / "surf")
    except ImportError:
        pass
    pattern = str(Path.home() / "mambaforge" / "envs" / "*" / "lib" /
                  "python*" / "site-packages" / "neuropythy" / "lib" / "data" /
                  "fsaverage" / "surf")
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit("No neuropythy fsaverage atlases found — pass "
                         "--atlas-dir.")
    return Path(hits[0])


def load_atlas(atlas_dir, stem):
    """{hemi: (n_vertices,) int label array} for one bundled atlas."""
    out = {}
    for hemi in ("lh", "rh"):
        hits = sorted(atlas_dir.glob(f"{hemi}.{stem}.*.mgz"))
        if not hits:
            raise SystemExit(f"Missing {hemi}.{stem}.*.mgz in {atlas_dir}")
        out[hemi] = np.asarray(nib.load(str(hits[0])).dataobj).squeeze()
    return out


def write_label(mask, out_path, dry_run=False):
    """One binary fsaverage label.gii, matching the existing NPC masks."""
    if dry_run:
        print(f"  [dry-run] {out_path.name}: {int(mask.sum())} vertices")
        return
    # float32, not float64: the GIFTI spec has no float64 and current nibabel
    # refuses to write it (the older NPC masks in this folder predate that check).
    darray = nib.gifti.GiftiDataArray(mask.astype(np.float32),
                                      datatype="NIFTI_TYPE_FLOAT32")
    nib.GiftiImage(darrays=[darray]).to_filename(str(out_path))
    print(f"  {out_path.name}: {int(mask.sum())} vertices")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--atlas-dir", default=None)
    p.add_argument("--wang-lo", action="store_true",
                   help="Also write LO1/LO2 from Wang as LO1w/LO2w/LOw.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    atlas_dir = find_atlas_dir(args.atlas_dir)
    out_dir = Path(args.bids_folder) / "derivatives" / "surface_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Atlases: {atlas_dir}\nOutput:  {out_dir}\n")

    atlases = {"wang": (load_atlas(atlas_dir, "wang15_mplbl"), WANG_LABELS),
               "benson": (load_atlas(atlas_dir, "benson14_varea"),
                          BENSON_LABELS)}

    def mask_for(source, names, hemi):
        data, labels = atlases[source]
        idx = [k for k, v in labels.items() if v in names]
        missing = set(names) - {labels[k] for k in idx}
        if missing:
            raise SystemExit(f"Unknown {source} area(s): {sorted(missing)}")
        return np.isin(data[hemi], idx)

    jobs = [(n, "wang", [n]) for n in WANG_WANTED]
    jobs += [(n, "benson", [n]) for n in BENSON_WANTED]
    jobs += [(n, src, parts) for n, (src, parts) in UNIONS.items()]
    if args.wang_lo:
        jobs += [("LO1w", "wang", ["LO1"]), ("LO2w", "wang", ["LO2"]),
                 ("LOw", "wang", ["LO1", "LO2"])]

    for name, source, parts in jobs:
        for hemi, side in (("lh", "L"), ("rh", "R")):
            mask = mask_for(source, parts, hemi)
            out = (out_dir / f"desc-{name}_{side}_space-fsaverage"
                             f"_hemi-{hemi}.label.gii")
            write_label(mask, out, args.dry_run)


if __name__ == "__main__":
    main()
