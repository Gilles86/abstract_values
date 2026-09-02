"""Eccentricity-restricted visual-area masks from the Benson-14 atlas.

`create_roi_masks.py` already makes a binary `BensonV1` mask covering the whole
of V1. But the gabor is an **annulus** — 7.5 deg outer, 1.5 deg inner diameter,
so it drives 0.75–3.75 deg eccentricity and nothing else. Roughly half of V1
sees no stimulus at all, and including it dilutes every tuning estimate and
every ROI-average cvR² with cortex that cannot possibly carry a response.

This takes the fsnative Benson maps written by `infer_neuropythy_atlas.py`,
intersects the area label with an eccentricity band, and projects the result to
a T1w-space volume — same neuropythy `cortex_to_image` path the other masks use,
so it lines up with the encoding-model volumes.

Output (desc encodes the band so masks with different limits coexist)::

    derivatives/masks/sub-XX/anat/
      sub-XX_space-T1w_desc-BensonV1ecc075-375_mask.nii.gz
      sub-XX_space-T1w_desc-BensonV1ecc075-375l_mask.nii.gz
      sub-XX_space-T1w_desc-BensonV1ecc075-375r_mask.nii.gz

Usage
-----
    python -m abstract_values.surface.make_benson_eccen_mask 29
    python -m abstract_values.surface.make_benson_eccen_mask 29 --area 2 \\
        --eccen-min 0.75 --eccen-max 3.75
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from nilearn import surface

from neuropythy.freesurfer import subject as fs_subject_fn
from neuropythy.io import load, save
from neuropythy.mri import image_clear, to_image

from abstract_values.utils.data import BIDS_FOLDER

# Gabor annulus geometry from experiment/settings/default.yml:
#   size 7.5 deg, hole_size 1.5 deg (both diameters) -> radii 3.75 / 0.75.
DEFAULT_ECCEN_MIN = 0.75
DEFAULT_ECCEN_MAX = 3.75

AREA_NAMES = {1: "V1", 2: "V2", 3: "V3"}


def _band_tag(lo, hi):
    """0.75, 3.75 -> 'ecc075-375' — BIDS-safe, no dots."""
    return f"ecc{lo:g}".replace(".", "") + "-" + f"{hi:g}".replace(".", "")


def load_atlas_prop(atlas_dir, subject, prop, hemi):
    fn = (Path(atlas_dir) / f"sub-{subject}" /
          f"sub-{subject}_desc-{prop}_space-fsnative_hemi-{hemi}.func.gii")
    if not fn.exists():
        raise FileNotFoundError(
            f"{fn} — run infer_neuropythy_atlas.py for this subject first")
    return np.asarray(surface.load_surf_data(str(fn))).ravel()


def main(subject, session=1, area=1, eccen_min=DEFAULT_ECCEN_MIN,
         eccen_max=DEFAULT_ECCEN_MAX, bids_folder=BIDS_FOLDER,
         fmriprep_deriv="fmriprep"):
    bids_folder = Path(bids_folder)
    deriv = bids_folder / "derivatives"
    atlas_dir = deriv / "neuropythy_atlas"
    fs_dir = (deriv / fmriprep_deriv / "sourcedata" / "freesurfer" /
              f"sub-{subject}_ses-{session}")
    t1w = (deriv / fmriprep_deriv / f"sub-{subject}" / f"ses-{session}" /
           "anat" / f"sub-{subject}_ses-{session}_desc-preproc_T1w.nii.gz")
    out_dir = deriv / "masks" / f"sub-{subject}" / "anat"
    out_dir.mkdir(parents=True, exist_ok=True)

    roi = f"Benson{AREA_NAMES.get(area, f'A{area}')}{_band_tag(eccen_min, eccen_max)}"
    prefix = f"sub-{subject}_space-T1w_desc-{roi}"

    masks = {}
    for hemi, fs_hemi in (("L", "lh"), ("R", "rh")):
        varea = load_atlas_prop(atlas_dir, subject, "benson14Varea", hemi)
        eccen = load_atlas_prop(atlas_dir, subject, "benson14Eccen", hemi)
        in_area = np.round(varea).astype(int) == area
        in_band = (eccen >= eccen_min) & (eccen <= eccen_max)
        m = (in_area & in_band).astype(np.float32)
        masks[fs_hemi] = m
        print(f"  {hemi}: area={int(in_area.sum())} verts, "
              f"within {eccen_min}-{eccen_max} deg -> {int(m.sum())} "
              f"({100 * m.sum() / max(in_area.sum(), 1):.0f}% of the area)")

    sub = fs_subject_fn(str(fs_dir))
    im = to_image(image_clear(load(str(t1w)), fill=0.0), dtype=np.int32)

    print("Projecting to T1w volume ...", flush=True)
    vol = sub.cortex_to_image((masks["lh"], masks["rh"]), im, hemi=None,
                              method="nearest", fill=0.0)
    save(str(out_dir / f"{prefix}_mask.nii.gz"), vol)

    vol = sub.cortex_to_image(masks["lh"], im, hemi="lh", method="nearest",
                              fill=0.0)
    save(str(out_dir / f"{prefix}l_mask.nii.gz"), vol)

    zero_lh = np.zeros_like(masks["lh"])
    vol = sub.cortex_to_image((zero_lh, masks["rh"]), im, hemi=None,
                              method="nearest", fill=0.0)
    save(str(out_dir / f"{prefix}r_mask.nii.gz"), vol)

    print(f"Done: {out_dir}/{prefix}[lr]_mask.nii.gz")
    return roi


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subject")
    p.add_argument("--session", type=int, default=1)
    p.add_argument("--area", type=int, default=1,
                   help="Benson visual area code (1=V1, 2=V2, 3=V3)")
    p.add_argument("--eccen-min", type=float, default=DEFAULT_ECCEN_MIN)
    p.add_argument("--eccen-max", type=float, default=DEFAULT_ECCEN_MAX)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--fmriprep-deriv", default="fmriprep")
    a = p.parse_args()
    main(a.subject, session=a.session, area=a.area, eccen_min=a.eccen_min,
         eccen_max=a.eccen_max, bids_folder=a.bids_folder,
         fmriprep_deriv=a.fmriprep_deriv)
