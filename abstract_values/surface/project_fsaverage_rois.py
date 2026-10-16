"""Project many fsaverage surface labels to per-subject T1w masks in one pass.

``get_surface_roi_mask.py`` does one ROI at a time (mri_surf2surf per hemi,
then one neuropythy ``cortex_to_image`` per output file, ~1 min per ROI per
subject). This does the same projection for a whole list of ROIs at once:

1. fsaverage -> fsnative by nearest neighbour on ``?h.sphere.reg`` (what
   mri_surf2surf's nearest-neighbour mapping does too),
2. every ROI becomes one bit of an integer per vertex (bit 0 cortex, bit 1
   right hemisphere, bit 2+ the ROIs), projected with a single
   ``cortex_to_image(method='nearest')`` call (~20 s per subject),
3. the bits are split back into bilateral / left / right masks.

Output naming matches ``get_surface_roi_mask.py``:
  derivatives/masks/sub-<s>/anat/sub-<s>_space-T1w_desc-<roi>[l|r]_mask.nii.gz

ROIs
----
Atlas ROIs from ``derivatives/surface_masks`` (Wang-15 IPS, Benson-14 the
rest; unions as listed in ``ATLAS``), the high-preferred-value clusters from
``make_value_cluster_mask.py`` and, when present, each subject's own
leave-one-subject-out version of those clusters (``desc-<name>loso``) -- the
non-circular one to analyse.

Usage (abstract_values env):
    python -m abstract_values.surface.project_fsaverage_rois            # all subjects
    python -m abstract_values.surface.project_fsaverage_rois 07 pil01
    python -m abstract_values.surface.project_fsaverage_rois --check     # Dice vs existing NPC masks
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from abstract_values.utils.data import BIDS_FOLDER

# output desc -> fsaverage atlas labels to union
ATLAS = {"hV4": ["hV4"], "LO": ["LO1", "LO2"], "TO": ["TO1", "TO2"],
         "V3ab": ["V3a", "V3b"], "IPS0": ["IPS0"], "IPS12": ["IPS1", "IPS2"],
         "IPS345": ["IPS3", "IPS4", "IPS5"],
         # primary motor cortex from the pycortex fsaverage overlay (control region)
         "M1": ["M1"]}
CLUSTERS = ["valueV3aIPS0", "valueCS"]


def read_label(path, n):
    if not path.exists():
        return np.zeros(n, bool)
    return np.asarray(nib.load(str(path)).darrays[0].data) > 0


def fsaverage_labels(deriv, subject, n):
    """{desc: (lh_bool, rh_bool)} on fsaverage for this subject."""
    root = deriv / "surface_masks"

    def pair(name, base=root):
        return tuple(read_label(base / f"desc-{name}_{H}_space-fsaverage_hemi-{h}.label.gii", n)
                     for H, h in (("L", "lh"), ("R", "rh")))

    out = {}
    for desc, parts in ATLAS.items():
        ps = [pair(p) for p in parts]
        out[desc] = (np.logical_or.reduce([p[0] for p in ps]),
                     np.logical_or.reduce([p[1] for p in ps]))
    for c in CLUSTERS:
        out[c] = pair(c)
        loso = pair(f"{c}loso", root / "loso" / f"sub-{subject}")
        if loso[0].any() or loso[1].any():
            out[f"{c}loso"] = loso
    return {k: v for k, v in out.items() if v[0].any() or v[1].any()}


def project_subject(subject, bids_folder, check=False):
    import neuropythy as ny
    from neuropythy.mri import image_clear, to_image

    bids_folder = Path(bids_folder)
    deriv = bids_folder / "derivatives"
    sd = deriv / "fmriprep" / "sourcedata" / "freesurfer"
    fs_dir = sd / f"sub-{subject}_ses-1"
    t1w = (deriv / "fmriprep" / f"sub-{subject}" / "ses-1" / "anat" /
           f"sub-{subject}_ses-1_desc-preproc_T1w.nii.gz")
    if not (fs_dir.exists() and t1w.exists()):
        print(f"sub-{subject}: missing freesurfer or T1w, skipped")
        return
    nv = 163842
    labels = fsaverage_labels(deriv, subject, nv)
    if check:
        labels["NPCcheck"] = tuple(read_label(
            deriv / "surface_masks" / f"desc-NPC_{H}_space-fsaverage_hemi-{h}.label.gii", nv)
            for H, h in (("L", "lh"), ("R", "rh")))
    names = list(labels)
    assert len(names) <= 29

    codes = []
    for hi, h in enumerate(("lh", "rh")):
        fsa = nib.freesurfer.read_geometry(str(sd / "fsaverage" / "surf" / f"{h}.sphere.reg"))[0]
        nat = nib.freesurfer.read_geometry(str(fs_dir / "surf" / f"{h}.sphere.reg"))[0]
        nn = cKDTree(fsa).query(nat)[1]
        code = np.full(len(nat), 1 | (hi << 1), np.int32)
        for k, name in enumerate(names):
            code |= labels[name][hi][nn].astype(np.int32) << (k + 2)
        codes.append(code)

    sub = ny.freesurfer_subject(str(fs_dir))
    im = to_image(image_clear(ny.load(str(t1w)), fill=0.0), dtype=np.int32)
    vol = sub.cortex_to_image(tuple(codes), im, hemi=None, method="nearest", fill=0)
    data = np.asarray(vol.dataobj).astype(np.int64)
    cortex_ = (data & 1) > 0
    right = (data & 2) > 0

    out_dir = deriv / "masks" / f"sub-{subject}" / "anat"
    out_dir.mkdir(parents=True, exist_ok=True)
    msg = []
    for k, name in enumerate(names):
        m = cortex_ & ((data >> (k + 2)) & 1 > 0)
        if name == "NPCcheck":
            ref = out_dir / f"sub-{subject}_space-T1w_desc-NPC_mask.nii.gz"
            if ref.exists():
                r = np.asarray(nib.load(str(ref)).dataobj).reshape(m.shape) > 0
                dice = 2 * (r & m).sum() / (r.sum() + m.sum())
                msg.append(f"NPC Dice vs existing {dice:.2f}")
            continue
        for suffix, mm in (("", m), ("l", m & ~right), ("r", m & right)):
            img = nib.Nifti1Image(mm.astype(np.uint8), vol.affine)
            img.to_filename(str(out_dir / f"sub-{subject}_space-T1w_desc-{name}{suffix}_mask.nii.gz"))
        msg.append(f"{name} {int(m.sum())}")
    print(f"sub-{subject}: " + ", ".join(msg), flush=True)


def all_subjects(bids_folder):
    sd = Path(bids_folder) / "derivatives" / "fmriprep" / "sourcedata" / "freesurfer"
    subs = [p.name.removeprefix("sub-").removesuffix("_ses-1")
            for p in sorted(sd.glob("sub-*_ses-1"))]
    return subs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subjects", nargs="*")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--check", action="store_true",
                   help="Also project NPC and report Dice against the existing mask.")
    a = p.parse_args()
    for s in a.subjects or all_subjects(a.bids_folder):
        project_subject(s, a.bids_folder, a.check)
