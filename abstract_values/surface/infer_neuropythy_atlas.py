"""Run neuropythy's anatomy-based retinotopy atlas for one subject, and put the
result into fsaverage space so it lines up vertex-for-vertex with the encoding
model's surface maps.

Ported from retinonumeral/surface/infer_neuropythy_atlas.py. The one
abstract_values-specific change: fmriprep runs anatomy per session here, so the
FreeSurfer recon is ``sub-XX_ses-N`` rather than ``sub-XX``.

Why this matters here: `BensonV1` masks already exist via create_roi_masks.py,
but only as a binary area label. This also gives **eccentricity**, which is what
lets a V1 mask be restricted to the eccentricity band the gabor actually
stimulated instead of all of V1 — the difference between a clean tuning estimate
and one diluted by unstimulated cortex. Wang-15 additionally reaches into
parietal cortex (IPS0-5), which Benson does not.

Two steps:

1. Interpolate the **Benson-14** retinotopy template and the **Wang-15**
   max-probability atlas onto the subject's own FreeSurfer surface, using the
   recon under ``derivatives/fmriprep/sourcedata/freesurfer``. This needs **no
   functional data** -- it rides on the FreeSurfer spherical registration.
   Properties written (per hemisphere):

   - ``benson14Angle`` -- predicted polar angle in degrees. 0 = upper vertical
     meridian, 90 = horizontal meridian, 180 = lower vertical meridian. It
     always describes the **contralateral** hemifield, so the left/right sign
     of the receptive field is carried by the hemisphere, not by this number.
   - ``benson14Eccen`` -- predicted eccentricity, degrees of visual angle
   - ``benson14Sigma`` -- predicted pRF size
   - ``benson14Varea`` -- visual-area label: 1 V1, 2 V2, 3 V3, 4 hV4, 5 VO1,
     6 VO2, 7 LO1, 8 LO2, 9 TO1, 10 TO2, 11 V3b, 12 V3a; 0 = outside the
     template's coverage
   - ``wang15Mplbl``   -- Wang max-probability labels, which unlike Benson do
     extend into parietal cortex: 1 V1v ... 18 IPS0, 19 IPS1, 20 IPS2,
     21 IPS3, 22 IPS4, 23 IPS5, 24 SPL1, 25 FEF
   - ``wang15Fplbl``   -- the underlying full probability maps

2. ``mri_surf2surf`` fsnative -> fsaverage. This mirrors the round trip the
   encoding-model weight maps already took (volume -> fsnative -> fsaverage),
   so the two are blurred comparably and share vertex ordering.

Everything is written under ``derivatives/neuropythy_atlas/sub-XX/``; the
FreeSurfer tree is left untouched.

These are **anatomical predictions, not measured retinotopy**. They are
accurate for V1-V3 and degrade steadily beyond; parietal values are weak
priors at best.

Note on the implementation: we drive ``neuropythy.commands.atlas.atlas_plan``
directly rather than shelling out to ``python -m neuropythy atlas``. The CLI's
pimms argument parser silently produced an empty file list for this argument
combination; calling the plan and saving the property vectors ourselves is
deterministic and skips the volume-export machinery we don't need.

Usage::

    python retinonumeral/surface/infer_neuropythy_atlas.py 01 \
        --bids_folder /shares/zne.uzh/gdehol/ds-retinonumeral
"""
import argparse
import os
import os.path as op
import subprocess

import nibabel as nb
import numpy as np

FREESURFER_HOME = os.environ.get(
    "AV_FREESURFER_HOME",
    "/shares/zne.uzh/containers/fmriprep-25.2.3/opt/freesurfer")
FS_LICENSE = os.environ.get(
    "AV_FS_LICENSE",
    "/shares/zne.uzh/containers/freesurfer/license.txt")

ATLASES = ("benson14", "wang15")
# Categorical maps: rounded back to integers after fsaverage interpolation so
# that averaging neighbours can't invent area codes that don't exist.
LABEL_PROPS = ("varea", "mplbl", "fplbl")

HEMIS = {"lh": "L", "rh": "R"}


def _fs_env(subjects_dir):
    env = os.environ.copy()
    env["FREESURFER_HOME"] = FREESURFER_HOME
    env["SUBJECTS_DIR"] = subjects_dir
    env["FS_LICENSE"] = FS_LICENSE
    env["PATH"] = f"{FREESURFER_HOME}/bin:" + env.get("PATH", "")
    return env


def _desc(atlas, prop):
    """benson14 + angle -> 'benson14Angle' (BIDS-safe desc entity)."""
    return f"{atlas}{prop[0].upper()}{prop[1:]}"


def _save_gii(data, fn):
    nb.save(nb.gifti.GiftiImage(
        darrays=[nb.gifti.GiftiDataArray(
            np.asarray(data, dtype=np.float32))]), fn)


def fs_subject_name(subject, session):
    """abstract_values FreeSurfer subjects carry a session suffix.

    fmriprep runs anatomy per session here, so the recon directory is
    ``sub-XX_ses-N``, not ``sub-XX`` as in retinonumeral. Everything the
    atlas and mri_surf2surf need is keyed on that name.
    """
    return f"sub-{subject}_ses-{session}"


def run_atlas(subject, bids_folder, out_dir, session=1):
    """Step 1: interpolate the atlases onto the subject's native surface."""
    subjects_dir = op.join(bids_folder, "derivatives", "fmriprep",
                           "sourcedata", "freesurfer")
    fs_sub = op.join(subjects_dir, fs_subject_name(subject, session))
    if not op.exists(fs_sub):
        raise FileNotFoundError(f"No FreeSurfer recon at {fs_sub}")
    os.environ["SUBJECTS_DIR"] = subjects_dir

    from neuropythy.commands.atlas import atlas_plan

    imap = atlas_plan(argv=[fs_subject_name(subject, session)], atlases=ATLASES,
                      output_path=out_dir, create_directory=True,
                      surface_export=True, output_format="mgz",
                      overwrite=True)
    props = imap["atlas_properties"]

    written = []
    for atlas in props:
        for version in props[atlas]:
            for fs_hemi in props[atlas][version]:
                hemi = HEMIS[fs_hemi]
                for prop in props[atlas][version][fs_hemi]:
                    data = np.asarray(props[atlas][version][fs_hemi][prop])
                    if data.ndim > 1:
                        # fplbl is (n_vertices, n_labels) probabilities; keep
                        # only the argmax label, which is what's interpretable
                        # vertex-wise.
                        continue
                    fn = op.join(
                        out_dir,
                        f"sub-{subject}_desc-{_desc(atlas, prop)}"
                        f"_space-fsnative_hemi-{hemi}.func.gii")
                    _save_gii(data, fn)
                    written.append(fn)
                    print(f"  native {op.basename(fn)} "
                          f"(n={data.size}, nonzero={int(np.sum(data != 0))})")
    if not written:
        raise RuntimeError("neuropythy produced no surface properties")
    return written


def to_fsaverage(subject, bids_folder, out_dir, session=1):
    """Step 2: fsnative -> fsaverage for every native map in out_dir."""
    import glob
    subjects_dir = op.join(bids_folder, "derivatives", "fmriprep",
                           "sourcedata", "freesurfer")
    env = _fs_env(subjects_dir)

    native = sorted(glob.glob(op.join(out_dir, "*space-fsnative*.func.gii")))
    if not native:
        raise RuntimeError(f"No fsnative maps to project in {out_dir}")

    written = []
    for src in native:
        out = src.replace("fsnative", "fsaverage")
        hemi = op.basename(src).split("hemi-")[1].split(".")[0]
        fs_hemi = "lh" if hemi == "L" else "rh"
        cmd = [f"{FREESURFER_HOME}/bin/mri_surf2surf",
               "--srcsubject", fs_subject_name(subject, session),
               "--trgsubject", "fsaverage",
               "--hemi", fs_hemi,
               "--sval", src,
               "--tval", out]
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)

        if any(k in op.basename(src).lower() for k in LABEL_PROPS):
            img = nb.load(out)
            _save_gii(np.round(img.darrays[0].data), out)
        written.append(out)
        print(f"  -> {op.basename(out)}")
    return written


def main(subject, bids_folder, skip_atlas=False, session=1):
    out_dir = op.join(bids_folder, "derivatives", "neuropythy_atlas",
                      f"sub-{subject}")
    os.makedirs(out_dir, exist_ok=True)

    if not skip_atlas:
        print(f"== neuropythy atlas for sub-{subject}")
        run_atlas(subject, bids_folder, out_dir, session)

    print("== projecting to fsaverage")
    written = to_fsaverage(subject, bids_folder, out_dir, session)
    print(f"\nsub-{subject}: {len(written)} fsaverage maps in {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("subject", type=str)
    p.add_argument("--bids_folder",
                   default="/shares/zne.uzh/gdehol/ds-abstractvalue")
    p.add_argument("--session", type=int, default=1,
                   help="Session whose FreeSurfer recon to use (default 1)")
    p.add_argument("--skip_atlas", action="store_true",
                   help="Only redo the fsaverage projection.")
    args = p.parse_args()
    main(args.subject, args.bids_folder, skip_atlas=args.skip_atlas,
         session=args.session)
