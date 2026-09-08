"""Per-voxel preferred values in each session, for the reorganisation question.

The cross-condition decoding says swapping the two sessions' tuning modes costs
accuracy in a value-dependent way. That is the decoder's view of the shift;
this dumps the shift itself, straight from the ``aprf-session-shift``
parameters, so the two can be compared:

  mode_1  preferred value (CHF) in session 1
  mode_2  preferred value (CHF) in session 2
  r2      joint fit quality, for selecting voxels the same way decoding does

Voxels are ranked by the joint aPRF R² and the top N kept, matching the
decoding runs' selection, so the dump describes the same population the
cross-decoding was computed on.

    python -m abstract_values.encoding_models.extract_session_shift_modes 03 \
        --roi NPCr --hemi None --out modes_sub-03.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from abstract_values.utils.data import Subject, BIDS_FOLDER


def load_modes(subject, roi="NPCr", hemi=None, n_voxels=100, smoothed=False,
               bids_folder=BIDS_FOLDER, fmriprep_deriv="fmriprep",
               model="aprf-session-shift", params=("mode_1", "mode_2"),
               select_model="aprf"):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)
    smooth = "_smoothed" if smoothed else ""
    ss = (bids_folder / "derivatives" / "encoding_models" / "aprf-session-shift"
          / f"sub-{subject}" / "func")
    betas = sub.get_single_trial_estimates(sub.get_sessions(), desc="gabor",
                                           smoothed=smoothed)
    masker = NiftiMasker(mask_img=sub.get_roi_mask(roi=roi, hemi=hemi),
                         target_affine=betas.affine,
                         target_shape=betas.shape[:3]).fit()

    def _p(desc, model):
        fn = (bids_folder / "derivatives" / "encoding_models" / model
              / f"sub-{subject}" / "func"
              / f"sub-{subject}_task-abstractvalue_space-T1w"
                f"_desc-{desc}{smooth}_pe.nii.gz")
        return masker.transform(nib.load(str(fn))).squeeze().astype(np.float32)

    df = pd.DataFrame({
        "subject": subject,
        "mode_1": _p(params[0], model), "mode_2": _p(params[1], model),
        "amplitude": _p("amplitude", model),
        "r2_shift": _p("r2", model),
        # Selection uses the JOINT fit's R², as the decoding runs do.
        "r2_joint": _p("r2", model=select_model),
    })
    df["voxel"] = np.arange(len(df))
    if n_voxels:
        df = df.nlargest(n_voxels, "r2_joint")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("subject")
    p.add_argument("--roi", default="NPCr")
    p.add_argument("--hemi", default="None")
    p.add_argument("--n-voxels", type=int, default=100)
    p.add_argument("--model", default="aprf-session-shift",
                   help="Session-shift fit to read (aprf-session-shift, or "
                        "vonmises-prf-session-shift for orientation).")
    p.add_argument("--params", nargs=2, default=["mode_1", "mode_2"],
                   help="The two per-session parameter descs (mu_1 mu_2 in "
                        "orientation space).")
    p.add_argument("--select-model", default="aprf",
                   help="Joint fit whose R2 ranks voxels (vonmises-prf for "
                        "orientation).")
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    d = load_modes(a.subject, roi=a.roi,
                   hemi=None if a.hemi == "None" else a.hemi,
                   n_voxels=a.n_voxels, smoothed=a.smoothed,
                   bids_folder=a.bids_folder, model=a.model,
                   params=tuple(a.params), select_model=a.select_model)
    d.to_csv(a.out, sep="\t", index=False)
    print(f"sub-{a.subject}: {len(d)} voxels -> {a.out}")
