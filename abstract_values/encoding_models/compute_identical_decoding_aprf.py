#!/usr/bin/env python3
"""
Decode both conditions with one IDENTICAL encoding model (no per-session
shift at all -- the "standard" aPRF fit: one mode/fwhm/amplitude/baseline
per voxel, jointly fit across both sessions).

Complements compute_cross_condition_decoding_aprf.py, which showed
cross-condition decoding (wrong-session mode) is far worse than matched
(own-session mode) decoding -- evidence some remapping is real. This script
asks the more basic question: is allowing the preferred value (mode) to
shift per condition better than forcing a single universal model on both
conditions? If session-shift (mode_1/mode_2, matched) decodes more
accurately than this identical model, that's direct, decoding-level
evidence the shift is functionally useful, not just noise.

For each session, decodes the REAL single-trial betas with the ONE
identical (standard aPRF) model -- same tuning curve regardless of
condition -- twice: once with a flat prior over the decode grid, once with
the objective (KDE of presented values) prior, mirroring
compute_matched_decoding_prior_aprf.py's flat-vs-objective comparison for
the mode-shift decoder. Same voxel selection / noise-model-fit / decode-
grid machinery as compute_cross_condition_decoding_aprf.py.

Output
------
  derivatives/encoding_models/aprf-session-shift/sub-<subject>/
    sub-<subject>_task-abstractvalue_mask-<roi>_nvoxels-<n>{noise}{smooth}
    _desc-identicaldecoded_pe.tsv

  One row per real gabor trial: run, trial_nr, true_value, test_condition,
  flat_mean, flat_sd, objective_mean, objective_sd, test_session
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from nilearn import image as nli

from braincoder.models import LogGaussianPRF
from braincoder.optimize import ResidualFitter

from abstract_values.utils.data import Subject, BIDS_FOLDER
from abstract_values.encoding_models.compute_cross_condition_decoding_aprf import (
    get_value_paradigm,
)
from abstract_values.encoding_models.compute_matched_decoding_prior_aprf import (
    _objective_prior,
)


def main(subject, roi="NPCr", hemi="None", n_voxels=100,
         n_noise_iterations=1000, n_values=200,
         value_min=0.5, value_max=50.0,
         bids_folder=BIDS_FOLDER, fmriprep_deriv="fmriprep",
         smoothed=False, spherical_noise=True):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)
    sessions = sorted(sub.get_sessions())
    assert len(sessions) == 2
    hemi_arg = None if hemi == "None" else hemi
    mask_desc = f"{roi}{'_hemi-' + hemi if hemi_arg else ''}"
    smooth_label = "_smoothed" if smoothed else ""

    print(f"sub-{subject}  sessions={sessions}  [IDENTICAL (standard aPRF) decoder]")

    ref_betas = sub.get_single_trial_estimates(sessions, desc="gabor", smoothed=smoothed)
    mask_img = sub.get_roi_mask(roi=roi, hemi=hemi_arg)
    masker = NiftiMasker(mask_img=mask_img, target_affine=ref_betas.affine,
                         target_shape=ref_betas.shape[:3]).fit()

    # Standard (fully joint, no session-shift) aPRF params -- ONE model for
    # both conditions.
    std_dir = (bids_folder / "derivatives" / "encoding_models" / "aprf"
               / f"sub-{subject}" / "func")
    smooth = "_smoothed" if smoothed else ""

    def load(desc):
        fn = (std_dir / f"sub-{subject}_task-abstractvalue_space-T1w"
                        f"_desc-{desc}{smooth}_pe.nii.gz")
        return nli.load_img(str(fn))

    mode = masker.transform(load("mode")).squeeze().astype(np.float32)
    fwhm = masker.transform(load("fwhm")).squeeze().astype(np.float32)
    amp = masker.transform(load("amplitude")).squeeze().astype(np.float32)
    base = masker.transform(load("baseline")).squeeze().astype(np.float32)
    r2 = pd.Series(masker.transform(load("r2")).squeeze().astype(np.float32))

    valid = (mode > 0) & (fwhm > 0) & (amp != 0)
    r2_valid = r2[valid]
    sel = r2_valid.sort_values(ascending=False).index[:n_voxels]
    print(f"  {len(sel)} voxels selected (R2 >= {float(r2.loc[sel].min()):.3f}, "
          f"standard joint-fit r2)")

    pars_full = pd.DataFrame({"mode": mode, "fwhm": fwhm, "amplitude": amp, "baseline": base})

    stimulus_range = np.linspace(value_min, value_max, n_values, dtype=np.float32)

    out_rows = []
    for test_ses in sessions:
        test_cond = sub.get_mapping(test_ses)
        print(f"\n  --- test ses-{test_ses} ({test_cond}) ---")

        test_paradigm = get_value_paradigm(sub, [test_ses])
        test_betas = sub.get_single_trial_estimates([test_ses], desc="gabor",
                                                     smoothed=smoothed)
        test_data = pd.DataFrame(masker.transform(test_betas).astype(np.float32))
        test_data_sel = test_data[sel]
        assert len(test_data_sel) == len(test_paradigm)

        model = LogGaussianPRF(allow_neg_amplitudes=True,
                               parameterisation="mode_fwhm_natural")
        model.parameters = pars_full
        model.apply_mask(sel)
        model.init_pseudoWWT(test_paradigm["x"].values, model.parameters)

        print(f"  fitting noise model on ses-{test_ses} real data "
              f"({n_noise_iterations} iter, spherical={spherical_noise})...")
        residfit = ResidualFitter(model, test_data_sel, test_paradigm)
        omega, dof = residfit.fit(init_sigma2=1e-2, init_dof=10.0,
                                  learning_rate=0.05,
                                  max_n_iterations=n_noise_iterations,
                                  spherical=spherical_noise)

        pdf = model.get_stimulus_pdf(test_data_sel, stimulus_range,
                                     parameters=model.parameters,
                                     omega=omega, dof=dof, normalize=False)
        pdf_vals = pdf.to_numpy(dtype=np.float64)
        pdf_vals = pdf_vals / pdf_vals.sum(axis=1, keepdims=True)  # flat-prior posterior

        prior = _objective_prior(test_paradigm["x"].values, stimulus_range)
        pdf_obj = pdf_vals * prior[np.newaxis, :]
        pdf_obj = pdf_obj / pdf_obj.sum(axis=1, keepdims=True)

        def _mean_sd(vals):
            mean = vals @ stimulus_range
            second = vals @ (stimulus_range ** 2)
            sd = np.sqrt(np.maximum(second - mean ** 2, 0.0))
            return mean, sd

        flat_mean, flat_sd = _mean_sd(pdf_vals)
        obj_mean, obj_sd = _mean_sd(pdf_obj)

        df = pd.DataFrame({
            "run": test_paradigm["run"], "trial_nr": test_paradigm["trial_nr"],
            "true_value": test_paradigm["x"], "test_condition": test_cond,
            "flat_mean": flat_mean, "flat_sd": flat_sd,
            "objective_mean": obj_mean, "objective_sd": obj_sd,
        })
        df["test_session"] = test_ses
        out_rows.append(df)

        mae_flat = float(np.abs(flat_mean - test_paradigm["x"].values).mean())
        mae_obj = float(np.abs(obj_mean - test_paradigm["x"].values).mean())
        print(f"  MAE identical flat={mae_flat:.2f} CHF  objective={mae_obj:.2f} CHF   "
              f"mean SD flat={flat_sd.mean():.2f}  objective={obj_sd.mean():.2f}")

    out = pd.concat(out_rows, ignore_index=True)
    out_dir = (bids_folder / "derivatives" / "encoding_models"
               / "aprf-session-shift" / f"sub-{subject}")
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_tag = "_noise-spherical" if spherical_noise else ""
    out_fn = (out_dir /
              f"sub-{subject}_task-abstractvalue_mask-{mask_desc}"
              f"_nvoxels-{n_voxels}{noise_tag}{smooth_label}_desc-identicaldecoded_pe.tsv")
    out.to_csv(out_fn, sep="\t", index=False)
    print(f"\nsaved {out_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject")
    parser.add_argument("--roi", default="NPCr")
    parser.add_argument("--hemi", default="None")
    parser.add_argument("--n-voxels", type=int, default=100)
    parser.add_argument("--n-noise-iterations", type=int, default=1000)
    parser.add_argument("--n-values", type=int, default=200)
    parser.add_argument("--value-min", type=float, default=0.5)
    parser.add_argument("--value-max", type=float, default=50.0)
    parser.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    parser.add_argument("--fmriprep-deriv", default="fmriprep")
    parser.add_argument("--smoothed", action="store_true")
    sph = parser.add_mutually_exclusive_group()
    sph.add_argument("--spherical-noise", dest="spherical_noise",
                     action="store_true", default=True)
    sph.add_argument("--no-spherical-noise", dest="spherical_noise",
                     action="store_false")
    args = parser.parse_args()
    main(args.subject, roi=args.roi, hemi=args.hemi, n_voxels=args.n_voxels,
         n_noise_iterations=args.n_noise_iterations, n_values=args.n_values,
         value_min=args.value_min, value_max=args.value_max,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed, spherical_noise=args.spherical_noise)
