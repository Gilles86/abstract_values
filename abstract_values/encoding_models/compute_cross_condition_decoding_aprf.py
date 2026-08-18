#!/usr/bin/env python3
"""
Cross-condition decoding test for the aPRF session-shift model.

Question: does NPCr's value code genuinely remap between the CDF and
InvCDF conditions, or does the "mode" shift fitted per session
(``aprf-session-shift``) mostly reflect noise?

For each subject and each held-out session (test session), we decode the
*real* single-trial GLMsingle betas twice, using the same voxel selection
and the same measurement-noise model (Omega, dof — fit once from the
test session's own data against its own matched tuning curves), but two
different tuning-curve sets:

  matched — mode_<test_session>   (the "correct" per-condition tuning)
  cross   — mode_<train_session>  (the OTHER session's tuning: what you'd
            get training on one condition and testing on the other)

fwhm / amplitude / baseline are shared across sessions in this model, so
only ``mode`` (the preferred CHF value) differs between matched and cross.

If the remapping is real, cross-decoding should show a systematic,
predictable bias relative to matched decoding (not just added noise).
If the mode shift is mostly fit noise, cross ≈ matched up to noise.

Output
------
  derivatives/encoding_models/aprf-session-shift/sub-<subject>/ses-<test>/func/
    sub-<subject>_ses-<test>_task-abstractvalue_mask-<roi>_nvoxels-<n>
    {noise_tag}{smooth}_desc-crossdecoded_pe.tsv

  One row per real gabor trial in the test session:
    run, trial_nr, true_value, train_condition, test_condition,
    matched_mean, matched_sd, cross_mean, cross_sd

Usage
-----
  python compute_cross_condition_decoding_aprf.py 03 --n-noise-iterations 200
  python compute_cross_condition_decoding_aprf.py 03 --roi NPCr --n-voxels 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import LogGaussianPRF
from braincoder.optimize import ResidualFitter

from abstract_values.utils.data import Subject, BIDS_FOLDER


def get_value_paradigm(sub, sessions):
    """DataFrame with a plain RangeIndex (matches the masker-transformed
    betas' default index) and columns run, trial_nr, x=value."""
    rows = []
    for session in sessions:
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values("onset")
            for _, row in run_ev[run_ev["event_type"] == "gabor"].iterrows():
                rows.append({"run": run, "trial_nr": int(row["trial_nr"]),
                             "x": np.float32(float(row["value"]))})
    return pd.DataFrame(rows)


def _load_session_shift_params(subject, smoothed, masker, bids_folder):
    ss_dir = (Path(bids_folder) / "derivatives" / "encoding_models"
              / "aprf-session-shift" / f"sub-{subject}" / "func")
    smooth = "_smoothed" if smoothed else ""

    def load(desc):
        fn = (ss_dir / f"sub-{subject}_task-abstractvalue_space-T1w"
                       f"_desc-{desc}{smooth}_pe.nii.gz")
        if not fn.exists():
            raise FileNotFoundError(f"No session-shift param: {fn}")
        import nilearn.image as nli
        return nli.load_img(str(fn))

    fwhm = masker.transform(load("fwhm")).squeeze().astype(np.float32)
    amp  = masker.transform(load("amplitude")).squeeze().astype(np.float32)
    base = masker.transform(load("baseline")).squeeze().astype(np.float32)
    r2   = pd.Series(masker.transform(load("r2")).squeeze().astype(np.float32))
    modes = {f"mode_{i}": masker.transform(load(f"mode_{i}")).squeeze().astype(np.float32)
             for i in (1, 2)}
    return fwhm, amp, base, r2, modes


def main(subject, roi="NPCr", hemi="None", n_voxels=100,
         n_noise_iterations=1000, n_values=200,
         value_min=0.5, value_max=50.0,
         bids_folder=BIDS_FOLDER, fmriprep_deriv="fmriprep",
         smoothed=False, spherical_noise=True):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)
    sessions = sorted(sub.get_sessions())
    assert len(sessions) == 2, f"expected 2 sessions, got {sessions}"
    hemi_arg = None if hemi == "None" else hemi
    mask_desc = f"{roi}{'_hemi-' + hemi if hemi_arg else ''}"
    smooth_label = "_smoothed" if smoothed else ""

    print(f"sub-{subject}  sessions={sessions}  [cross-condition decoding]")

    ref_betas = sub.get_single_trial_estimates(sessions, desc="gabor", smoothed=smoothed)
    mask_img = sub.get_roi_mask(roi=roi, hemi=hemi_arg)
    masker = NiftiMasker(mask_img=mask_img, target_affine=ref_betas.affine,
                         target_shape=ref_betas.shape[:3]).fit()

    fwhm_arr, amp_arr, base_arr, r2, modes = _load_session_shift_params(
        subject, smoothed, masker, bids_folder)

    valid = (fwhm_arr > 0) & (amp_arr != 0)
    for m in modes.values():
        valid &= m > 0
    r2_valid = r2[valid]
    sel = r2_valid.sort_values(ascending=False).index[:n_voxels]
    print(f"  {len(sel)} voxels selected (R2 >= {float(r2.loc[sel].min()):.3f})")

    stimulus_range = np.linspace(value_min, value_max, n_values, dtype=np.float32)

    out_rows = []
    for test_ses in sessions:
        train_ses = [s for s in sessions if s != test_ses][0]
        test_cond = sub.get_mapping(test_ses)
        train_cond = sub.get_mapping(train_ses)
        print(f"\n  --- test ses-{test_ses} ({test_cond})  "
              f"train ses-{train_ses} ({train_cond}) ---")

        test_paradigm = get_value_paradigm(sub, [test_ses])
        test_betas = sub.get_single_trial_estimates([test_ses], desc="gabor",
                                                     smoothed=smoothed)
        test_data = pd.DataFrame(masker.transform(test_betas).astype(np.float32))
        test_data_sel = test_data[sel]
        assert len(test_data_sel) == len(test_paradigm)

        pars_matched_full = pd.DataFrame({
            "mode": modes[f"mode_{test_ses}"], "fwhm": fwhm_arr,
            "amplitude": amp_arr, "baseline": base_arr})
        pars_cross_full = pd.DataFrame({
            "mode": modes[f"mode_{train_ses}"], "fwhm": fwhm_arr,
            "amplitude": amp_arr, "baseline": base_arr})

        # Matched model: also used to fit the noise model (best-fitting
        # reference -> cleanest residual/noise estimate).
        model_matched = LogGaussianPRF(allow_neg_amplitudes=True,
                                       parameterisation="mode_fwhm_natural")
        model_matched.parameters = pars_matched_full
        model_matched.apply_mask(sel)
        model_matched.init_pseudoWWT(test_paradigm["x"].values, model_matched.parameters)

        print(f"  fitting noise model on ses-{test_ses} real data "
              f"({n_noise_iterations} iter, spherical={spherical_noise})...")
        residfit = ResidualFitter(model_matched, test_data_sel, test_paradigm)
        omega, dof = residfit.fit(init_sigma2=1e-2, init_dof=10.0,
                                  learning_rate=0.05,
                                  max_n_iterations=n_noise_iterations,
                                  spherical=spherical_noise)
        dof_str = f"{float(dof):.1f}" if dof is not None else "None (Gaussian)"
        print(f"  noise model: dof={dof_str}")

        pdf_matched = model_matched.get_stimulus_pdf(
            test_data_sel, stimulus_range, parameters=model_matched.parameters,
            omega=omega, dof=dof, normalize=False)

        model_cross = LogGaussianPRF(allow_neg_amplitudes=True,
                                     parameterisation="mode_fwhm_natural")
        model_cross.parameters = pars_cross_full
        model_cross.apply_mask(sel)
        model_cross.init_pseudoWWT(test_paradigm["x"].values, model_cross.parameters)
        pdf_cross = model_cross.get_stimulus_pdf(
            test_data_sel, stimulus_range, parameters=model_cross.parameters,
            omega=omega, dof=dof, normalize=False)

        def _mean_sd(pdf):
            vals = pdf.to_numpy(dtype=np.float64)
            vals = vals / vals.sum(axis=1, keepdims=True)
            mean = vals @ stimulus_range
            second = vals @ (stimulus_range ** 2)
            sd = np.sqrt(np.maximum(second - mean ** 2, 0.0))
            return mean, sd

        matched_mean, matched_sd = _mean_sd(pdf_matched)
        cross_mean, cross_sd = _mean_sd(pdf_cross)

        df = pd.DataFrame({
            "run": test_paradigm["run"], "trial_nr": test_paradigm["trial_nr"],
            "true_value": test_paradigm["x"],
            "train_condition": train_cond, "test_condition": test_cond,
            "matched_mean": matched_mean, "matched_sd": matched_sd,
            "cross_mean": cross_mean, "cross_sd": cross_sd,
        })
        df["test_session"] = test_ses
        out_rows.append(df)

        mae_matched = float(np.abs(matched_mean - test_paradigm["x"].values).mean())
        mae_cross = float(np.abs(cross_mean - test_paradigm["x"].values).mean())
        print(f"  MAE matched={mae_matched:.2f} CHF   MAE cross={mae_cross:.2f} CHF")

    out = pd.concat(out_rows, ignore_index=True)
    out_dir = (bids_folder / "derivatives" / "encoding_models"
               / "aprf-session-shift" / f"sub-{subject}")
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_tag = "_noise-spherical" if spherical_noise else ""
    out_fn = (out_dir /
              f"sub-{subject}_task-abstractvalue_mask-{mask_desc}"
              f"_nvoxels-{n_voxels}{noise_tag}{smooth_label}_desc-crossdecoded_pe.tsv")
    out.to_csv(out_fn, sep="\t", index=False)
    print(f"\nsaved {out_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
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
