#!/usr/bin/env python3
"""
Permutation-null control for compute_cross_condition_decoding_aprf.py.

The cross-condition decoder (wrong-session ``mode``, shared fwhm/amplitude/
baseline) showed a shallower-than-matched but non-zero relationship to true
value. Because fwhm/amplitude/baseline are shared across sessions, a
log-Gaussian tuning curve's flanks stay monotonic even when centered on the
wrong mode -- so some residual signal could be pure curve-shape geometry,
not a genuinely preserved per-voxel code.

This script decodes each test session with ``n_shuffles`` random
permutations of the train-session mode values across the *same* selected
voxels (same marginal mode distribution, same shared fwhm/amp/baseline, no
per-voxel spatial correspondence). If real cross-decoding tracks true value
better than this shuffled null, that is evidence some real invariant
structure survives the condition switch (incomplete/partial remapping). If
real cross ~= shuffled null, the residual slope is just tuning-curve
geometry, and the remapping looks functionally total.

Reuses the noise model (omega, dof) fit once per test session against the
*matched* model -- shuffling only changes which tuning curve is used to
interpret the response, not the measurement-noise characterization.

Output
------
  derivatives/encoding_models/aprf-session-shift/sub-<subject>/ses-<test>/func/
    sub-<subject>_ses-<test>_task-abstractvalue_mask-<roi>_nvoxels-<n>
    {noise_tag}{smooth}_desc-crossdecodedshuffled_pe.tsv

  One row per (real trial x shuffle): run, trial_nr, true_value,
  train_condition, test_condition, shuffle_idx, shuffled_mean, shuffled_sd
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
from abstract_values.encoding_models.compute_cross_condition_decoding_aprf import (
    get_value_paradigm, _load_session_shift_params,
)


def main(subject, roi="NPCr", hemi="None", n_voxels=100,
         n_noise_iterations=1000, n_values=200,
         value_min=0.5, value_max=50.0, n_shuffles=20, seed=0,
         bids_folder=BIDS_FOLDER, fmriprep_deriv="fmriprep",
         smoothed=False, spherical_noise=True):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)
    sessions = sorted(sub.get_sessions())
    assert len(sessions) == 2
    hemi_arg = None if hemi == "None" else hemi
    mask_desc = f"{roi}{'_hemi-' + hemi if hemi_arg else ''}"
    smooth_label = "_smoothed" if smoothed else ""
    rng = np.random.default_rng(seed)

    print(f"sub-{subject}  sessions={sessions}  [shuffled cross-condition null]")

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
              f"train ses-{train_ses} ({train_cond}) [shuffled] ---")

        test_paradigm = get_value_paradigm(sub, [test_ses])
        test_betas = sub.get_single_trial_estimates([test_ses], desc="gabor",
                                                     smoothed=smoothed)
        test_data = pd.DataFrame(masker.transform(test_betas).astype(np.float32))
        test_data_sel = test_data[sel]
        assert len(test_data_sel) == len(test_paradigm)

        pars_matched_full = pd.DataFrame({
            "mode": modes[f"mode_{test_ses}"], "fwhm": fwhm_arr,
            "amplitude": amp_arr, "baseline": base_arr})

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

        # Train-session mode values on the SAME selected voxels (post-mask
        # order matches model_matched.parameters row order == sel order).
        train_mode_sel = modes[f"mode_{train_ses}"][sel]
        fwhm_sel = fwhm_arr[sel]; amp_sel = amp_arr[sel]; base_sel = base_arr[sel]

        def _mean_sd(pdf):
            vals = pdf.to_numpy(dtype=np.float64)
            vals = vals / vals.sum(axis=1, keepdims=True)
            mean = vals @ stimulus_range
            second = vals @ (stimulus_range ** 2)
            sd = np.sqrt(np.maximum(second - mean ** 2, 0.0))
            return mean, sd

        for shuf_i in range(n_shuffles):
            perm = rng.permutation(len(train_mode_sel))
            pars_shuf = pd.DataFrame({
                "mode": train_mode_sel[perm], "fwhm": fwhm_sel,
                "amplitude": amp_sel, "baseline": base_sel})
            model_shuf = LogGaussianPRF(allow_neg_amplitudes=True,
                                        parameterisation="mode_fwhm_natural")
            model_shuf.parameters = pars_shuf
            pdf_shuf = model_shuf.get_stimulus_pdf(
                test_data_sel, stimulus_range, parameters=model_shuf.parameters,
                omega=omega, dof=dof, normalize=False)
            shuf_mean, shuf_sd = _mean_sd(pdf_shuf)

            df = pd.DataFrame({
                "run": test_paradigm["run"], "trial_nr": test_paradigm["trial_nr"],
                "true_value": test_paradigm["x"],
                "train_condition": train_cond, "test_condition": test_cond,
                "shuffle_idx": shuf_i,
                "shuffled_mean": shuf_mean, "shuffled_sd": shuf_sd,
            })
            df["test_session"] = test_ses
            out_rows.append(df)

        mae_shuf = float(np.mean([
            np.abs(d["shuffled_mean"].values - d["true_value"].values).mean()
            for d in out_rows if d["test_session"].iloc[0] == test_ses]))
        print(f"  mean MAE across {n_shuffles} shuffles = {mae_shuf:.2f} CHF")

    out = pd.concat(out_rows, ignore_index=True)
    out_dir = (bids_folder / "derivatives" / "encoding_models"
               / "aprf-session-shift" / f"sub-{subject}")
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_tag = "_noise-spherical" if spherical_noise else ""
    out_fn = (out_dir /
              f"sub-{subject}_task-abstractvalue_mask-{mask_desc}"
              f"_nvoxels-{n_voxels}{noise_tag}{smooth_label}_desc-crossdecodedshuffled_pe.tsv")
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
    parser.add_argument("--n-shuffles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
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
         n_shuffles=args.n_shuffles, seed=args.seed,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed, spherical_noise=args.spherical_noise)
