#!/usr/bin/env python3
"""Expected decoded VALUE from the weighted log-Gaussian basis — per session.

The value-space twin of ``compute_expected_decoded_orientation_vonmises.py``,
and the piece the expected-uncertainty analysis was missing: the existing value
EU script reads *single-bell* aPRF (session-shift) parameters, so there was no
way to ask what the basis model — the one the sweeps actually settled, and the
one the decoding runs on — says about precision across value space.

Statistics are linear here, not circular: value is a bounded interval in CHF,
not a periodic axis, so the posterior summary is the plain probability-weighted
mean and SD over the grid.

For each session:
  1. Fit the joint basis weights (closed-form ridge at the project alpha).
  2. Select top-N voxels by joint R².
  3. Re-fit the residual noise model on THIS session's betas.
  4. For every presented CHF value, simulate ``n_simulations`` noisy response
     vectors, decode each via ``get_stimulus_pdf``, take the posterior mean.
  5. Aggregate per true value: mean_E, sd_E, mean_error, mean_abs_error.

Fitting the weights jointly while re-fitting the noise per session is the same
choice the orientation script makes, and it is the point: a voxel whose value
tuning survives the mapping flip is fit by one set of weights, so any
per-session difference in expected uncertainty comes from the noise, not from
letting the tuning move.

Output (one TSV per session) at
  derivatives/encoding_models/aprf-weighted/sub-<S>/ses-<i>/func/
    sub-<S>_ses-<i>_task-abstractvalue_mask-<mask>
    _nvoxels-<n>_nsims-<n>[_smoothed]_desc-expected_decoded_value_pe.tsv

Usage
-----
  python compute_expected_decoded_value_weighted.py 03 --roi NPCr --hemi None
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import LogGaussianPRF
from braincoder.optimize import ResidualFitter, WeightFitter
from braincoder.utils import get_rsq

from abstract_values.encoding_models.decode_value import (
    get_value_paradigm, make_value_basis_parameters)
from abstract_values.encoding_models.ridge_alpha import (
    DEFAULT_RIDGE_ALPHA, enforce_default_alpha)
from abstract_values.utils.data import Subject, BIDS_FOLDER


def posterior_mean_sd(pdf_rows: np.ndarray, grid: np.ndarray):
    """Probability-weighted mean and SD of each posterior row over ``grid``."""
    p = np.clip(pdf_rows, 0.0, None)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    mean = p @ grid
    var = p @ (grid ** 2) - mean ** 2
    return mean, np.sqrt(np.clip(var, 0.0, None))


def aggregate_per_stimulus(true_arr, decoded_arr, stim_grid):
    """Per true value, aggregate across simulations."""
    rows = []
    errors = decoded_arr - true_arr
    for x in stim_grid:
        mask = true_arr == x
        if not mask.any():
            continue
        dec, err = decoded_arr[mask], errors[mask]
        rows.append({
            "value":          float(x),
            "mean_E":         float(dec.mean()),
            "var_E":          float(dec.var()),
            "sd_E":           float(dec.std()),
            "mean_error":     float(err.mean()),
            "mean_abs_error": float(np.abs(err).mean()),
            "n_sims":         int(mask.sum()),
        })
    return pd.DataFrame(rows)


def simulate_decode_session(model, basis_pars, weights_sel, omega, dof,
                            stim_grid, n_simulations, batch_stimuli=25):
    """Simulate noisy responses per stimulus and decode them back."""
    stim_df_full = pd.DataFrame({"x": stim_grid.astype(np.float32)})
    stim_df_full.index.name = "stimulus"
    true_all, decoded_all = [], []
    for start in range(0, len(stim_grid), batch_stimuli):
        stop = min(start + batch_stimuli, len(stim_grid))
        stim_batch = stim_df_full.iloc[start:stop].copy()
        sim_data = model.simulate(stim_batch, parameters=basis_pars,
                                  weights=weights_sel, noise=omega, dof=dof,
                                  n_repeats=n_simulations)
        pdf = model.get_stimulus_pdf(sim_data, parameters=basis_pars,
                                     weights=weights_sel, omega=omega, dof=dof,
                                     stimulus_range=stim_grid, normalize=True)
        dec, _ = posterior_mean_sd(pdf.values, stim_grid)

        idx = sim_data.index
        if isinstance(idx, pd.MultiIndex):
            stim_lvl = next(n for n in idx.names if n in ("stimulus", "value"))
            keys = idx.get_level_values(stim_lvl)
        else:
            keys = idx
        true_vals = stim_batch["x"].reindex(keys).to_numpy()

        true_all.append(true_vals)
        decoded_all.append(dec)
        print(f"    [stim {start}:{stop}/{len(stim_grid)}] "
              f"simulated+decoded {(stop - start) * n_simulations} trials",
              flush=True)
    return np.concatenate(true_all), np.concatenate(decoded_all)


def main(subject, sessions=None, roi="NPCr", hemi="None", n_voxels=100,
         n_basis=8, basis_fwhm=None, weight_alpha=DEFAULT_RIDGE_ALPHA,
         n_simulations=1000, n_noise_iterations=1000, batch_stimuli=25,
         bids_folder=BIDS_FOLDER, fmriprep_deriv="fmriprep",
         smoothed=False, spherical_noise=True, per_session_weights=False):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)
    sessions = sorted(sessions or sub.get_sessions())
    smooth_label = "_smoothed" if smoothed else ""
    hemi_arg = None if hemi == "None" else hemi
    mask_desc = f"{roi}{'_hemi-' + hemi if hemi_arg else ''}"
    print(f"sub-{subject}  sessions={sessions}  "
          f"[weighted-basis expected-decoded-value simulation]")

    paradigm = get_value_paradigm(sub, sessions).reset_index(drop=True)
    value_min, value_max = float(paradigm["x"].min()), float(paradigm["x"].max())
    print(f"  {len(paradigm)} gabor trials  value {value_min:.1f}-{value_max:.1f} CHF")

    betas_img = sub.get_single_trial_estimates(sessions, desc="gabor",
                                               smoothed=smoothed)
    mask_img = sub.get_roi_mask(roi=roi, hemi=hemi_arg)
    masker = NiftiMasker(mask_img=mask_img, target_affine=betas_img.affine,
                         target_shape=betas_img.shape[:3]).fit()
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32))
    print(f"  {data.shape[1]} voxels in mask ({mask_desc})")

    basis_pars = make_value_basis_parameters(n_basis, value_min, value_max,
                                             fwhm=basis_fwhm)
    print(f"  {n_basis} log-Gaussian basis functions, "
          f"fwhm={float(basis_pars['fwhm'].iloc[0]):.2f} CHF, alpha={weight_alpha}")

    model = LogGaussianPRF(parameterisation="mode_fwhm_natural")
    weights = WeightFitter(model, basis_pars, data, paradigm).fit(
        alpha=weight_alpha)
    pred = pd.DataFrame(model.basis_predictions(paradigm, basis_pars)
                        @ weights.values,
                        index=data.index, columns=data.columns)
    r2 = get_rsq(data, pred)

    if n_voxels == 0:
        sel = r2[r2 > 0].index
        sel_tag = "nvoxels-0"
        print(f"  {len(sel)} voxels selected (all R² > 0)")
    else:
        sel = r2.sort_values(ascending=False).index[:n_voxels]
        sel_tag = f"nvoxels-{n_voxels}"
        print(f"  {len(sel)} voxels selected  (R² ≥ {float(r2.loc[sel].min()):.3f})")
    if len(sel) == 0:
        raise SystemExit(f"sub-{subject}: no voxel has R² > 0 in {mask_desc}.")
    weights_sel = weights[sel]

    # Simulate only values that were actually presented — asking the decoder
    # for stimuli the encoder never saw inflates SD as an extrapolation
    # artefact (same reasoning as the orientation script's trained grid).
    stim_grid = np.asarray(sorted(set(np.round(paradigm["x"].values, 5))),
                           dtype=np.float32)
    print(f"  simulation grid: {len(stim_grid)} presented values "
          f"({stim_grid.min():.1f}-{stim_grid.max():.1f} CHF)")

    for ses_i in sessions:
        cond = sub.get_mapping(ses_i)
        print(f"\n  --- session {ses_i} (condition={cond}) ---", flush=True)
        ses_paradigm = get_value_paradigm(sub, [ses_i]).reset_index(drop=True)
        ses_betas = sub.get_single_trial_estimates([ses_i], desc="gabor",
                                                   smoothed=smoothed)
        ses_data = pd.DataFrame(masker.transform(ses_betas).astype(np.float32))

        # Per-session tuning turns this from a shape into a test: with the
        # weights shared, one curve in value space is built in, so the two
        # conditions cannot disagree on either axis. Voxel selection stays on
        # the joint fit, so the voxel set is the same in both conditions.
        ses_weights = weights_sel
        if per_session_weights:
            w_ses = WeightFitter(model, basis_pars, ses_data[sel],
                                 ses_paradigm).fit(alpha=weight_alpha)
            ses_weights = w_ses
            print("  refitted weights on this session only")

        print(f"  fitting noise model ({n_noise_iterations} iter)…", flush=True)
        omega, dof = ResidualFitter(
            model, ses_data[sel], ses_paradigm, parameters=basis_pars,
            weights=ses_weights).fit(
                init_sigma2=1e-2, init_dof=10.0, learning_rate=0.05,
                max_n_iterations=n_noise_iterations, spherical=spherical_noise)
        print(f"  noise model: dof="
              f"{f'{float(dof):.1f}' if dof is not None else 'None (Gaussian)'}")

        print(f"  simulating {n_simulations} repeats × {len(stim_grid)} values"
              f" ({n_simulations * len(stim_grid)} trials)…", flush=True)
        true_arr, decoded_arr = simulate_decode_session(
            model, basis_pars, ses_weights, omega, dof, stim_grid,
            n_simulations, batch_stimuli=batch_stimuli)
        agg = aggregate_per_stimulus(true_arr, decoded_arr, stim_grid)

        out_dir = (bids_folder / "derivatives" / "encoding_models"
                   / "aprf-weighted" / f"sub-{subject}" / f"ses-{ses_i}" / "func")
        out_dir.mkdir(parents=True, exist_ok=True)
        noise_tag = "_noise-spherical" if spherical_noise else ""
        ses_tag = "_perses" if per_session_weights else ""
        out_fn = (out_dir /
                  f"sub-{subject}_ses-{ses_i}_task-abstractvalue"
                  f"_mask-{mask_desc}_{sel_tag}_nsims-{n_simulations}"
                  f"{noise_tag}{ses_tag}{smooth_label}"
                  f"_desc-expected_decoded_value_pe.tsv")
        agg.insert(0, "condition", cond)
        agg.insert(0, "session", ses_i)
        agg.insert(0, "subject", subject)
        agg.to_csv(out_fn, sep="\t", index=False)
        print(f"  saved {out_fn}")
        print(f"  mean sd_E = {agg['sd_E'].mean():.3f} CHF, "
              f"mean |error| = {agg['mean_abs_error'].mean():.3f} CHF")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subject")
    p.add_argument("--sessions", type=int, nargs="+", default=None)
    p.add_argument("--roi", default="NPCr")
    p.add_argument("--hemi", default="None")
    p.add_argument("--n-voxels", type=int, default=100)
    p.add_argument("--n-basis", type=int, default=8)
    p.add_argument("--basis-fwhm", type=float, default=None,
                   help="Basis FWHM in CHF (default: 2x inter-basis spacing)")
    p.add_argument("--weight-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    p.add_argument("--allow-nondefault-alpha", action="store_true")
    p.add_argument("--n-simulations", type=int, default=1000)
    p.add_argument("--n-noise-iterations", type=int, default=1000)
    p.add_argument("--batch-stimuli", type=int, default=25)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--fmriprep-deriv", default="fmriprep")
    p.add_argument("--per-session-weights", action="store_true",
                   help="Refit the basis weights within each session instead "
                        "of sharing them, so the two conditions can disagree "
                        "on either stimulus axis.")
    p.add_argument("--smoothed", action="store_true")
    sph = p.add_mutually_exclusive_group()
    sph.add_argument("--spherical-noise", dest="spherical_noise",
                     action="store_true", default=True)
    sph.add_argument("--no-spherical-noise", dest="spherical_noise",
                     action="store_false")
    args = p.parse_args()
    enforce_default_alpha(args.weight_alpha, args.allow_nondefault_alpha,
                          "weighted-basis value EU")
    main(args.subject, sessions=args.sessions, roi=args.roi, hemi=args.hemi,
         n_voxels=args.n_voxels, n_basis=args.n_basis,
         basis_fwhm=args.basis_fwhm, weight_alpha=args.weight_alpha,
         n_simulations=args.n_simulations,
         n_noise_iterations=args.n_noise_iterations,
         batch_stimuli=args.batch_stimuli, bids_folder=args.bids_folder,
         per_session_weights=args.per_session_weights,
         fmriprep_deriv=args.fmriprep_deriv, smoothed=args.smoothed,
         spherical_noise=args.spherical_noise)
