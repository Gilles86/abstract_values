#!/usr/bin/env python3
"""
Expected decoded value from the aPRF (session-shift) model — per-value.

Mirrors ``neural_priors/encoding_model/get_expected_uncertainty.py``.
For each session's mode parameters (i.e. each condition), we:

  1. Load the session-shift PRF params and select the top-N voxels by R².
  2. Fit a Student-t residual noise model (ResidualFitter) on that
     session's BOLD betas.
  3. For each value in a dense grid, simulate ``n_simulations`` noisy
     response vectors from the encoding model + noise model.
  4. Compute the posterior P(value | response) for each simulated trial
     via ``model.get_stimulus_pdf`` and the posterior expected value.
  5. Aggregate per-stimulus:
       mean_E         — average of the posterior means
       mean_error     — mean_E − true value  (decoder bias)
       var_E          — variance of posterior means across simulations
                        (≈ inverse of empirical Fisher information)
       mean_abs_error — average |posterior mean − true value|

Two TSVs per subject (one per session), at
  derivatives/encoding_models/aprf-session-shift/sub-<S>/ses-<i>/func/
    sub-<S>_ses-<i>_task-abstractvalue_mask-<roi>_nvoxels-<n>
    _nsims-<n>_desc-expected_decoded_pe.tsv

Usage
-----
  python compute_expected_decoded_value_aprf.py 03 --n-simulations 1000
  python compute_expected_decoded_value_aprf.py 04 --roi NPCr --n-voxels 100
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
from braincoder.utils.math import get_expected_value

from abstract_values.utils.data import Subject, BIDS_FOLDER


def get_value_paradigm(sub, sessions):
    """Per-trial DataFrame with column 'x' = objective CHF value."""
    rows = []
    for session in sessions:
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values("onset")
            for _, row in run_ev[run_ev["event_type"] == "gabor"].iterrows():
                rows.append(float(row["value"]))
    return pd.DataFrame({"x": np.array(rows, dtype=np.float32)})


def _load_session_shift_params(subject, smoothed, masker, bids_folder):
    """Return (fwhm, amp, baseline, r2, {mode_i: arr}) — all (n_vox,) arrays."""
    ss_dir = (Path(bids_folder) / "derivatives" / "encoding_models"
              / "aprf-session-shift" / f"sub-{subject}" / "func")
    smooth = "_smoothed" if smoothed else ""

    def load(desc):
        fn = (ss_dir / f"sub-{subject}_task-abstractvalue_space-T1w"
                       f"_desc-{desc}{smooth}_pe.nii.gz")
        if not fn.exists():
            raise FileNotFoundError(f"No session-shift param: {fn}")
        return nli.load_img(str(fn))

    fwhm = masker.transform(load("fwhm")).squeeze().astype(np.float32)
    amp  = masker.transform(load("amplitude")).squeeze().astype(np.float32)
    base = masker.transform(load("baseline")).squeeze().astype(np.float32)
    r2   = pd.Series(masker.transform(load("r2")).squeeze().astype(np.float32))

    modes: dict[str, np.ndarray] = {}
    for desc in ("mode_1", "mode_2"):
        try:
            modes[desc] = masker.transform(load(desc)).squeeze().astype(np.float32)
        except FileNotFoundError:
            pass
    return fwhm, amp, base, r2, modes


def _simulate_and_decode(model, pars_df, omega, dof,
                          stimulus_grid: np.ndarray,
                          n_simulations: int, batch_stimuli: int = 25):
    """Simulate noisy responses + decode them per stimulus.

    Returns a DataFrame with columns: value, sim_idx, E (posterior mean).

    Memory note: we simulate one batch of stimuli at a time
    (``batch_stimuli`` stimuli × ``n_simulations`` repeats); the full
    pdf cube would otherwise blow up at 200 stimuli × 1000 sims × 200
    grid × 100 voxels.
    """
    stim_df_full = pd.DataFrame({"x": stimulus_grid.astype(np.float32)})
    stim_df_full.index.name = "stimulus"
    n_total = len(stimulus_grid)
    rows = []
    for start in range(0, n_total, batch_stimuli):
        stop = min(start + batch_stimuli, n_total)
        stim_batch = stim_df_full.iloc[start:stop].copy()
        # simulate(stim, pars, noise, n_repeats) returns DataFrame of shape
        # (len(stim) * n_repeats, n_voxels). The MultiIndex carries
        # (stimulus, repeat).
        sim_data = model.simulate(stim_batch, pars_df, noise=omega, dof=dof,
                                  n_repeats=n_simulations)
        pdf = model.get_stimulus_pdf(sim_data, parameters=pars_df,
                                     omega=omega, dof=dof,
                                     stimulus_range=stim_df_full,
                                     normalize=False)
        # pdf columns include the full value grid; drop the extra index level
        # carrying the per-batch stimulus.
        if pdf.columns.nlevels > 1:
            pdf = pdf.droplevel(1, axis=1)
        # Expected value per simulated trial
        E = get_expected_value(pdf, normalize=True)
        # Rebuild: each row of sim_data is (stim_idx_in_batch, repeat) → recover true value
        idx = sim_data.index
        # sim_data.index is a MultiIndex; figure out which level is which
        # Use first level as stimulus index, second as repeat
        levels = idx.names
        # 'stimulus' should be one of the level names from stim_batch
        stim_lvl = next((n for n in levels if n in ("stimulus", "value")), levels[0])
        rep_lvl = next((n for n in levels if n not in (stim_lvl,)), levels[-1])
        true_vals = stim_batch["x"].reindex(idx.get_level_values(stim_lvl)).values
        for tv, e_val in zip(true_vals, np.asarray(E)):
            rows.append({"value": float(tv), "E": float(e_val)})
        print(f"    [stim {start}:{stop}/{n_total}] simulated+decoded "
              f"{(stop - start) * n_simulations} trials")
    return pd.DataFrame(rows)


def main(subject, sessions=None, roi="NPCr", hemi="None",
         n_voxels=100, n_simulations=1000, n_values=200,
         n_noise_iterations=1000, batch_stimuli=25,
         value_min=0.5, value_max=50.0,
         bids_folder=BIDS_FOLDER, fmriprep_deriv="fmriprep",
         smoothed=False):

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)
    if sessions is None:
        sessions = sub.get_sessions()
    sessions = sorted(sessions)
    smooth_label = "_smoothed" if smoothed else ""
    hemi_arg = None if hemi == "None" else hemi
    mask_desc = f"{roi}{'_hemi-' + hemi if hemi_arg else ''}"

    print(f"sub-{subject}  sessions={sessions}  "
          f"[aPRF session-shift expected-decoded-value simulation]")

    # ROI mask + masker (in BOLD grid)
    ref_betas = sub.get_single_trial_estimates(sessions, desc="gabor",
                                               smoothed=smoothed)
    mask_img = sub.get_roi_mask(roi=roi, hemi=hemi_arg)
    masker = NiftiMasker(mask_img=mask_img,
                         target_affine=ref_betas.affine,
                         target_shape=ref_betas.shape[:3]).fit()

    fwhm_arr, amp_arr, base_arr, r2, modes = _load_session_shift_params(
        subject, smoothed, masker, bids_folder)

    valid = (fwhm_arr > 0) & (amp_arr != 0)
    for m in modes.values():
        valid &= m > 0
    r2_valid = r2[valid]

    if n_voxels == 0:
        sel = r2_valid[r2_valid > 0].index
    else:
        sel = r2_valid.sort_values(ascending=False).index[:n_voxels]
    print(f"  {len(sel)} voxels selected  (R² ≥ {float(r2.loc[sel].min()):.3f})")

    stimulus_grid = np.linspace(value_min, value_max, n_values, dtype=np.float32)

    for ses_i in sessions:
        mode_desc = f"mode_{ses_i}"
        if mode_desc not in modes:
            print(f"  skip ses-{ses_i}: no {mode_desc} param")
            continue
        cond = sub.get_mapping(ses_i)
        print(f"\n  --- session {ses_i} ({mode_desc}, condition={cond}) ---")

        pars_df = pd.DataFrame({
            "mode":      modes[mode_desc][sel.values],
            "fwhm":      fwhm_arr[sel.values],
            "amplitude": amp_arr[sel.values],
            "baseline":  base_arr[sel.values],
        })

        ses_paradigm = get_value_paradigm(sub, [ses_i])
        ses_betas = sub.get_single_trial_estimates([ses_i], desc="gabor",
                                                   smoothed=smoothed)
        ses_data = pd.DataFrame(masker.transform(ses_betas).astype(np.float32))
        ses_data = ses_data.iloc[:, sel.values]
        print(f"  {len(ses_paradigm)} gabor trials  "
              f"value range {float(ses_paradigm['x'].min()):.1f}–"
              f"{float(ses_paradigm['x'].max()):.1f} CHF")

        model = LogGaussianPRF(allow_neg_amplitudes=True,
                               parameterisation="mode_fwhm_natural")
        model.init_pseudoWWT(stimulus_grid, pars_df)

        print(f"  fitting noise model ({n_noise_iterations} iter)...")
        rfit = ResidualFitter(model, ses_data, ses_paradigm, pars_df)
        omega, dof = rfit.fit(init_sigma2=1e-2, init_dof=10.0,
                              learning_rate=0.05,
                              max_n_iterations=n_noise_iterations)
        dof_str = f"{float(dof):.1f}" if dof is not None else "None (Gaussian)"
        print(f"  noise model: dof={dof_str}")

        print(f"  simulating {n_simulations} repeats × {n_values} stimuli "
              f"({n_simulations * n_values} trials)...")
        E_long = _simulate_and_decode(model, pars_df, omega, dof,
                                       stimulus_grid, n_simulations,
                                       batch_stimuli=batch_stimuli)

        agg = (E_long.assign(error=E_long["E"] - E_long["value"],
                              abs_error=(E_long["E"] - E_long["value"]).abs())
               .groupby("value").agg(mean_E=("E", "mean"),
                                      var_E=("E", "var"),
                                      mean_error=("error", "mean"),
                                      mean_abs_error=("abs_error", "mean"),
                                      n_sims=("E", "count"))
               .reset_index())

        out_dir = (bids_folder / "derivatives" / "encoding_models"
                   / "aprf-session-shift" / f"sub-{subject}"
                   / f"ses-{ses_i}" / "func")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fn = (out_dir /
                  f"sub-{subject}_ses-{ses_i}_task-abstractvalue"
                  f"_mask-{mask_desc}_nvoxels-{n_voxels}_nsims-{n_simulations}"
                  f"{smooth_label}_desc-expected_decoded_pe.tsv")
        agg.to_csv(out_fn, sep="\t", index=False)
        print(f"  saved {out_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subject", help="Subject label without 'sub-'")
    parser.add_argument("--sessions", type=int, nargs="+", default=None)
    parser.add_argument("--roi", default="NPCr")
    parser.add_argument("--hemi", default="None")
    parser.add_argument("--n-voxels", type=int, default=100)
    parser.add_argument("--n-simulations", type=int, default=1000)
    parser.add_argument("--n-values", type=int, default=200)
    parser.add_argument("--value-min", type=float, default=0.5)
    parser.add_argument("--value-max", type=float, default=50.0)
    parser.add_argument("--n-noise-iterations", type=int, default=1000)
    parser.add_argument("--batch-stimuli", type=int, default=25)
    parser.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    parser.add_argument("--fmriprep-deriv", default="fmriprep",
                        choices=["fmriprep", "fmriprep-t2w", "fmriprep-flair"])
    parser.add_argument("--smoothed", action="store_true")
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions,
         roi=args.roi, hemi=args.hemi,
         n_voxels=args.n_voxels, n_simulations=args.n_simulations,
         n_values=args.n_values, value_min=args.value_min,
         value_max=args.value_max,
         n_noise_iterations=args.n_noise_iterations,
         batch_stimuli=args.batch_stimuli,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed)
