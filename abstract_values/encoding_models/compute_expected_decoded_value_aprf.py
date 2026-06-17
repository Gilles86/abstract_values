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
from scipy.stats import gaussian_kde
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


def _load_session_shift_params(subject, smoothed, masker, bids_folder,
                                  model_dir="aprf-session-shift"):
    """Return (fwhm, amp, baseline, r2, {mode_i: arr}) — all (n_vox,) arrays.

    ``model_dir`` selects which session-shift fit to load:
      - ``aprf-session-shift``       — log-Gaussian PRFs (lognormal model)
      - ``aprf-gauss-session-shift`` — symmetric Gaussian PRFs
    Both have the same 5-parameter (mode_1, mode_2, fwhm, amp, baseline)
    interface, so the loader is identical.
    """
    ss_dir = (Path(bids_folder) / "derivatives" / "encoding_models"
              / model_dir / f"sub-{subject}" / "func")
    smooth = "_smoothed" if smoothed else ""

    def load(desc):
        fn = (ss_dir / f"sub-{subject}_task-abstractvalue_space-T1w"
                       f"_desc-{desc}{smooth}_pe.nii.gz")
        if not fn.exists():
            raise FileNotFoundError(f"No session-shift param: {fn}")
        return nli.load_img(str(fn))

    def load_opt(desc):                            # shared param, may be absent
        try:
            return masker.transform(load(desc)).squeeze().astype(np.float32)
        except FileNotFoundError:
            return None
    fwhm = load_opt("fwhm")                        # absent for fwhm-/fully-shift
    amp  = load_opt("amplitude")                   # absent for fully-shift
    base = load_opt("baseline")                    # absent for fully-shift
    r2   = pd.Series(masker.transform(load("r2")).squeeze().astype(np.float32))

    modes: dict[str, np.ndarray] = {}
    fwhms: dict[str, np.ndarray] = {}             # per-session (fwhm-/fully-shift)
    amps: dict[str, np.ndarray] = {}              # per-session (fully-shift)
    bases: dict[str, np.ndarray] = {}
    for pref, store in (("mode", modes), ("fwhm", fwhms),
                        ("amplitude", amps), ("baseline", bases)):
        for i in (1, 2):
            try:
                store[f"{pref}_{i}"] = masker.transform(
                    load(f"{pref}_{i}")).squeeze().astype(np.float32)
            except FileNotFoundError:
                pass
    return fwhm, amp, base, r2, modes, fwhms, amps, bases


def _simulate_and_decode_legacy(model, pars_df, omega, dof,
                                 stimulus_grid: np.ndarray,
                                 n_simulations: int, batch_stimuli: int = 25):
    """Legacy manual loop, kept for reference. Prefer
    ``EncodingModel.get_expected_uncertainty`` (added in braincoder
    0.5.2+) — same aggregation in one call."""
    stim_df_full = pd.DataFrame({"x": stimulus_grid.astype(np.float32)})
    stim_df_full.index.name = "stimulus"
    n_total = len(stimulus_grid)
    rows = []
    for start in range(0, n_total, batch_stimuli):
        stop = min(start + batch_stimuli, n_total)
        stim_batch = stim_df_full.iloc[start:stop].copy()
        sim_data = model.simulate(stim_batch, pars_df, noise=omega, dof=dof,
                                  n_repeats=n_simulations)
        pdf = model.get_stimulus_pdf(sim_data, parameters=pars_df,
                                     omega=omega, dof=dof,
                                     stimulus_range=stim_df_full,
                                     normalize=False)
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


def _objective_value_prior(presented_values, grid, bw=None):
    """Empirical objective-value prior on ``grid`` — a KDE of the CHF
    values actually presented this session, normalised to integrate to 1
    over the grid. CDF mapping gives a ~flat density; InvCDF mapping a
    peaked one — this is the per-condition prior the user asked to inject
    (vs. the flat/uniform-over-grid default)."""
    kde = gaussian_kde(np.asarray(presented_values, dtype=np.float64),
                       bw_method=bw)
    p = kde(grid)
    return p / np.trapz(p, grid)


def _decode_with_prior(model, pars_df, omega, dof, true_values,
                       decode_grid, prior, n_simulations, batch_stimuli=25):
    """Simulate at each ``true_values`` stimulus, decode over a fine
    ``decode_grid`` with an arbitrary ``prior`` weighting (posterior ∝
    likelihood × prior), aggregate per true value.

    Returns a DataFrame matching ``get_expected_uncertainty``'s schema
    (value, mean_E, var_E, mean_error, mean_abs_error, n_sims) so the
    downstream collectors/plotters need no change.
    """
    decode_df = pd.DataFrame({"x": np.asarray(decode_grid, dtype=np.float32)})
    decode_df.index.name = "stimulus"
    prior = np.asarray(prior, dtype=np.float64)
    true_values = np.asarray(true_values, dtype=np.float32)
    n_total = len(true_values)
    rows = []
    for start in range(0, n_total, batch_stimuli):
        stop = min(start + batch_stimuli, n_total)
        stim_batch = pd.DataFrame({"x": true_values[start:stop]})
        stim_batch.index.name = "stimulus"
        sim_data = model.simulate(stim_batch, pars_df, noise=omega, dof=dof,
                                  n_repeats=n_simulations)
        pdf = model.get_stimulus_pdf(sim_data, parameters=pars_df,
                                     omega=omega, dof=dof,
                                     stimulus_range=decode_df, normalize=False)
        if pdf.columns.nlevels > 1:
            pdf = pdf.droplevel(1, axis=1)
        # posterior ∝ likelihood × prior  (columns align with decode_grid)
        pdf = pd.DataFrame(pdf.values * prior[np.newaxis, :],
                           index=pdf.index, columns=pdf.columns)
        E = np.asarray(get_expected_value(pdf, normalize=True))
        idx = sim_data.index
        levels = idx.names
        stim_lvl = next((n for n in levels if n in ("stimulus", "value")), levels[0])
        true_arr = stim_batch["x"].reindex(idx.get_level_values(stim_lvl)).values
        rows.extend(zip([float(t) for t in true_arr], [float(e) for e in E]))
        print(f"    [stim {start}:{stop}/{n_total}] simulated+decoded "
              f"{(stop - start) * n_simulations} trials")
    d = pd.DataFrame(rows, columns=["value", "E"])
    d["abs_err"] = (d["E"] - d["value"]).abs()
    agg = d.groupby("value").agg(mean_E=("E", "mean"), var_E=("E", "var"),
                                 mean_abs_error=("abs_err", "mean"),
                                 n_sims=("E", "size"))
    agg["mean_error"] = agg["mean_E"] - agg.index.to_numpy()
    return agg.reset_index()[["value", "mean_E", "var_E", "mean_error",
                              "mean_abs_error", "n_sims"]]


def main(subject, sessions=None, roi="NPCr", hemi="None",
         n_voxels=100, n_simulations=1000, n_values=200,
         n_noise_iterations=1000, batch_stimuli=25,
         value_min=0.5, value_max=50.0,
         match_trained=True,
         fdr_alpha=None, p_signal_thr=None, fdr_fallback_n_voxels=100,
         bids_folder=BIDS_FOLDER, fmriprep_deriv="fmriprep",
         smoothed=False, spherical_noise=True, model_type="lognormal",
         prior="none", prior_bw=None):
    """If ``fdr_alpha`` is set, voxels are selected by FDR-thresholding the
    aprf-weighted whole-brain R² mixture instead of the per-subject top-N.
    ``fdr_fallback_n_voxels`` is the top-N fallback when the mixture is
    flagged degenerate (sub-06 has been one such case). ``p_signal_thr``
    is an alternative selection on P(signal | r²). Mutually exclusive."""
    assert not (fdr_alpha is not None and p_signal_thr is not None), \
        "Pass at most one of --fdr-alpha / --p-signal-thr"

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

    # Choose which session-shift fit directory to read (and the matching
    # PRF class to instantiate below). lognormal → aprf-session-shift +
    # SessionShiftedLogGaussianPRF; gaussian → aprf-gauss-session-shift +
    # SessionShiftedGaussianValuePRF. Both share the 5-param interface.
    ss_subdir = {"lognormal":     "aprf-session-shift",
                  "gaussian":      "aprf-gauss-session-shift",
                  "fwhm-shift":    "aprf-fwhm-shift",
                  "fully-shifted": "aprf-fully-shifted"}[model_type]
    out_subdir = ss_subdir
    (fwhm_arr, amp_arr, base_arr, r2, modes, fwhms, amps, bases
     ) = _load_session_shift_params(
        subject, smoothed, masker, bids_folder, model_dir=ss_subdir)

    valid = np.ones(len(r2), dtype=bool)
    for m in modes.values():
        valid &= m > 0
    if fwhms:                                      # fwhm-/fully-shift: per-session
        for f in fwhms.values():
            valid &= f > 0
    elif fwhm_arr is not None:                      # session-shift: shared fwhm
        valid &= fwhm_arr > 0
    if amps:                                        # fully-shift: per-session amp
        for a in amps.values():
            valid &= a != 0
    elif amp_arr is not None:
        valid &= amp_arr != 0
    r2_valid = r2[valid]

    # Voxel selection: top-N by joint R², or FDR/p_signal against the
    # aprf-weighted whole-brain mixture (same cutoff logic as decode_value
    # / decode_gabor). Fallback to top-N when the mixture is degenerate.
    if fdr_alpha is not None or p_signal_thr is not None:
        if fdr_alpha is not None:
            from abstract_values.encoding_models.compute_r2_mixture \
                import get_brain_fdr_threshold
            res = get_brain_fdr_threshold(
                subject, model="aprf-weighted", bids_folder=bids_folder,
                alpha=fdr_alpha, smoothed=smoothed)
            crit_label = f"FDR≤{fdr_alpha:.2f}"
            sel_tag = f"nvoxels-fdr{int(round(fdr_alpha * 100)):02d}"
        else:
            from abstract_values.encoding_models.compute_r2_mixture \
                import get_brain_p_signal_threshold
            res = get_brain_p_signal_threshold(
                subject, model="aprf-weighted", bids_folder=bids_folder,
                p=p_signal_thr, smoothed=smoothed)
            crit_label = f"P(signal)≥{p_signal_thr:.2f}"
            sel_tag = f"nvoxels-psig{int(round(p_signal_thr * 100)):02d}"
        if res is None:
            raise RuntimeError("aprf-weighted whole-brain mixture missing "
                                 "and auto-fit failed. Run "
                                 "compute_r2_mixture --models aprf-weighted first.")
        thr = res["threshold"]
        if res["degenerate"] or not np.isfinite(thr):
            sel = r2_valid.sort_values(ascending=False).index[:fdr_fallback_n_voxels]
            print(f"  {len(sel)} voxels selected  (mixture degenerate ⇒ "
                  f"fallback to top-{fdr_fallback_n_voxels} by R²)")
        else:
            sel = r2_valid[r2_valid > thr].index
            if len(sel) < 10:
                sel = r2_valid.sort_values(ascending=False).index[:fdr_fallback_n_voxels]
                print(f"  {len(sel)} voxels selected  "
                      f"(only {(r2_valid > thr).sum()} passed {crit_label} → "
                      f"R² > {thr:.3f}; fallback to top-{fdr_fallback_n_voxels})")
            else:
                print(f"  {len(sel)} voxels selected  ({crit_label} → "
                      f"R² > {thr:.3f})")
    elif n_voxels == 0:
        sel = r2_valid[r2_valid > 0].index
        sel_tag = "nvoxels-0"
        print(f"  {len(sel)} voxels selected (all r²>0)")
    else:
        sel = r2_valid.sort_values(ascending=False).index[:n_voxels]
        sel_tag = f"nvoxels-{n_voxels}"
        print(f"  {len(sel)} voxels selected  (R² ≥ {float(r2.loc[sel].min()):.3f})")

    # Stimulus / simulation grid construction:
    #   match_trained=True (default): the unique CHF values actually
    #       presented in *this session* (per-session, since CDF and
    #       InvCDF have different 23-value grids). Bounded uniform prior
    #       on those discrete points. Matches the V1 trained-grid
    #       convention — see notes/v1_decoder_edge_effect/.
    #   match_trained=False: the legacy continuous grid
    #       np.linspace(value_min, value_max, n_values) — wide bounds
    #       (e.g. [0.5, 50] CHF) that include CHF values never presented
    #       during training, which inflates SD by extrapolation.
    if not match_trained:
        stimulus_grid = np.linspace(value_min, value_max, n_values,
                                     dtype=np.float32)
        print(f"  simulation grid: continuous [{value_min}, {value_max}] CHF "
              f"({n_values} pts)")

    for ses_i in sessions:
        mode_desc = f"mode_{ses_i}"
        if mode_desc not in modes:
            print(f"  skip ses-{ses_i}: no {mode_desc} param")
            continue
        cond = sub.get_mapping(ses_i)
        print(f"\n  --- session {ses_i} ({mode_desc}, condition={cond}) ---")

        # Full-voxel param frame; mask gets applied to the model below so
        # init_pseudoWWT + ResidualFitter see consistent shapes (same pattern
        # as compute_fisher_information_aprf.py).
        # per-session params where the model provides them; else shared
        pars_full = pd.DataFrame({
            "mode":      modes[mode_desc],
            "fwhm":      fwhms.get(f"fwhm_{ses_i}", fwhm_arr),
            "amplitude": amps.get(f"amplitude_{ses_i}", amp_arr),
            "baseline":  bases.get(f"baseline_{ses_i}", base_arr),
        })

        ses_paradigm = get_value_paradigm(sub, [ses_i])
        if match_trained:
            stimulus_grid = np.array(
                sorted(ses_paradigm["x"].unique()), dtype=np.float32)
            print(f"  simulation grid: {len(stimulus_grid)} trained CHF "
                  f"values ({stimulus_grid.min():.2f}–{stimulus_grid.max():.2f})")
        ses_betas = sub.get_single_trial_estimates([ses_i], desc="gabor",
                                                   smoothed=smoothed)
        ses_data = pd.DataFrame(masker.transform(ses_betas).astype(np.float32))
        print(f"  {len(ses_paradigm)} gabor trials  "
              f"value range {float(ses_paradigm['x'].min()):.1f}–"
              f"{float(ses_paradigm['x'].max()):.1f} CHF")

        if model_type in ("lognormal", "fwhm-shift", "fully-shifted"):
            model = LogGaussianPRF(allow_neg_amplitudes=True,
                                    parameterisation="mode_fwhm_natural")
        else:
            from abstract_values.encoding_models.models import GaussianValuePRF
            model = GaussianValuePRF(allow_neg_amplitudes=True)
        model.parameters = pars_full
        model.apply_mask(sel)
        # pseudoWWT must reflect the data being fit (the paradigm), not the
        # simulation grid — otherwise ResidualFitter shape-mismatches at fit().
        model.init_pseudoWWT(ses_paradigm["x"].values, model.parameters)

        # Use selected voxels' data
        ses_data_sel = ses_data[sel]
        pars_df = model.parameters  # the post-mask DataFrame

        print(f"  fitting noise model ({n_noise_iterations} iter)...")
        rfit = ResidualFitter(model, ses_data_sel, ses_paradigm)
        omega, dof = rfit.fit(init_sigma2=1e-2, init_dof=10.0,
                              learning_rate=0.05,
                              max_n_iterations=n_noise_iterations,
                              spherical=spherical_noise)
        dof_str = f"{float(dof):.1f}" if dof is not None else "None (Gaussian)"
        print(f"  noise model: dof={dof_str}")

        n_sim_stim = len(stimulus_grid)
        print(f"  simulating {n_simulations} repeats × {n_sim_stim} stimuli "
              f"({n_simulations * n_sim_stim} trials)  prior={prior}…")
        if prior == "none":
            # Flat prior over the (discrete) simulation grid itself.
            # Single call: braincoder.get_expected_uncertainty does the
            # simulate → decode → aggregate cycle and returns the
            # per-stimulus DataFrame directly.
            agg = model.get_expected_uncertainty(
                stimulus_grid, omega=omega, dof=dof,
                parameters=pars_df, n_simulations=n_simulations,
                batch_stimuli=batch_stimuli, progress=True,
            ).reset_index().rename(columns={"value": "value"})
        else:
            # Decode over a fine continuous grid spanning the trained range,
            # weighting the posterior by a prior. prior="flat" → uniform
            # density (control, isolates grid resolution); prior="objective"
            # → empirical per-condition objective-value density (KDE). Both
            # simulate true values at the trained grid so the per-value EU is
            # directly comparable to the flat-prior (prior="none") run.
            decode_grid = np.linspace(float(stimulus_grid.min()),
                                      float(stimulus_grid.max()),
                                      n_values, dtype=np.float32)
            if prior == "objective":
                prior_w = _objective_value_prior(
                    ses_paradigm["x"].values, decode_grid, bw=prior_bw)
            else:                                   # flat
                prior_w = np.ones_like(decode_grid, dtype=np.float64)
            print(f"  decode grid: continuous [{decode_grid.min():.2f}, "
                  f"{decode_grid.max():.2f}] CHF ({n_values} pts), "
                  f"prior='{prior}'")
            agg = _decode_with_prior(
                model, pars_df, omega, dof, stimulus_grid,
                decode_grid, prior_w, n_simulations,
                batch_stimuli=batch_stimuli)

        out_dir = (bids_folder / "derivatives" / "encoding_models"
                   / out_subdir / f"sub-{subject}"
                   / f"ses-{ses_i}" / "func")
        out_dir.mkdir(parents=True, exist_ok=True)
        # Backward-compatible naming: full noise keeps the original (untagged)
        # filename; spherical adds a _noise-spherical tag so both can coexist.
        noise_tag = "_noise-spherical" if spherical_noise else ""
        prior_tag = "" if prior == "none" else f"_prior-{prior}"
        out_fn = (out_dir /
                  f"sub-{subject}_ses-{ses_i}_task-abstractvalue"
                  f"_mask-{mask_desc}_{sel_tag}_nsims-{n_simulations}"
                  f"{noise_tag}{prior_tag}{smooth_label}_desc-expected_decoded_pe.tsv")
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
    parser.add_argument("--fdr-alpha", type=float, default=None,
                        help="FDR α on the aprf-weighted whole-brain mixture "
                             "(mutually exclusive with --p-signal-thr)")
    parser.add_argument("--p-signal-thr", type=float, default=None,
                        help="P(signal|r²) threshold on the same mixture "
                             "(mutually exclusive with --fdr-alpha)")
    parser.add_argument("--fdr-fallback-n-voxels", type=int, default=100,
                        help="Top-N fallback when the mixture is flagged degenerate")
    parser.add_argument("--n-simulations", type=int, default=1000)
    parser.add_argument("--n-values", type=int, default=200,
                        help="Continuous-grid size; only used when "
                             "--full-grid is passed.")
    parser.add_argument("--full-grid", action="store_true",
                        help="Use the continuous [--value-min, --value-max] "
                             "grid (default: use the actually-presented 23 "
                             "CHF values per session as the bounded prior).")
    parser.add_argument("--value-min", type=float, default=0.5)
    parser.add_argument("--value-max", type=float, default=50.0)
    parser.add_argument("--n-noise-iterations", type=int, default=1000)
    parser.add_argument("--batch-stimuli", type=int, default=25)
    parser.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    parser.add_argument("--fmriprep-deriv", default="fmriprep",
                        choices=["fmriprep", "fmriprep-t2w", "fmriprep-flair"])
    parser.add_argument("--smoothed", action="store_true")
    parser.add_argument("--model", default="lognormal",
                         choices=["lognormal", "gaussian", "fwhm-shift", "fully-shifted"],
                         help="Encoding-model PRF shape. lognormal "
                              "(default) loads aprf-session-shift fits; "
                              "gaussian loads aprf-gauss-session-shift "
                              "fits (symmetric Gaussian on CHF). Both have "
                              "5 params per voxel; output dir mirrors the "
                              "input model dir.")
    # Default noise model is spherical (iid Gaussian residuals on the
    # selected voxels). Empirically gives cleaner per-stimulus SD curves
    # and avoids ill-conditioned covariance estimates on small voxel
    # subsets. Pass --no-spherical-noise to go back to full residual Omega.
    sph_grp = parser.add_mutually_exclusive_group()
    sph_grp.add_argument("--spherical-noise",    dest="spherical_noise",
                          action="store_true", default=True,
                          help="(default) iid Gaussian residual noise. "
                               "Output: _noise-spherical filename tag.")
    sph_grp.add_argument("--no-spherical-noise", dest="spherical_noise",
                          action="store_false",
                          help="Fit full residual covariance instead of "
                               "spherical. Output: no _noise-* tag.")
    parser.add_argument("--prior", default="none",
                        choices=["none", "flat", "objective"],
                        help="Decoding prior. 'none' (default): flat prior "
                             "over the discrete trained grid (current "
                             "behaviour, get_expected_uncertainty). 'flat': "
                             "uniform density over a fine continuous grid "
                             "(control). 'objective': empirical per-condition "
                             "objective-value density (KDE of presented CHF "
                             "values). flat/objective add a _prior-<p> tag.")
    parser.add_argument("--prior-bw", type=float, default=None,
                        help="KDE bandwidth (scipy bw_method) for the "
                             "objective prior; default Scott's rule.")
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions,
         roi=args.roi, hemi=args.hemi,
         n_voxels=args.n_voxels, n_simulations=args.n_simulations,
         n_values=args.n_values, value_min=args.value_min,
         value_max=args.value_max,
         match_trained=not args.full_grid,
         fdr_alpha=args.fdr_alpha, p_signal_thr=args.p_signal_thr,
         fdr_fallback_n_voxels=args.fdr_fallback_n_voxels,
         n_noise_iterations=args.n_noise_iterations,
         batch_stimuli=args.batch_stimuli,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed, spherical_noise=args.spherical_noise,
         model_type=args.model, prior=args.prior, prior_bw=args.prior_bw)
