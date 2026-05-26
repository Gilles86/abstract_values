#!/usr/bin/env python3
"""
V1 expected decoded orientation from the AxialVonMisesPRF model — per session.

Mirror of ``compute_expected_decoded_value_aprf.py``, but on the orientation
axis with π-periodic circular statistics. Per-session noise model only —
the basis weights are fit jointly across sessions (V1 is expected to be
condition-invariant in orientation tuning; that's the test).

For each session:
  1. Fit the joint vonmises basis weights (closed-form, same as fit_vonmises_model).
  2. Select top-N voxels by joint R².
  3. Re-fit the Student-t residual noise model on THIS session's betas.
  4. For each orientation in a fine grid, simulate ``n_simulations`` noisy
     response vectors, decode each via ``get_stimulus_pdf``, and compute
     the posterior **circular mean** (Jammalamadaka–Sarma, period π).
  5. Aggregate per true orientation:
       mean_E         — circular mean of posterior circular means
       sd_E           — circular SD across simulations
       mean_error     — mean signed circular distance (decoded − true)
       mean_abs_error — mean absolute circular distance

Output (one TSV per session) at
  derivatives/encoding_models/vonmises/sub-<S>/ses-<i>/func/
    sub-<S>_ses-<i>_task-abstractvalue_mask-<mask>
    _nvoxels-<n>_nsims-<n>[_smoothed]_desc-expected_decoded_orientation_pe.tsv

Usage
-----
  python compute_expected_decoded_orientation_vonmises.py 03 --n-simulations 1000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import AxialVonMisesPRF
from braincoder.optimize import ResidualFitter, WeightFitter
from braincoder.utils import get_rsq

from abstract_values.utils.data import Subject, BIDS_FOLDER


def get_gabor_paradigm(sub, sessions):
    """Per-trial DataFrame with column 'x' = gabor orientation in radians."""
    rows = []
    for session in sessions:
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values("onset")
            for _, row in run_ev[run_ev["event_type"] == "gabor"].iterrows():
                rows.append(float(np.deg2rad(float(row["orientation"]))))
    return pd.DataFrame({"x": np.array(rows, dtype=np.float32)})


def make_basis_parameters(n_basis=8, kappa=2.0):
    mus = np.linspace(0, np.pi, n_basis, endpoint=False).astype(np.float32)
    return pd.DataFrame({
        "mu":        mus,
        "kappa":     np.full(n_basis, kappa, dtype=np.float32),
        "amplitude": np.ones(n_basis,  dtype=np.float32),
        "baseline":  np.zeros(n_basis, dtype=np.float32),
    })


# ── circular statistics (period π) ──────────────────────────────────────────
#
# Prior choice — explicit: the simulation/posterior grid is
#   np.linspace(0, np.pi, n_orientations, endpoint=False)
# i.e. ONE full period uniformly sampled. The inter-grid spacing equals
# the wrap-around gap, so the implicit prior is UNIFORM on the π-periodic
# axis (the right prior for axial orientation). Posterior aggregation
# uses the doubled-angle Jammalamadaka–Sarma trick (SCALE = 2 below) so
# wrap is preserved end-to-end.
#
# Observation — elevated SD at 0°/90°/180°: empirical sanity check
# (sd_circ vs √(π/2) × empirical MAE across orientations) shows the two
# agree within ~10% with the SAME orientation pattern, so this is NOT a
# circular-statistics artifact. It's a real property of the decoder.
# Likely mechanism: the 8 fixed vonmises bases sit at 0° and 22.5° /
# 157.5°, so at true θ=0° the two flanking bases contribute symmetric
# off-axis energy that broadens the local posterior. The pattern shows
# up identically in both the circular SD and the empirical mean-|err|,
# confirming the doubled-angle Jammalamadaka–Sarma maths is sound.

PERIOD = np.pi
SCALE = 2 * np.pi / PERIOD          # 2 — double angles before averaging


def posterior_circular_mean(pdf_rows: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Circular mean of each row of ``pdf_rows`` over the orientation grid.

    ``pdf_rows`` shape ``(n_trials, n_orientations)``; ``grid`` is the
    orientation grid in radians. Posteriors are renormalised to sum to 1.
    Returns the decoded orientation in radians in ``[0, π)``.
    """
    pdf_rows = pdf_rows / pdf_rows.sum(axis=1, keepdims=True)
    s = pdf_rows @ np.sin(SCALE * grid)
    c = pdf_rows @ np.cos(SCALE * grid)
    return (np.arctan2(s, c) / SCALE) % PERIOD


def circular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed minimal distance from ``b`` to ``a`` on a π-periodic axis,
    in ``(-π/2, π/2]``."""
    d = (a - b) % PERIOD
    return np.where(d > PERIOD / 2, d - PERIOD, d)


def aggregate_per_stimulus(true_arr: np.ndarray,
                           decoded_arr: np.ndarray,
                           stim_grid: np.ndarray) -> pd.DataFrame:
    """Per true-orientation circular aggregation across simulations."""
    rows = []
    # Decoded errors, with circular wrap.
    errors = circular_distance(decoded_arr, true_arr)
    for theta in stim_grid:
        mask = true_arr == theta
        if not mask.any():
            continue
        dec = decoded_arr[mask]
        err = errors[mask]
        # Circular mean of the decoded values.
        s = np.sin(SCALE * dec).mean()
        c = np.cos(SCALE * dec).mean()
        R = np.sqrt(s * s + c * c)
        # Circular mean
        mean_E = (np.arctan2(s, c) / SCALE) % PERIOD
        # Circular SD — scaled to the π-period domain (so SDs read as radians
        # of orientation, not radians of doubled angle).
        sd_E = (np.sqrt(-2.0 * np.log(np.clip(R, 1e-12, 1.0))) / SCALE)
        rows.append({
            "value":          float(theta),                    # generic 'value' column
            "value_deg":      float(np.rad2deg(theta)),
            "mean_E":         float(mean_E),
            "var_E":          float(sd_E ** 2),
            "sd_E":           float(sd_E),
            "mean_error":     float(np.mean(err)),
            "mean_abs_error": float(np.mean(np.abs(err))),
            "n_sims":         int(mask.sum()),
        })
    return pd.DataFrame(rows)


def simulate_decode_session(model, basis_pars, weights_sel, omega, dof,
                             stim_grid: np.ndarray, n_simulations: int,
                             batch_stimuli: int = 25) -> pd.DataFrame:
    """Simulate noisy responses + decode per stimulus, returning a per-trial
    long DataFrame with columns ``true`` and ``decoded`` (both in radians).

    Stimuli are processed in small batches to bound memory.
    """
    stim_df_full = pd.DataFrame({"x": stim_grid.astype(np.float32)})
    stim_df_full.index.name = "stimulus"
    n_total = len(stim_grid)
    true_all, decoded_all = [], []
    for start in range(0, n_total, batch_stimuli):
        stop = min(start + batch_stimuli, n_total)
        stim_batch = stim_df_full.iloc[start:stop].copy()
        sim_data = model.simulate(stim_batch, parameters=basis_pars,
                                  weights=weights_sel,
                                  noise=omega, dof=dof,
                                  n_repeats=n_simulations)
        pdf = model.get_stimulus_pdf(sim_data, parameters=basis_pars,
                                     weights=weights_sel,
                                     omega=omega, dof=dof,
                                     stimulus_range=stim_grid,
                                     normalize=True)
        # pdf columns are the full orientation grid.
        # Per-trial circular mean:
        dec = posterior_circular_mean(pdf.values, stim_grid)

        # Identify the true stimulus for each simulated trial.
        idx = sim_data.index
        if isinstance(idx, pd.MultiIndex):
            stim_lvl = next(n for n in idx.names
                            if n in ("stimulus", "value"))
            true_pos = idx.get_level_values(stim_lvl).to_numpy(dtype=int)
        else:
            # n_repeats=1 collapses to a single 'stimulus' index
            true_pos = idx.to_numpy(dtype=int)
        # `true_pos` is the row position within stim_batch — translate to value.
        true_vals = stim_batch["x"].iloc[true_pos - true_pos.min()].to_numpy() \
            if true_pos.min() != 0 else stim_batch["x"].iloc[true_pos].to_numpy()
        # Robust fallback when index-to-position mapping isn't trivial: re-key.
        # In recent braincoder, simulate uses the stim_batch's own index, so
        # `true_pos` already lines up with stim_batch.index. Use reindex.
        try:
            true_vals = stim_batch["x"].reindex(idx.get_level_values(stim_lvl)
                                                  if isinstance(idx, pd.MultiIndex)
                                                  else idx).to_numpy()
        except Exception:
            pass
        true_all.append(true_vals)
        decoded_all.append(dec)
        print(f"    [stim {start}:{stop}/{n_total}] "
              f"simulated+decoded {(stop - start) * n_simulations} trials")
    return np.concatenate(true_all), np.concatenate(decoded_all)


def main(subject, sessions=None, roi="BensonV1", hemi="LR",
         n_voxels=100, n_basis=8, kappa=2.0,
         n_orientations=180, n_simulations=1000,
         n_noise_iterations=1000, batch_stimuli=25,
         fdr_alpha=None, p_signal_thr=None, fdr_fallback_n_voxels=100,
         bids_folder=BIDS_FOLDER, fmriprep_deriv="fmriprep",
         smoothed=False):
    """If ``fdr_alpha`` is set, voxels are selected by FDR-thresholding the
    vonmises whole-brain R² mixture instead of top-N by joint R²."""
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
          f"[V1 vonmises expected-decoded-orientation simulation]")

    paradigm = get_gabor_paradigm(sub, sessions)
    print(f"  {len(paradigm)} gabor trials  "
          f"orientation range {np.rad2deg(paradigm['x'].min()):.1f}–"
          f"{np.rad2deg(paradigm['x'].max()):.1f} deg")

    betas_img = sub.get_single_trial_estimates(sessions, desc="gabor",
                                               smoothed=smoothed)
    mask_img = sub.get_roi_mask(roi=roi, hemi=hemi_arg)
    masker = NiftiMasker(mask_img=mask_img,
                         target_affine=betas_img.affine,
                         target_shape=betas_img.shape[:3]).fit()
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32))
    print(f"  {data.shape[1]} voxels in mask ({mask_desc})")

    basis_pars = make_basis_parameters(n_basis=n_basis, kappa=kappa)
    print(f"  {n_basis} Von Mises basis functions, kappa={kappa}")

    # Joint weight fit (V1 expected to be condition-invariant)
    model = AxialVonMisesPRF()
    weights = WeightFitter(model, basis_pars, data, paradigm).fit()

    basis_pred = model.basis_predictions(paradigm, basis_pars)
    pred = pd.DataFrame(basis_pred @ weights.values,
                        index=data.index, columns=data.columns)
    r2 = get_rsq(data, pred)
    # Voxel selection: top-N by joint R², or FDR/p_signal against the
    # *vonmises* whole-brain mixture (same selection logic as the
    # decode_gabor / FI pipelines). Fallback to top-N on degenerate mixture.
    if fdr_alpha is not None or p_signal_thr is not None:
        if fdr_alpha is not None:
            from abstract_values.encoding_models.compute_r2_mixture \
                import get_brain_fdr_threshold
            res = get_brain_fdr_threshold(
                subject, model="vonmises", bids_folder=bids_folder,
                alpha=fdr_alpha, smoothed=smoothed)
            crit_label = f"FDR≤{fdr_alpha:.2f}"
            sel_tag = f"nvoxels-fdr{int(round(fdr_alpha * 100)):02d}"
        else:
            from abstract_values.encoding_models.compute_r2_mixture \
                import get_brain_p_signal_threshold
            res = get_brain_p_signal_threshold(
                subject, model="vonmises", bids_folder=bids_folder,
                p=p_signal_thr, smoothed=smoothed)
            crit_label = f"P(signal)≥{p_signal_thr:.2f}"
            sel_tag = f"nvoxels-psig{int(round(p_signal_thr * 100)):02d}"
        if res is None:
            raise RuntimeError("vonmises whole-brain mixture missing and "
                                 "auto-fit failed. Run compute_r2_mixture "
                                 "--models vonmises first.")
        thr = res["threshold"]
        if res["degenerate"] or not np.isfinite(thr):
            sel = r2.sort_values(ascending=False).index[:fdr_fallback_n_voxels]
            print(f"  {len(sel)} voxels selected  (mixture degenerate ⇒ "
                  f"fallback to top-{fdr_fallback_n_voxels} by R²)")
        else:
            sel = r2[r2 > thr].index
            if len(sel) < 10:
                sel = r2.sort_values(ascending=False).index[:fdr_fallback_n_voxels]
                print(f"  {len(sel)} voxels selected  "
                      f"(only {(r2 > thr).sum()} passed {crit_label} → "
                      f"R² > {thr:.3f}; fallback to top-{fdr_fallback_n_voxels})")
            else:
                print(f"  {len(sel)} voxels selected  ({crit_label} → "
                      f"R² > {thr:.3f})")
    elif n_voxels == 0:
        sel = r2[r2 > 0].index
        sel_tag = "nvoxels-0"
        print(f"  {len(sel)} voxels selected (all r²>0)")
    else:
        sel = r2.sort_values(ascending=False).index[:n_voxels]
        sel_tag = f"nvoxels-{n_voxels}"
        print(f"  {len(sel)} voxels selected  (R² ≥ {float(r2.loc[sel].min()):.3f})")

    weights_sel = weights[sel]

    # Stimulus grid spans [0, π) at uniform resolution
    stim_grid = np.linspace(0, np.pi, n_orientations, endpoint=False,
                            dtype=np.float32)

    for ses_i in sessions:
        cond = sub.get_mapping(ses_i)
        print(f"\n  --- session {ses_i} (condition={cond}) ---")
        ses_paradigm = get_gabor_paradigm(sub, [ses_i])
        ses_betas = sub.get_single_trial_estimates([ses_i], desc="gabor",
                                                   smoothed=smoothed)
        ses_data = pd.DataFrame(masker.transform(ses_betas).astype(np.float32))
        ses_data_sel = ses_data[sel]

        print(f"  fitting noise model ({n_noise_iterations} iter)…")
        residfit = ResidualFitter(model, ses_data_sel, ses_paradigm,
                                   parameters=basis_pars,
                                   weights=weights_sel)
        omega, dof = residfit.fit(init_sigma2=1e-2, init_dof=10.0,
                                   learning_rate=0.05,
                                   max_n_iterations=n_noise_iterations)
        dof_str = f"{float(dof):.1f}" if dof is not None else "None (Gaussian)"
        print(f"  noise model: dof={dof_str}")

        print(f"  simulating {n_simulations} repeats × {n_orientations} "
              f"orientations ({n_simulations * n_orientations} trials)…")
        true_arr, decoded_arr = simulate_decode_session(
            model, basis_pars, weights_sel, omega, dof,
            stim_grid, n_simulations, batch_stimuli=batch_stimuli)

        agg = aggregate_per_stimulus(true_arr, decoded_arr, stim_grid)

        out_dir = (bids_folder / "derivatives" / "encoding_models"
                   / "vonmises" / f"sub-{subject}"
                   / f"ses-{ses_i}" / "func")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fn = (out_dir /
                  f"sub-{subject}_ses-{ses_i}_task-abstractvalue"
                  f"_mask-{mask_desc}_{sel_tag}"
                  f"_nsims-{n_simulations}{smooth_label}"
                  f"_desc-expected_decoded_orientation_pe.tsv")
        agg.to_csv(out_fn, sep="\t", index=False)
        print(f"  saved {out_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subject", help="Subject label without 'sub-'")
    parser.add_argument("--sessions", type=int, nargs="+", default=None)
    parser.add_argument("--roi", default="BensonV1")
    parser.add_argument("--hemi", default="LR")
    parser.add_argument("--n-voxels", type=int, default=100)
    parser.add_argument("--fdr-alpha", type=float, default=None,
                        help="FDR α on the vonmises whole-brain mixture "
                             "(mutually exclusive with --p-signal-thr)")
    parser.add_argument("--p-signal-thr", type=float, default=None,
                        help="P(signal|r²) threshold on the same mixture")
    parser.add_argument("--fdr-fallback-n-voxels", type=int, default=100)
    parser.add_argument("--n-basis", type=int, default=8)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--n-orientations", type=int, default=180)
    parser.add_argument("--n-simulations", type=int, default=1000)
    parser.add_argument("--n-noise-iterations", type=int, default=1000)
    parser.add_argument("--batch-stimuli", type=int, default=25)
    parser.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    parser.add_argument("--fmriprep-deriv", default="fmriprep",
                        choices=["fmriprep", "fmriprep-t2w", "fmriprep-flair"])
    parser.add_argument("--smoothed", action="store_true")
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions, roi=args.roi, hemi=args.hemi,
         n_voxels=args.n_voxels, n_basis=args.n_basis, kappa=args.kappa,
         n_orientations=args.n_orientations,
         n_simulations=args.n_simulations,
         n_noise_iterations=args.n_noise_iterations,
         batch_stimuli=args.batch_stimuli,
         fdr_alpha=args.fdr_alpha, p_signal_thr=args.p_signal_thr,
         fdr_fallback_n_voxels=args.fdr_fallback_n_voxels,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed)
