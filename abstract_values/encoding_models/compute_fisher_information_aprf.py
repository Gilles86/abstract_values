#!/usr/bin/env python3
"""
Compute Fisher information for abstract value from the aPRF (LogGaussianPRF) model.

Loads already-fitted aPRF parameters, selects the top n_voxels within the
supplied mask by R², fits a noise model (ResidualFitter), then computes
Fisher information over a dense value grid.

For Gaussian noise (default method) the Fisher information is computed
analytically:
    FI(x) = grad(x)ᵀ Ω⁻¹ grad(x)
where grad(x)[v] = df_v/dx and Ω is the voxel noise covariance.

Output
------
  The aPRF parameters themselves are always loaded from the all-sessions
  joint fit at ``derivatives/encoding_models/aprf/sub-<subject>/func/``.
  The ``--sessions`` argument here only selects which session's BOLD data
  the noise model and Fisher information are computed *on*.

  standard model:
    derivatives/encoding_models/aprf/sub-<subject>/[ses-<N>/]func/
      sub-<subject>[_ses-<N>]_task-abstractvalue_mask-<mask_desc>_nvoxels-<n>_desc-fisherinfo_pe.tsv

  session-shift model (one file per session):
    derivatives/encoding_models/aprf-session-shift/sub-<subject>/ses-<N>/func/
      sub-<subject>_ses-<N>_task-abstractvalue_mask-<mask_desc>_nvoxels-<n>_desc-fisherinfo_pe.tsv

  TSV: one row per value (CHF), one column: fisher_information

Usage
-----
  python compute_fisher_information_aprf.py pil01 --sessions 1
  python compute_fisher_information_aprf.py pil01 --sessions 1 --roi BensonV1 --hemi LR
  python compute_fisher_information_aprf.py pil01 --roi NPC --hemi None
  python compute_fisher_information_aprf.py pil01 --model session-shift
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import LogGaussianPRF
from braincoder.optimize import ResidualFitter

from abstract_values.utils.data import Subject, BIDS_FOLDER


def _select_voxels(r2, n_voxels, fdr_alpha, fdr_fallback_n_voxels,
                    subject, smoothed, bids_folder, mixture_model='aprf'):
    """Return (selected_voxel_index, sel_tag) for the FI computation.
    Mirrors the EU + decode selection logic: --fdr-alpha overrides
    --n-voxels and uses the whole-brain R² mixture for the given model
    (``aprf`` for NPCr, ``vonmises`` for V1)."""
    if fdr_alpha is not None:
        from abstract_values.encoding_models.compute_r2_mixture \
            import get_brain_fdr_threshold
        res = get_brain_fdr_threshold(
            subject, model=mixture_model, bids_folder=bids_folder,
            alpha=fdr_alpha, smoothed=smoothed)
        if res is None:
            raise RuntimeError(
                f'Whole-brain {mixture_model} R² mixture missing and '
                f'auto-fit failed for sub-{subject}.')
        thr = res['threshold']
        if res['degenerate'] or not np.isfinite(thr):
            sel = r2.sort_values(ascending=False).index[:fdr_fallback_n_voxels]
            label = (f'fdr{int(round(fdr_alpha * 100)):02d} → fallback '
                      f'top-{fdr_fallback_n_voxels} (mixture degenerate)')
        else:
            sel = r2[r2 > thr].index
            if len(sel) < 10:
                sel = r2.sort_values(ascending=False).index[:fdr_fallback_n_voxels]
                label = (f'fdr{int(round(fdr_alpha * 100)):02d} → fallback '
                          f'top-{fdr_fallback_n_voxels} (only '
                          f'{(r2 > thr).sum()} passed)')
            else:
                label = f'fdr{int(round(fdr_alpha * 100)):02d} → R² > {thr:.3f}'
        tag = f'fdr{int(round(fdr_alpha * 100)):02d}'
    elif n_voxels == 0:
        sel = r2[r2 > 0].index
        label = 'all R²>0'
        tag = '0'
    else:
        sel = r2.sort_values(ascending=False).index[:n_voxels]
        label = f'top {n_voxels} by R²'
        tag = str(n_voxels)
    print(f'  {len(sel)} voxels selected  ({label})')
    return sel, tag


def get_value_paradigm(sub, sessions):
    """Return DataFrame with column 'x' = objective CHF value (float32)."""
    rows = []
    for session in sessions:
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values('onset')
            for _, row in run_ev[run_ev['event_type'] == 'gabor'].iterrows():
                rows.append(float(row['value']))
    return pd.DataFrame({'x': np.array(rows, dtype=np.float32)})


def _fit_and_compute_fi(model_obj, pars_df, sel, data_sel, paradigm,
                         n_noise_iterations, n_mc_samples, n_values,
                         value_min=0.5, value_max=50.0,
                         spherical_noise=True):
    """Fit noise model and compute Fisher information. Returns (stimuli, fisher_info)."""
    model_obj.parameters = pars_df
    model_obj.apply_mask(sel)
    model_obj.init_pseudoWWT(paradigm['x'].values, model_obj.parameters)
    print(f'  fitting noise model ({n_noise_iterations} iterations, '
          f'spherical={spherical_noise})...')
    residfit = ResidualFitter(model_obj, data_sel, paradigm)
    omega, dof = residfit.fit(
        init_sigma2=1e-2, init_dof=10.0,
        learning_rate=0.05, spherical=spherical_noise,
        max_n_iterations=n_noise_iterations)
    dof_str = f'{float(dof):.1f}' if dof is not None else 'None (Gaussian)'
    print(f'  noise model: dof={dof_str}')

    stimuli = np.linspace(value_min, value_max, n_values, dtype=np.float32)
    print(f'  computing Fisher information over {n_values} values '
          f'({value_min:.1f}–{value_max:.1f} CHF)...')
    # Always use the closed-form (analytical) path — braincoder 0.5.2+ has
    # the multivariate-t prefactor (ν+p)/(ν+p+2), so dof can be passed
    # directly. The MC branch is left in braincoder for reference but
    # adds noise + an int32 overflow risk at large n × n_vox × n_stim.
    fisher_info = model_obj.get_fisher_information(
        stimuli=stimuli, omega=omega, dof=dof, weights=None,
        parameters=model_obj.parameters, analytical=True)

    print(f'  mean FI={float(fisher_info.mean()):.4f}  '
          f'peak at {float(stimuli[fisher_info.values.argmax()]):.2f} CHF')
    return stimuli, fisher_info


def main(subject, sessions=None, roi='NPCr', hemi='None', n_voxels=250,
         n_values=200, n_noise_iterations=1000, n_mc_samples=1000,
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         smoothed=False, model='standard',
         value_min=0.5, value_max=50.0,
         spherical_noise=True,
         fdr_alpha=None, fdr_fallback_n_voxels=100):

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if sessions is None:
        sessions = sub.get_sessions()
    sessions = sorted(sessions)

    smooth_label = '_smoothed' if smoothed else ''
    hemi_arg  = None if hemi == 'None' else hemi
    mask_desc = f'{roi}{"_hemi-" + hemi if hemi_arg else ""}'
    print(f'sub-{subject}  model={model}  [aPRF Fisher information]')

    # ── ROI mask (need any beta image for affine/shape) ───────────────────────
    ref_betas = sub.get_single_trial_estimates(sessions, desc='gabor',
                                               smoothed=smoothed)
    mask_img = sub.get_roi_mask(roi=roi, hemi=hemi_arg)
    masker = NiftiMasker(mask_img=mask_img,
                         target_affine=ref_betas.affine,
                         target_shape=ref_betas.shape[:3]).fit()

    # ── branch on model type ──────────────────────────────────────────────────
    if model == 'session-shift':
        _main_session_shift(
            subject=subject, sub=sub, sessions=sessions,
            masker=masker, mask_desc=mask_desc,
            n_voxels=n_voxels, n_values=n_values,
            n_noise_iterations=n_noise_iterations,
            n_mc_samples=n_mc_samples,
            bids_folder=bids_folder, smoothed=smoothed,
            smooth_label=smooth_label,
            spherical_noise=spherical_noise,
            fdr_alpha=fdr_alpha,
            fdr_fallback_n_voxels=fdr_fallback_n_voxels)
    else:
        _main_standard(
            subject=subject, sub=sub, sessions=sessions,
            masker=masker, mask_desc=mask_desc,
            ref_betas=ref_betas,
            n_voxels=n_voxels, n_values=n_values,
            n_noise_iterations=n_noise_iterations,
            n_mc_samples=n_mc_samples,
            bids_folder=bids_folder, smoothed=smoothed,
            smooth_label=smooth_label,
            value_min=value_min, value_max=value_max,
            spherical_noise=spherical_noise,
            fdr_alpha=fdr_alpha,
            fdr_fallback_n_voxels=fdr_fallback_n_voxels)


def _main_standard(subject, sub, sessions, masker, mask_desc, ref_betas,
                   n_voxels, n_values, n_noise_iterations, n_mc_samples,
                   bids_folder, smoothed, smooth_label,
                   value_min=0.5, value_max=50.0,
                   spherical_noise=True,
                   fdr_alpha=None, fdr_fallback_n_voxels=100):
    """Standard aPRF Fisher information.

    Selects voxels and computes FI using the BOLD data from ``sessions``;
    aPRF parameters always come from the joint (all-sessions) fit.
    """
    # FI output filename encodes which session's BOLD data was used for
    # the noise model / FI computation. When all sessions are pooled, no
    # ses- entity is added.
    ses_dir    = f'ses-{sessions[0]}' if len(sessions) == 1 else ''
    ses_entity = f'_ses-{sessions[0]}' if len(sessions) == 1 else ''

    paradigm = get_value_paradigm(sub, sessions)
    print(f'  {len(paradigm)} gabor trials  '
          f'value range: {float(paradigm["x"].min()):.1f}–{float(paradigm["x"].max()):.1f} CHF'
          f'  (FI grid: {value_min:.1f}–{value_max:.1f} CHF)')

    assert ref_betas.shape[3] == len(paradigm), (
        f'Beta count mismatch: {ref_betas.shape[3]} vs {len(paradigm)}')

    data = pd.DataFrame(masker.transform(ref_betas).astype(np.float32))
    print(f'  {data.shape[1]} voxels in mask ({mask_desc})')

    # aPRF parameters always come from the all-sessions joint fit. The
    # `sessions` argument controls only the BOLD selection above; it does
    # not affect which aPRF parameter file is loaded. (Subject.get_prf_parameters
    # now always returns the joint-fit path regardless of any argument.)
    pars_imgs = sub.get_prf_parameters(smoothed=smoothed)
    pars_df = pd.DataFrame({
        'mode':      masker.transform(pars_imgs['mode']).squeeze().astype(np.float32),
        'fwhm':      masker.transform(pars_imgs['fwhm']).squeeze().astype(np.float32),
        'amplitude': masker.transform(pars_imgs['amplitude']).squeeze().astype(np.float32),
        'baseline':  masker.transform(pars_imgs['baseline']).squeeze().astype(np.float32),
    })
    r2 = pd.Series(masker.transform(pars_imgs['r2']).squeeze().astype(np.float32))

    sel, sel_tag = _select_voxels(r2, n_voxels, fdr_alpha,
                                     fdr_fallback_n_voxels,
                                     subject, smoothed,
                                     bids_folder, mixture_model='aprf')

    model_obj = LogGaussianPRF(allow_neg_amplitudes=True,
                               parameterisation='mode_fwhm_natural')
    stimuli, fisher_info = _fit_and_compute_fi(
        model_obj, pars_df, sel, data[sel], paradigm,
        n_noise_iterations, n_mc_samples, n_values,
        value_min=value_min, value_max=value_max,
        spherical_noise=spherical_noise)

    out_dir = bids_folder / 'derivatives' / 'encoding_models' / 'aprf' / f'sub-{subject}'
    if ses_dir:
        out_dir = out_dir / ses_dir
    out_dir = out_dir / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)

    noise_tag = '_noise-spherical' if spherical_noise else ''
    out_fn = (out_dir /
              f'sub-{subject}{ses_entity}_task-abstractvalue'
              f'_mask-{mask_desc}_nvoxels-{sel_tag}{noise_tag}{smooth_label}_desc-fisherinfo_pe.tsv')
    pd.DataFrame({'fisher_information': fisher_info.values}, index=stimuli).to_csv(
        out_fn, sep='\t', header=True)
    print(f'  saved to {out_fn}')


def _main_session_shift(subject, sub, sessions, masker, mask_desc,
                        n_voxels, n_values, n_noise_iterations, n_mc_samples,
                        bids_folder, smoothed, smooth_label,
                        spherical_noise=True,
                        fdr_alpha=None, fdr_fallback_n_voxels=100):
    """Session-shift aPRF Fisher information (one FI curve per session).

    Parameters are loaded from aprf-session-shift/.  For each session i the
    mode is taken from mode_i; fwhm/amplitude/baseline are shared.  The noise
    model is fitted separately on each session's betas.
    """
    from nilearn import image as nli

    ss_dir = (bids_folder / 'derivatives' / 'encoding_models'
              / 'aprf-session-shift' / f'sub-{subject}' / 'func')

    def load_param(desc):
        # smooth_label is "" for unsmoothed, "_smoothed" otherwise — matches
        # how the session-shift fitter names its outputs.
        fn = (ss_dir / f'sub-{subject}_task-abstractvalue_space-T1w'
                       f'_desc-{desc}{smooth_label}_pe.nii.gz')
        if not fn.exists():
            raise FileNotFoundError(f'No session-shift parameter file: {fn}')
        return nli.load_img(str(fn))

    fwhm_arr = masker.transform(load_param('fwhm')).squeeze().astype(np.float32)
    amp_arr  = masker.transform(load_param('amplitude')).squeeze().astype(np.float32)
    base_arr = masker.transform(load_param('baseline')).squeeze().astype(np.float32)
    r2       = pd.Series(masker.transform(load_param('r2')).squeeze().astype(np.float32))

    # Filter voxels with invalid parameters:
    # - mode must be > 0 (log-Gaussian undefined at mode ≤ 0)
    # - fwhm must be > 0 (zero fwhm = flat tuning = zero gradient = zero FI)
    mode_descs = [f'mode_{i}' for i in range(1, len(sessions) + 1)]
    mode_arrays = {md: masker.transform(load_param(md)).squeeze().astype(np.float32)
                   for md in mode_descs}
    valid = (fwhm_arr > 0) & (amp_arr != 0)
    for arr in mode_arrays.values():
        valid &= arr > 0
    r2_valid = r2[valid]

    print(f'  {len(r2)} voxels in mask ({mask_desc}), {valid.sum()} with valid modes')

    sel, sel_tag = _select_voxels(r2_valid, n_voxels, fdr_alpha,
                                     fdr_fallback_n_voxels,
                                     subject, smoothed,
                                     bids_folder, mixture_model='aprf')

    for ses_i, mode_desc in zip(sessions, mode_descs):
        print(f'\n  --- session {ses_i} ({mode_desc}) ---')

        mode_arr = mode_arrays[mode_desc]
        pars_df = pd.DataFrame({
            'mode':      mode_arr,
            'fwhm':      fwhm_arr,
            'amplitude': amp_arr,
            'baseline':  base_arr,
        })

        ses_paradigm = get_value_paradigm(sub, [ses_i])
        ses_betas    = sub.get_single_trial_estimates([ses_i], desc='gabor',
                                                     smoothed=smoothed)
        ses_data     = pd.DataFrame(masker.transform(ses_betas).astype(np.float32))

        print(f'  {len(ses_paradigm)} gabor trials  '
              f'value range: {float(ses_paradigm["x"].min()):.1f}–'
              f'{float(ses_paradigm["x"].max()):.1f} CHF')

        model_obj = LogGaussianPRF(allow_neg_amplitudes=True,
                                   parameterisation='mode_fwhm_natural')
        stimuli, fisher_info = _fit_and_compute_fi(
            model_obj, pars_df, sel, ses_data[sel], ses_paradigm,
            n_noise_iterations, n_mc_samples, n_values,
            spherical_noise=spherical_noise)

        out_dir = (bids_folder / 'derivatives' / 'encoding_models'
                   / 'aprf-session-shift' / f'sub-{subject}'
                   / f'ses-{ses_i}' / 'func')
        out_dir.mkdir(parents=True, exist_ok=True)

        noise_tag = '_noise-spherical' if spherical_noise else ''
        out_fn = (out_dir /
                  f'sub-{subject}_ses-{ses_i}_task-abstractvalue'
                  f'_mask-{mask_desc}_nvoxels-{sel_tag}{noise_tag}{smooth_label}_desc-fisherinfo_pe.tsv')
        pd.DataFrame({'fisher_information': fisher_info.values}, index=stimuli).to_csv(
            out_fn, sep='\t', header=True)
        print(f'  saved to {out_fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-'")
    parser.add_argument('--sessions', type=int, nargs='+', default=None)
    parser.add_argument('--roi', default='NPCr',
                        help='ROI label (default: NPCr)')
    parser.add_argument('--hemi', default='None',
                        help="Hemisphere: LR, L, R, or None (default: None)")
    parser.add_argument('--n-voxels', type=int, default=250,
                        help='Top-N voxels by R² within mask (0 = all R²>0, default: 250)')
    parser.add_argument('--n-values', type=int, default=200,
                        help='Number of value points in Fisher information grid (default: 200)')
    parser.add_argument('--value-min', type=float, default=0.5,
                        help='Lower bound of FI grid in CHF (default: 0.5)')
    parser.add_argument('--value-max', type=float, default=50.0,
                        help='Upper bound of FI grid in CHF (default: 50.0)')
    parser.add_argument('--n-noise-iterations', type=int, default=1000)
    parser.add_argument('--n-mc-samples', type=int, default=1000,
                        help='Monte Carlo samples for Fisher information (default: 1000)')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                        choices=['fmriprep', 'fmriprep-t2w', 'fmriprep-flair'])
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--fdr-alpha', type=float, default=None,
                        help='FDR α on the whole-brain aprf R² mixture '
                             '(overrides --n-voxels).')
    parser.add_argument('--fdr-fallback-n-voxels', type=int, default=100)
    sph_grp = parser.add_mutually_exclusive_group()
    sph_grp.add_argument('--spherical-noise',    dest='spherical_noise',
                          action='store_true', default=True,
                          help='(default) iid Gaussian residual noise. '
                               'Output filename gets _noise-spherical tag.')
    sph_grp.add_argument('--no-spherical-noise', dest='spherical_noise',
                          action='store_false',
                          help='Fit full residual covariance instead. '
                               'Output: no _noise-* tag (legacy filename).')
    parser.add_argument('--model', default='standard',
                        choices=['standard', 'session-shift'],
                        help='aPRF model type (default: standard)')
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions,
         roi=args.roi, hemi=args.hemi,
         n_voxels=args.n_voxels, n_values=args.n_values,
         n_noise_iterations=args.n_noise_iterations,
         n_mc_samples=args.n_mc_samples,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed, model=args.model,
         value_min=args.value_min, value_max=args.value_max,
         spherical_noise=args.spherical_noise,
         fdr_alpha=args.fdr_alpha,
         fdr_fallback_n_voxels=args.fdr_fallback_n_voxels)
