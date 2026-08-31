#!/usr/bin/env python3
"""
Bayesian decoding of abstract value (CHF) from single-trial fMRI amplitudes.

Overview
--------
Leave-one-run-out cross-validation.  In each fold:

  1. Fit a Gaussian nPRF encoding model on training runs (grid search +
     gradient descent via ParameterFitter; same approach as fit_aprf.py).
  2. Select voxels: top n_voxels by training R², or (when n_voxels=0) all
     voxels with nested cross-validated R² > 0 (inner leave-one-run-out CV
     within the training set — no circularity). With ``--null-gate``
     (requires n_voxels=0), the threshold is instead the nested cv-R² of
     a null (per-voxel training-mean) model, evaluated on the same inner
     folds — i.e. ``cvR² > cvR²_null`` rather than ``cvR² > 0``.
  3. Fit a multivariate Student-t residual noise model (ResidualFitter).
  4. Evaluate P(data | value) over the stimulus grid for each test trial.
     This unnormalised likelihood serves as the posterior PDF under a flat
     prior.

Mask suggestions
----------------
  brain (default): fmriprep whole-brain mask
  V1/V2/V3:        derivatives/masks/sub-<s>/ses-<n>/anat/
                   sub-<s>_ses-<n>_space-T1w_hemi-LR_desc-BensonV1_mask.nii.gz
  IPS:             create with a script analogous to create_roi_masks.py
                   (project IPS surface labels to T1w volume)

Output
------
  derivatives/decoding/value/sub-<subject>/<ses_label>/func/
    sub-<subject>_<ses_label>_mask-<mask_desc>_nvoxels-<n>
    _noise-<spherical|full>[_smoothed]_pars.tsv

  One row per test trial, columns = value grid (CHF).
  Row index: (session, run, trial_nr, true_value_chf).

Usage
-----
  python decode_value.py pil01 --sessions 1
  python decode_value.py pil01 --sessions 1 --n-voxels 200
  python decode_value.py pil01 --sessions 1 \\
      --mask /data/ds-abstractvalue/derivatives/masks/sub-pil01/ses-1/anat/\\
             sub-pil01_ses-1_space-T1w_hemi-LR_desc-BensonV1_mask.nii.gz \\
      --mask-desc BensonV1
  python decode_value.py pil01 --sessions 1 --spherical-noise --smoothed
  python decode_value.py pil01 --sessions 1 --debug
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import LogGaussianPRF, LinearModelWithBaseline
from braincoder.optimize import ParameterFitter, ResidualFitter, WeightFitter
from braincoder.utils import get_rsq

from abstract_values.encoding_models.models import GaussianValuePRF
from abstract_values.utils.data import Subject, BIDS_FOLDER
from abstract_values.encoding_models.geodesic_noise import (
    geodesic_snap_for_masker, geodesic_D_for_selection,
)
from abstract_values.encoding_models.voxel_selection import (
    STATUS_EMPTY, STATUS_OK, concat_posteriors, select_voxels, warn_if_degraded,
)


def _build_model(model_type, allow_neg_amplitudes=True):
    """Return a fresh encoding model for the requested tuning family.

    For ``'weighted'`` (the basis-set analog of vonmises), use
    LogGaussianPRF with the ``mode_fwhm_natural`` parameterisation —
    matches ``fit_aprf_weighted.py``."""
    if model_type == 'loggauss':
        return LogGaussianPRF(allow_neg_amplitudes=allow_neg_amplitudes,
                              parameterisation='mu_sd_natural')
    if model_type == 'gaussian':
        return GaussianValuePRF(allow_neg_amplitudes=allow_neg_amplitudes)
    if model_type == 'weighted':
        return LogGaussianPRF(parameterisation='mode_fwhm_natural')
    raise ValueError(f'Unknown model_type: {model_type!r}')


def _grid_ranges(model_type, value_min, value_max, n_loc, n_width):
    """Location + width grids, matched to the tuning family's parameterisation."""
    locs = np.linspace(value_min, value_max, n_loc, dtype=np.float32)
    if model_type == 'loggauss':
        # (mu, sd) — sd capped at half the value range
        widths = np.linspace(1.0, (value_max - value_min) / 2, n_width, dtype=np.float32)
    else:  # gaussian: (mode, fwhm) — fwhm ≈ 2.355 × sd so allow full range
        widths = np.linspace(1.0, value_max - value_min, n_width, dtype=np.float32)
    return locs, widths


def make_value_basis_parameters(n_basis, value_min, value_max, fwhm=None):
    """Fixed log-Gaussian basis on the value axis (amp=1, baseline=0).

    Mirrors ``fit_aprf_weighted.py``: modes are uniformly spaced over
    [value_min, value_max]; default fwhm = 2 × inter-basis spacing in
    CHF, matching the orientation-side vonmises basis (8 bumps with
    width ≈ 2 × spacing)."""
    modes = np.linspace(value_min, value_max, n_basis).astype(np.float32)
    spacing = modes[1] - modes[0] if n_basis > 1 else (value_max - value_min)
    if fwhm is None:
        fwhm = float(2.0 * spacing)
    return pd.DataFrame({
        'mode':      modes,
        'fwhm':      np.full(n_basis, fwhm, dtype=np.float32),
        'amplitude': np.ones(n_basis,  dtype=np.float32),
        'baseline':  np.zeros(n_basis, dtype=np.float32),
    })


def _out_subdir(model_type):
    return {
        'loggauss': 'value',
        'gaussian': 'value-gauss',
        'weighted': 'value-weighted',
        'linear':   'value-linear',
    }[model_type]


def get_value_paradigm(sub, sessions):
    """Return DataFrame indexed by (session, run, trial_nr) with column 'x'.

    x = objective CHF value, in the same order as the gabor betas written by
    fit_glmsingle (session → run → events sorted by onset, gabor only).
    """
    rows = []
    for session in sessions:
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values('onset')
            for _, row in run_ev[run_ev['event_type'] == 'gabor'].iterrows():
                rows.append({
                    'session':  session,
                    'run':      run,
                    'trial_nr': int(row['trial_nr']),
                    'x':        np.float32(float(row['value'])),
                })
    df = pd.DataFrame(rows).set_index(['session', 'run', 'trial_nr'])
    return df[['x']]


def _run_weighted_folds(sub, sessions, paradigm, data, stimulus_range,
                        value_min, value_max,
                        n_voxels, fdr_alpha, p_signal_thr,
                        fdr_fallback_n_voxels,
                        n_basis, basis_fwhm,
                        weight_alpha, lambd, spherical_noise,
                        smoothed, bids_folder, debug):
    """Basis-set (WeightFitter) decoding flow — value-axis analog of decode_gabor.

    Mirrors decode_gabor.py: 8 fixed log-Gaussian basis RFs, closed-form
    weight fit, optional FDR/p_signal voxel selection against the
    *aprf-weighted* whole-brain mixture (not aprf).
    """
    basis_pars = make_value_basis_parameters(n_basis, value_min, value_max,
                                             fwhm=basis_fwhm)
    eff_fwhm = float(basis_pars['fwhm'].iloc[0])
    print(f'  {n_basis} log-Gaussian basis functions  fwhm={eff_fwhm:.2f} CHF')

    all_pdfs = []
    fold_meta = []
    all_runs = [(s, r) for s in sessions for r in sub.get_runs(s)]

    for test_session, test_run in all_runs:
        print(f'\n  [fold] hold-out ses-{test_session} run-{test_run}')

        test_idx  = (paradigm.index.get_level_values('session') == test_session) & \
                    (paradigm.index.get_level_values('run') == test_run)
        train_idx = ~test_idx

        train_paradigm = paradigm.loc[train_idx]
        test_paradigm  = paradigm.loc[test_idx]
        train_data     = data.loc[train_idx]
        test_data      = data.loc[test_idx]

        # ── fit basis weights (closed-form lstsq, optional ridge) ───────────
        model = _build_model('weighted')
        weights = WeightFitter(model, basis_pars, train_data,
                               train_paradigm).fit(alpha=weight_alpha)
        # weights: DataFrame (n_basis × n_voxels)

        # ── voxel selection ────────────────────────────────────────────────
        if n_voxels == 0 or fdr_alpha is not None or p_signal_thr is not None:
            inner_runs = sorted(set(zip(
                train_paradigm.index.get_level_values('session'),
                train_paradigm.index.get_level_values('run'))))
            inner_r2s = []
            for inner_ses, inner_run in inner_runs:
                inner_test_idx = (
                    (train_paradigm.index.get_level_values('session') == inner_ses) &
                    (train_paradigm.index.get_level_values('run') == inner_run))
                inner_train_paradigm = train_paradigm.loc[~inner_test_idx]
                inner_test_paradigm  = train_paradigm.loc[inner_test_idx]
                inner_train_data     = train_data.loc[~inner_test_idx]
                inner_test_data      = train_data.loc[inner_test_idx]

                inner_model = _build_model('weighted')
                inner_w = WeightFitter(inner_model, basis_pars,
                                       inner_train_data,
                                       inner_train_paradigm).fit(alpha=weight_alpha)
                inner_bp = inner_model.basis_predictions(inner_test_paradigm, basis_pars)
                inner_pred = pd.DataFrame(inner_bp @ inner_w.values,
                                          index=inner_test_data.index,
                                          columns=inner_test_data.columns)
                inner_r2s.append(get_rsq(inner_test_data, inner_pred))

            cv_r2 = pd.concat(inner_r2s, axis=1).mean(axis=1)
            sel, status, msg = select_voxels(
                cv_r2, mixture_model='aprf-weighted',
                subject=sub.subject_id, bids_folder=bids_folder,
                smoothed=smoothed, fdr_alpha=fdr_alpha,
                p_signal_thr=p_signal_thr,
                fdr_fallback_n_voxels=fdr_fallback_n_voxels)
            print(msg)
        else:
            basis_pred = model.basis_predictions(train_paradigm, basis_pars)
            pred_train = pd.DataFrame(basis_pred @ weights.values,
                                      index=train_data.index,
                                      columns=train_data.columns)
            r2_train = get_rsq(train_data, pred_train)
            sel = r2_train.sort_values(ascending=False).index[:n_voxels]
            status = STATUS_OK
            print(f'    {len(sel)} voxels selected  '
                  f'(train R² ≥ {float(r2_train.loc[sel].min()):.3f})')

        fold_meta.append(dict(session=test_session, run=test_run,
                              n_voxels_selected=len(sel), status=status))
        if status == STATUS_EMPTY:
            # Previously unguarded: an empty selection fell through into
            # ResidualFitter and died there with an opaque error.
            continue
        weights_sel    = weights[sel]
        train_data_sel = train_data[sel]
        test_data_sel  = test_data[sel]

        # ── fit noise model ────────────────────────────────────────────────
        n_iter_noise = 100 if debug else 1000
        residfit = ResidualFitter(model, train_data_sel, train_paradigm,
                                  parameters=basis_pars, weights=weights_sel,
                                  lambd=lambd)
        omega, dof = residfit.fit(
            init_sigma2=0.1, init_dof=10.0, method='t',
            learning_rate=0.05, spherical=spherical_noise,
            max_n_iterations=n_iter_noise)
        print(f'    noise model: dof={float(dof):.1f}')

        # ── decode ─────────────────────────────────────────────────────────
        pdf = model.get_stimulus_pdf(test_data_sel, stimulus_range,
                                     parameters=basis_pars,
                                     weights=weights_sel,
                                     omega=omega, dof=dof,
                                     normalize=False)
        pdf.columns = stimulus_range
        pdf.index = pd.MultiIndex.from_arrays([
            test_paradigm.index.get_level_values('session'),
            test_paradigm.index.get_level_values('run'),
            test_paradigm.index.get_level_values('trial_nr'),
            test_paradigm['x'].values,
        ], names=['session', 'run', 'trial_nr', 'true_value_chf'])

        all_pdfs.append(pdf)

    return all_pdfs, fold_meta


def _run_linear_folds(sub, sessions, paradigm, data, stimulus_range,
                      n_voxels, fdr_alpha, p_signal_thr,
                      fdr_fallback_n_voxels,
                      weight_alpha, lambd, spherical_noise,
                      smoothed, bids_folder, debug):
    """Linear (signed-slope) decoding flow — value-axis analog of
    ``_run_weighted_folds``, but with a single linear regressor (the raw
    CHF value) instead of an 8-function log-Gaussian basis set.

    Slope + baseline are fit jointly per fold via one closed-form OLS
    regression (``WeightFitter(..., fit_intercept=True)`` — see the
    ``fit_intercept`` addition to braincoder's ``WeightFitter``, and its
    encoding-side analog ``LinearValuePRF`` / ``ModelSpec.is_linear`` in
    ``fit_pipeline.py``). No grid search, no gradient descent, no basis
    functions — mirrors the ``EncodingModel._predict`` path so
    ``get_stimulus_pdf`` (Bayesian decoding) and ``ResidualFitter`` (noise
    model) work exactly as they do for the weighted-basis flow.
    """
    all_pdfs = []
    fold_meta = []
    all_runs = [(s, r) for s in sessions for r in sub.get_runs(s)]

    def _fit_slope_baseline(train_x, train_d):
        m = LinearModelWithBaseline(paradigm=train_x, parameters=None)
        w, b = WeightFitter(m, None, train_d, train_x).fit(
            alpha=weight_alpha, fit_intercept=True)
        return m, w, pd.DataFrame({'baseline': b})

    def _predict(m, x, w, b_df):
        pred = m.predict(paradigm=x, parameters=b_df, weights=w)
        return pred if isinstance(pred, pd.DataFrame) else pd.DataFrame(
            np.asarray(pred), index=x.index, columns=w.columns)

    for test_session, test_run in all_runs:
        print(f'\n  [fold] hold-out ses-{test_session} run-{test_run}')

        test_idx  = (paradigm.index.get_level_values('session') == test_session) & \
                    (paradigm.index.get_level_values('run') == test_run)
        train_idx = ~test_idx

        train_paradigm = paradigm.loc[train_idx, ['x']]
        test_paradigm  = paradigm.loc[test_idx, ['x']]
        train_data     = data.loc[train_idx]
        test_data      = data.loc[test_idx]

        # ── fit slope + baseline (closed-form OLS) ───────────────────────────
        model, weights, baseline_df = _fit_slope_baseline(train_paradigm, train_data)

        # ── voxel selection ────────────────────────────────────────────────
        if n_voxels == 0 or fdr_alpha is not None or p_signal_thr is not None:
            inner_runs = sorted(set(zip(
                train_paradigm.index.get_level_values('session'),
                train_paradigm.index.get_level_values('run'))))
            inner_r2s = []
            for inner_ses, inner_run in inner_runs:
                inner_test_idx = (
                    (train_paradigm.index.get_level_values('session') == inner_ses) &
                    (train_paradigm.index.get_level_values('run') == inner_run))
                inner_train_paradigm = train_paradigm.loc[~inner_test_idx]
                inner_test_paradigm  = train_paradigm.loc[inner_test_idx]
                inner_train_data     = train_data.loc[~inner_test_idx]
                inner_test_data      = train_data.loc[inner_test_idx]

                inner_model, inner_w, inner_b_df = _fit_slope_baseline(
                    inner_train_paradigm, inner_train_data)
                inner_pred = _predict(inner_model, inner_test_paradigm, inner_w, inner_b_df)
                inner_r2s.append(get_rsq(inner_test_data, inner_pred))

            cv_r2 = pd.concat(inner_r2s, axis=1).mean(axis=1)
            sel, status, msg = select_voxels(
                cv_r2, mixture_model='aprf-linear',
                subject=sub.subject_id, bids_folder=bids_folder,
                smoothed=smoothed, fdr_alpha=fdr_alpha,
                p_signal_thr=p_signal_thr,
                fdr_fallback_n_voxels=fdr_fallback_n_voxels)
            print(msg)
        else:
            pred_train = _predict(model, train_paradigm, weights, baseline_df)
            r2_train = get_rsq(train_data, pred_train)
            sel = r2_train.sort_values(ascending=False).index[:n_voxels]
            status = STATUS_OK
            print(f'    {len(sel)} voxels selected  '
                  f'(train R² ≥ {float(r2_train.loc[sel].min()):.3f})')

        fold_meta.append(dict(session=test_session, run=test_run,
                              n_voxels_selected=len(sel), status=status))
        if status == STATUS_EMPTY:
            # Was a silent top-1 fallback — a posterior decoded from a voxel
            # with out-of-sample R² <= 0, indistinguishable downstream from a
            # real one. Drop the fold and record why instead.
            continue
        weights_sel    = weights[sel]
        baseline_sel   = baseline_df.loc[sel]
        train_data_sel = train_data[sel]
        test_data_sel  = test_data[sel]

        # ── fit noise model ────────────────────────────────────────────────
        n_iter_noise = 100 if debug else 1000
        residfit = ResidualFitter(model, train_data_sel, train_paradigm,
                                  parameters=baseline_sel, weights=weights_sel,
                                  lambd=lambd)
        omega, dof = residfit.fit(
            init_sigma2=0.1, init_dof=10.0, method='t',
            learning_rate=0.05, spherical=spherical_noise,
            max_n_iterations=n_iter_noise)
        print(f'    noise model: dof={float(dof):.1f}')

        # ── decode ─────────────────────────────────────────────────────────
        pdf = model.get_stimulus_pdf(test_data_sel, stimulus_range,
                                     parameters=baseline_sel,
                                     weights=weights_sel,
                                     omega=omega, dof=dof,
                                     normalize=False)
        pdf.columns = stimulus_range
        test_paradigm_full = paradigm.loc[test_idx]
        pdf.index = pd.MultiIndex.from_arrays([
            test_paradigm_full.index.get_level_values('session'),
            test_paradigm_full.index.get_level_values('run'),
            test_paradigm_full.index.get_level_values('trial_nr'),
            test_paradigm_full['x'].values,
        ], names=['session', 'run', 'trial_nr', 'true_value_chf'])

        all_pdfs.append(pdf)

    return all_pdfs, fold_meta


def main(subject, sessions=None, n_voxels=100, fdr_alpha=None,
         p_signal_thr=None, fdr_fallback_n_voxels=100,
         null_gate=False,
         n_iterations=1000,
         n_grid_mus=20, n_grid_sds=15, n_stimulus_grid=50,
         n_basis=8, basis_fwhm=None, weight_alpha=0.0,
         lambd=0.0, mask=None, mask_desc=None, spherical_noise=False,
         geodesic_noise=False, geodesic_hemi='R',
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         smoothed=False, debug=False, model_type='loggauss'):
    """If fdr_alpha is set, voxels are selected by FDR-thresholding the
    nested-CV R² using the whole-brain mixture. ``fdr_fallback_n_voxels``
    is used as a top-N fallback when the mixture is flagged degenerate.
    Output filename uses ``nvoxels-fdrNN`` where NN is alpha*100.

    If p_signal_thr is set instead, voxels are selected by thresholding
    the nested-CV R² at P(signal | r²) ≥ p_signal_thr (using the same
    whole-brain mixture). Same degenerate-fallback semantics. Output
    filename uses ``nvoxels-psigNN`` where NN is p*100. ``fdr_alpha`` and
    ``p_signal_thr`` are mutually exclusive.

    If null_gate is set (requires n_voxels=0, incompatible with
    fdr_alpha/p_signal_thr), the same inner leave-one-run-out folds also
    score a null (per-voxel training-mean) model, and voxels are selected
    by ``cvR² > cvR²_null`` rather than ``cvR² > 0``. Output filename uses
    ``nvoxels-nullgated``."""

    assert not (fdr_alpha is not None and p_signal_thr is not None), \
        "Pass at most one of --fdr-alpha / --p-signal-thr"
    assert not (null_gate and (fdr_alpha is not None or p_signal_thr is not None)), \
        "--null-gate is incompatible with --fdr-alpha / --p-signal-thr"
    assert not (null_gate and n_voxels != 0), \
        "--null-gate requires --n-voxels 0"

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if sessions is None:
        sessions = sub.get_sessions()

    if debug:
        n_iterations = 100

    ses_dir    = f'ses-{sessions[0]}' if len(sessions) == 1 else ''
    ses_entity = f'_ses-{sessions[0]}' if len(sessions) == 1 else ''
    print(f'sub-{subject}  {ses_dir or "all-sessions"}  '
          f'[abstract value decoding  model={model_type}]')

    # ── paradigm + data ───────────────────────────────────────────────────────
    paradigm = get_value_paradigm(sub, sessions)
    value_min = float(paradigm['x'].min())
    value_max = float(paradigm['x'].max())
    print(f'  {len(paradigm)} trials  value range: {value_min:.1f}–{value_max:.1f} CHF')

    betas_img = sub.get_single_trial_estimates(sessions, desc='gabor',
                                               smoothed=smoothed)
    assert betas_img.shape[3] == len(paradigm)

    if mask is None:
        raise ValueError('Please provide --mask and --mask-desc (whole-brain decoding is not supported)')
    masker = NiftiMasker(mask_img=mask,
                        target_affine=betas_img.affine,
                        target_shape=betas_img.shape[:3]).fit()

    data = pd.DataFrame(
        masker.transform(betas_img).astype(np.float32),
        index=paradigm.index)
    print(f'  {data.shape[1]} voxels in mask ({mask_desc})')

    # Geodesic noise model: build the voxel×voxel distance matrix once (in
    # masker column order); folds subset it via the selected-voxel positions.
    assert not (geodesic_noise and spherical_noise), \
        'geodesic_noise and spherical_noise are mutually exclusive'
    # braincoder's ResidualFitter.get_omega() checks lambd>0 BEFORE D: when
    # both are set it silently routes to the sample-covariance shrinkage
    # Omega and never touches alpha/beta/D at all (confirmed empirically on
    # decode_gabor.py's port — alpha/beta sit frozen at their init values).
    # Output would still say noise-geodesic while actually being
    # non-geodesic. Fail loudly instead of writing a misleadingly labeled
    # file.
    assert not (geodesic_noise and lambd > 0.0), \
        ('--geodesic-noise and --lambd > 0 are incompatible in the current '
         'braincoder ResidualFitter (lambd>0 silently overrides the '
         'geodesic Omega). Pass --lambd 0 with --geodesic-noise.')
    geo_snap = None
    if geodesic_noise:
        geo_snap = geodesic_snap_for_masker(
            masker, geodesic_hemi, subject, bids_folder, fmriprep_deriv)

    # ── stimulus grid ─────────────────────────────────────────────────────────
    stimulus_range = np.linspace(value_min, value_max, n_stimulus_grid,
                                 dtype=np.float32)
    print(f'  stimulus grid: {n_stimulus_grid} values '
          f'({value_min:.1f}–{value_max:.1f} CHF)')

    # ── output ────────────────────────────────────────────────────────────────
    out_dir = bids_folder / 'derivatives' / 'decoding' / _out_subdir(model_type) / f'sub-{subject}'
    if ses_dir:
        out_dir = out_dir / ses_dir
    out_dir = out_dir / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_label  = ('geodesic' if geodesic_noise
                    else 'spherical' if spherical_noise else 'full')
    smooth_label = '_smoothed' if smoothed else ''
    lambd_label  = f'_lambda-{lambd}' if lambd != 0.0 else ''
    if fdr_alpha is not None:
        nvox_tag = f'fdr{int(round(fdr_alpha * 100)):02d}'
    elif p_signal_thr is not None:
        nvox_tag = f'psig{int(round(p_signal_thr * 100)):02d}'
    elif null_gate:
        nvox_tag = 'nullgated'
    else:
        nvox_tag = str(n_voxels)
    out_fn = (out_dir /
              f'sub-{subject}{ses_entity}_mask-{mask_desc}'
              f'_nvoxels-{nvox_tag}_noise-{noise_label}{smooth_label}{lambd_label}_pars.tsv')

    # ── dispatch to basis-set flow if model_type=='weighted' ──────────────────
    if model_type == 'weighted':
        all_pdfs, fold_meta = _run_weighted_folds(
            sub, sessions, paradigm, data, stimulus_range,
            value_min, value_max,
            n_voxels, fdr_alpha, p_signal_thr, fdr_fallback_n_voxels,
            n_basis, basis_fwhm, weight_alpha, lambd, spherical_noise,
            smoothed, bids_folder, debug)
        warn_if_degraded(fold_meta, subject, mask_desc)
        pdfs = concat_posteriors(
            all_pdfs, stimulus_range,
            ['session', 'run', 'trial_nr', 'true_value_chf'])
        pdfs.to_csv(out_fn, sep='\t')
        print(f'\n  saved to {out_fn}')
        meta_fn = out_fn.with_name(out_fn.stem.replace('_pars', '_meta') + '.tsv')
        pd.DataFrame(fold_meta).to_csv(meta_fn, sep='\t', index=False)
        print(f'  meta  to {meta_fn}')
        return

    # ── dispatch to linear (signed-slope) flow if model_type=='linear' ────────
    if model_type == 'linear':
        all_pdfs, fold_meta = _run_linear_folds(
            sub, sessions, paradigm, data, stimulus_range,
            n_voxels, fdr_alpha, p_signal_thr, fdr_fallback_n_voxels,
            weight_alpha, lambd, spherical_noise,
            smoothed, bids_folder, debug)
        warn_if_degraded(fold_meta, subject, mask_desc)
        pdfs = concat_posteriors(
            all_pdfs, stimulus_range,
            ['session', 'run', 'trial_nr', 'true_value_chf'])
        pdfs.to_csv(out_fn, sep='\t')
        print(f'\n  saved to {out_fn}')
        meta_fn = out_fn.with_name(out_fn.stem.replace('_pars', '_meta') + '.tsv')
        pd.DataFrame(fold_meta).to_csv(meta_fn, sep='\t', index=False)
        print(f'  meta  to {meta_fn}')
        return

    # ── per-voxel parametric (loggauss / gaussian) flow below ────────────────
    # leave-one-run-out cross-validation
    all_pdfs = []
    fold_meta = []   # track n_voxels_selected per fold
    all_runs = [(s, r) for s in sessions for r in sub.get_runs(s)]

    for test_session, test_run in all_runs:
        print(f'\n  [fold] hold-out ses-{test_session} run-{test_run}')

        test_idx  = (paradigm.index.get_level_values('session') == test_session) & \
                    (paradigm.index.get_level_values('run') == test_run)
        train_idx = ~test_idx

        train_paradigm = paradigm.loc[train_idx]
        test_paradigm  = paradigm.loc[test_idx]
        train_data     = data.loc[train_idx]
        test_data      = data.loc[test_idx]

        # ── fit encoding model (grid search → gradient descent) ──────────────
        model = _build_model(model_type)
        fitter = ParameterFitter(model, train_data, train_paradigm)

        mus, sds = _grid_ranges(model_type, value_min, value_max,
                                 n_grid_mus, n_grid_sds)
        amplitudes = np.array([1.0], dtype=np.float32)
        baselines  = np.array([0.0], dtype=np.float32)

        grid_pars = fitter.fit_grid(mus, sds, amplitudes, baselines,
                                    use_correlation_cost=True)
        grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)
        pars = fitter.fit(max_n_iterations=n_iterations, init_pars=grid_pars)

        # ── voxel selection ────────────────────────────────────────────────────
        if n_voxels == 0 or fdr_alpha is not None or p_signal_thr is not None:
            # Nested CV: leave-one-run-out within training set to get unbiased R²
            inner_runs = list(zip(
                train_paradigm.index.get_level_values('session'),
                train_paradigm.index.get_level_values('run')))
            inner_runs = sorted(set(inner_runs))

            inner_r2s = []
            inner_null_r2s = [] if null_gate else None
            for inner_ses, inner_run in inner_runs:
                print(f'      [inner CV] hold-out ses-{inner_ses} run-{inner_run}')
                inner_test_idx = (
                    (train_paradigm.index.get_level_values('session') == inner_ses) &
                    (train_paradigm.index.get_level_values('run') == inner_run))
                inner_train_paradigm = train_paradigm.loc[~inner_test_idx]
                inner_test_paradigm  = train_paradigm.loc[inner_test_idx]
                inner_train_data     = train_data.loc[~inner_test_idx]
                inner_test_data      = train_data.loc[inner_test_idx]

                inner_model = _build_model(model_type)
                inner_fitter = ParameterFitter(inner_model, inner_train_data,
                                               inner_train_paradigm)
                inner_grid = inner_fitter.fit_grid(
                    mus, sds, amplitudes, baselines, use_correlation_cost=True)
                inner_grid = inner_fitter.refine_baseline_and_amplitude(inner_grid)
                inner_pars = inner_fitter.fit(max_n_iterations=n_iterations,
                                              init_pars=inner_grid)

                inner_pred = inner_model.predict(parameters=inner_pars,
                                                  paradigm=inner_test_paradigm)
                inner_r2s.append(get_rsq(inner_test_data, inner_pred))

                if null_gate:
                    # Null baseline on the SAME inner fold: per-voxel training
                    # mean, broadcast across the held-out inner-test trials.
                    mean_per_voxel = inner_train_data.mean(axis=0)
                    inner_null_pred = pd.DataFrame(
                        np.tile(mean_per_voxel.values, (len(inner_test_data), 1)),
                        index=inner_test_data.index, columns=inner_test_data.columns)
                    inner_null_r2s.append(get_rsq(inner_test_data, inner_null_pred))

            cv_r2 = pd.concat(inner_r2s, axis=1).mean(axis=1)
            cv_r2_null = (pd.concat(inner_null_r2s, axis=1).mean(axis=1)
                          if null_gate else None)
            sel, status, msg = select_voxels(
                cv_r2, mixture_model='aprf', subject=subject,
                bids_folder=bids_folder, smoothed=smoothed,
                fdr_alpha=fdr_alpha, p_signal_thr=p_signal_thr,
                fdr_fallback_n_voxels=fdr_fallback_n_voxels,
                cv_r2_null=cv_r2_null)
            print(msg)
        else:
            pred_train = model.predict(parameters=pars, paradigm=train_paradigm)
            r2_train   = get_rsq(train_data, pred_train)
            sel = r2_train.sort_values(ascending=False).index[:n_voxels]
            status = STATUS_OK
            print(f'    {len(sel)} voxels selected  '
                  f'(train R² ≥ {float(r2_train.loc[sel].min()):.3f})')

        fold_meta.append(dict(session=test_session, run=test_run,
                              n_voxels_selected=len(sel), status=status))
        if status == STATUS_EMPTY:
            # Previously unguarded: an empty selection fell through into
            # ResidualFitter and died there with an opaque error.
            continue
        pars_sel       = pars.loc[sel]
        train_data_sel = train_data[sel]
        test_data_sel  = test_data[sel]

        # ── fit noise model ───────────────────────────────────────────────────
        # Re-create model so state is clean for the selected voxels.
        # GaussianPRF-family models use a pseudo-WWT, so init_pseudoWWT must run first.
        model_sel = _build_model(model_type)
        model_sel.init_pseudoWWT(stimulus_range, pars_sel)
        n_iter_noise = 100 if debug else 5000
        residfit = ResidualFitter(model_sel, train_data_sel, train_paradigm,
                                  parameters=pars_sel, lambd=lambd)
        geo_kw = {}
        if geo_snap is not None:
            # sel are masker column positions (data.columns is a RangeIndex)
            geo_kw = dict(D=geodesic_D_for_selection(geo_snap, np.asarray(sel)),
                          init_alpha=0.5, init_beta=0.05)
        omega, dof = residfit.fit(
            init_sigma2=0.1, init_dof=10.0, method='t',
            learning_rate=0.05, spherical=spherical_noise,
            max_n_iterations=n_iter_noise, **geo_kw)
        print(f'    noise model: dof={float(dof):.1f}'
              + ('  (geodesic Ω)' if geo_snap is not None else ''))

        # ── decode ────────────────────────────────────────────────────────────
        pdf = model_sel.get_stimulus_pdf(test_data_sel, stimulus_range,
                                         parameters=pars_sel,
                                         omega=omega, dof=dof,
                                         normalize=False)
        pdf.columns = stimulus_range
        pdf.index = pd.MultiIndex.from_arrays([
            test_paradigm.index.get_level_values('session'),
            test_paradigm.index.get_level_values('run'),
            test_paradigm.index.get_level_values('trial_nr'),
            test_paradigm['x'].values,
        ], names=['session', 'run', 'trial_nr', 'true_value_chf'])

        all_pdfs.append(pdf)

    # ── save ──────────────────────────────────────────────────────────────────
    warn_if_degraded(fold_meta, subject, mask_desc)
    pdfs = concat_posteriors(
        all_pdfs, stimulus_range,
        ['session', 'run', 'trial_nr', 'true_value_chf'])
    pdfs.to_csv(out_fn, sep='\t')
    print(f'\n  saved to {out_fn}')

    # Save fold metadata (n_voxels_selected per fold)
    meta_fn = out_fn.with_name(out_fn.stem.replace('_pars', '_meta') + '.tsv')
    pd.DataFrame(fold_meta).to_csv(meta_fn, sep='\t', index=False)
    print(f'  meta  to {meta_fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-'")
    parser.add_argument('--sessions', type=int, nargs='+', default=None)
    parser.add_argument('--n-voxels', type=int, default=100,
                        help='Top-N voxels by training R² (0 = nested CV R²>0)')
    parser.add_argument('--fdr-alpha', type=float, default=None,
                        help='If set, voxels are selected by FDR-thresholding '
                             'nested-CV R² with the whole-brain mixture '
                             '(via compute_r2_mixture). Output: nvoxels-fdrNN.')
    parser.add_argument('--p-signal-thr', type=float, default=None,
                        help='If set, voxels are selected by thresholding '
                             'nested-CV R² at P(signal | r²) ≥ p (whole-brain '
                             'mixture via compute_r2_mixture). Mutually exclusive '
                             'with --fdr-alpha. Output: nvoxels-psigNN.')
    parser.add_argument('--fdr-fallback-n-voxels', type=int, default=100,
                        help='Top-N voxels by cv-R² to use when the whole-brain '
                             'mixture is flagged degenerate (default: 100). '
                             'Applies to both --fdr-alpha and --p-signal-thr.')
    parser.add_argument('--null-gate', action='store_true',
                        help='Requires --n-voxels 0. Select voxels by nested '
                             'CV R² > nested CV R²_null (per-voxel training-mean '
                             'baseline scored on the same inner folds), instead '
                             'of the plain nested CV R² > 0 threshold. '
                             'Output: nvoxels-nullgated.')
    parser.add_argument('--n-iterations', type=int, default=1000,
                        help='Max gradient descent iterations (default: 1000)')
    parser.add_argument('--n-stimulus-grid', type=int, default=50,
                        help='Number of value bins in stimulus grid (default: 50)')
    parser.add_argument('--lambd', type=float, default=0.0,
                        help='Lambda regularization for noise model (default: 0)')
    parser.add_argument('--mask', default=None,
                        help='Brain mask NIfTI (default: fmriprep brain mask)')
    parser.add_argument('--mask-desc', default=None,
                        help='Short label for mask used in output filename')
    parser.add_argument('--spherical-noise', action='store_true',
                        help='Fit isotropic noise model instead of full covariance')
    parser.add_argument('--geodesic-noise', action='store_true',
                        help='Fit a structured Omega with a geodesic-distance '
                             'spatial component (single-hemisphere ROI). '
                             'Output: noise-geodesic.')
    parser.add_argument('--geodesic-hemi', default='R', choices=['L', 'R'],
                        help='Hemisphere whose white surface defines geodesic '
                             'distance (default: R, for NPCr)')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                        choices=['fmriprep', 'fmriprep-t2w'])
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--model', default='loggauss',
                        choices=['loggauss', 'gaussian', 'weighted', 'linear'],
                        help="Tuning family. 'loggauss'/'gaussian' = "
                             "per-voxel parametric; 'weighted' = basis-set "
                             "log-Gaussian (matched to vonmises gabor decoder); "
                             "'linear' = signed-slope + baseline, closed-form "
                             "OLS (no tuning bump — see LinearValuePRF).")
    parser.add_argument('--n-basis', type=int, default=8,
                        help='[weighted only] number of basis RFs (default: 8)')
    parser.add_argument('--basis-fwhm', type=float, default=None,
                        help='[weighted only] basis fwhm in CHF '
                             '(default: 2 × inter-basis spacing)')
    parser.add_argument('--weight-alpha', type=float, default=0.0,
                        help='[weighted only] ridge α for WeightFitter (default: 0)')
    parser.add_argument('--debug', action='store_true',
                        help='100 iterations each (fast test)')
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions, n_voxels=args.n_voxels,
         fdr_alpha=args.fdr_alpha,
         p_signal_thr=args.p_signal_thr,
         fdr_fallback_n_voxels=args.fdr_fallback_n_voxels,
         null_gate=args.null_gate,
         n_iterations=args.n_iterations, n_stimulus_grid=args.n_stimulus_grid,
         n_basis=args.n_basis, basis_fwhm=args.basis_fwhm,
         weight_alpha=args.weight_alpha,
         lambd=args.lambd, mask=args.mask, mask_desc=args.mask_desc,
         spherical_noise=args.spherical_noise,
         geodesic_noise=args.geodesic_noise, geodesic_hemi=args.geodesic_hemi,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed, debug=args.debug, model_type=args.model)
