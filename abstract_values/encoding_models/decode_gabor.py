#!/usr/bin/env python3
"""
Bayesian decoding of gabor orientation from single-trial fMRI amplitudes.

Overview
--------
Leave-one-run-out cross-validation.  In each fold:

  1. Fit a Von Mises basis-set encoding model on the training runs
     (WeightFitter — closed-form lstsq; basis mus/kappa are fixed).
  2. Select voxels: top n_voxels by training R², or (when n_voxels=0) all
     voxels with nested cross-validated R² > 0 (inner leave-one-run-out CV
     within the training set — no circularity).
  3. Fit a multivariate Student-t residual noise model (ResidualFitter) on
     the training set to get a noise covariance omega and degrees of freedom.
  4. Evaluate P(data | orientation) over all presented orientations for each
     test trial via model.get_stimulus_pdf().  This unnormalised likelihood
     serves as the posterior PDF under a flat prior.

The model is fitted only inside the supplied mask (default: fmriprep brain
mask; pass --mask with a V1/V2 ROI for a smaller, targeted analysis).

Output
------
  derivatives/decoding/gabor/sub-<subject>/<ses_label>/func/
    sub-<subject>_<ses_label>_mask-<mask_desc>_nvoxels-<n>_pars.tsv

  One row per test trial, columns = decoded orientation grid (radians).
  Row index: (session, run, trial_nr, true_orientation_rad).

Usage
-----
  python decode_gabor.py pil01 --sessions 1
  python decode_gabor.py pil01 --sessions 1 --n-voxels 200
  python decode_gabor.py pil01 --sessions 1 --n-voxels 0   # all R²>0
  python decode_gabor.py pil01 --sessions 1 \\
      --mask /data/ds-abstractvalue/derivatives/masks/sub-pil01/ses-1/anat/\\
             sub-pil01_ses-1_space-T1w_hemi-LR_desc-BensonV1_mask.nii.gz \\
      --mask-desc BensonV1
  python decode_gabor.py pil01 --sessions 1 --debug
  python decode_gabor.py pil01 --sessions 1 --geodesic-noise --geodesic-hemi R
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import (AxialVonMisesPRF, LinearModelWithBaseline,
                               LogGaussianPRF)
from braincoder.optimize import WeightFitter, ResidualFitter
from braincoder.utils import get_rsq

from abstract_values.utils.data import Subject, BIDS_FOLDER
from abstract_values.encoding_models.geodesic_noise import (
    geodesic_snap_for_masker, geodesic_D_for_selection,
)
from abstract_values.encoding_models.voxel_selection import (
    STATUS_EMPTY, STATUS_OK, concat_posteriors, select_voxels, warn_if_degraded,
)


def get_gabor_paradigm(sub, sessions):
    """Return DataFrame indexed by (session, run, trial_nr) with column 'x'.

    x = gabor orientation in radians, in the same order as the gabor betas
    written by fit_glmsingle (session → run → events sorted by onset).
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
                    'x':        np.float32(np.deg2rad(float(row['orientation']))),
                })
    df = pd.DataFrame(rows).set_index(['session', 'run', 'trial_nr'])
    return df[['x']]


def make_basis_parameters(n_basis, kappa):
    """Fixed Von Mises basis parameters (amplitude=1, baseline=0)."""
    mus = np.linspace(0, np.pi, n_basis, endpoint=False).astype(np.float32)
    return pd.DataFrame({
        'mu':        mus,
        'kappa':     np.full(n_basis, kappa, dtype=np.float32),
        'amplitude': np.ones(n_basis,  dtype=np.float32),
        'baseline':  np.zeros(n_basis, dtype=np.float32),
    })


def _out_subdir(model_type):
    return {'vonmises': 'gabor', 'linear': 'gabor-linear'}[model_type]


def _circular_basis(x_rad):
    """[cos(2x), sin(2x)] — the axial (pi-periodic) first-harmonic basis.

    Doubling the angle matches AxialVonMisesPRF's own axial convention
    (period pi, not 2pi — a gabor at theta looks identical to theta+pi).
    A model linear in this 2-column basis is the smoothest possible
    orientation-selective response: one broad preferred/anti-preferred
    orientation, no sharper tuning than that. It's the circular analogue
    of LinearValuePRF's straight ramp — no tuning bump, just a single
    graded direction of preference.
    """
    return pd.DataFrame({'cos2x': np.cos(2 * x_rad), 'sin2x': np.sin(2 * x_rad)},
                        index=x_rad.index if hasattr(x_rad, 'index') else None)


#: The deployed value basis (fit_aprf_weighted's defaults) as the rival model.
RIVAL_N_BASIS, RIVAL_ALPHA = 8, 10.0


def _rival_value_cv_r2(train_data, train_val, value_min, value_max,
                       n_basis=RIVAL_N_BASIS, alpha=RIVAL_ALPHA):
    """Nested CV R2 of the log-Gaussian VALUE basis, on the same inner folds.

    The mirror of decode_value's orientation rival: a voxel is only evidence
    for orientation coding if its orientation model beats a value model given
    the same held-out runs. Within a session the two stimuli are deterministic
    functions of each other, so "beats the null" is satisfied by either.

    Imported lazily: decode_value imports this module for the orientation
    paradigm, so a module-level import here would close the cycle.
    """
    from abstract_values.encoding_models.decode_value import (
        make_value_basis_parameters)

    model = LogGaussianPRF(parameterisation='mode_fwhm_natural')
    basis_pars = make_value_basis_parameters(n_basis, value_min, value_max)
    sess = train_val.index.get_level_values('session')
    runs = train_val.index.get_level_values('run')
    r2s = []
    for inner_ses, inner_run in sorted(set(zip(sess, runs))):
        te = (sess == inner_ses) & (runs == inner_run)
        w = WeightFitter(model, basis_pars, train_data.loc[~te],
                         train_val.loc[~te]).fit(alpha=alpha)
        bp = model.basis_predictions(train_val.loc[te], basis_pars)
        pred = pd.DataFrame(bp @ w.values, index=train_data.loc[te].index,
                            columns=train_data.columns)
        r2s.append(get_rsq(train_data.loc[te], pred))
    return pd.concat(r2s, axis=1).mean(axis=1)


def _run_linear_folds(sub, sessions, paradigm, data, stimulus_range,
                      n_voxels, fdr_alpha, p_signal_thr,
                      fdr_fallback_n_voxels,
                      weight_alpha, lambd, spherical_noise,
                      smoothed, bids_folder, debug, rival_val=None,
                      rival_val_range=None):
    """Linear (no tuning bump) orientation decoding — the circular analog
    of decode_value.py's ``_run_linear_folds``. Signed response along a
    single graded orientation axis: slope on [cos(2x), sin(2x)] + baseline,
    fit jointly in closed form via WeightFitter(fit_intercept=True).
    Mirrors the Von Mises flow above fold-for-fold; only the basis and
    stimulus-grid construction differ.
    """
    all_pdfs = []
    fold_meta = []
    all_runs = [(s, r) for s in sessions for r in sub.get_runs(s)]

    # 2-column [cos(2x), sin(2x)] grid matching stimulus_range, used for
    # decoding; the model itself only ever sees this circular basis, never
    # raw radians.
    grid_basis = _circular_basis(pd.Series(stimulus_range))

    def _fit_slope_baseline(train_x, train_d):
        basis = _circular_basis(train_x['x'])
        m = LinearModelWithBaseline(paradigm=basis, parameters=None)
        w, b = WeightFitter(m, None, train_d, basis).fit(
            alpha=weight_alpha, fit_intercept=True)
        return m, w, pd.DataFrame({'baseline': b})

    def _predict(m, x, w, b_df):
        basis = _circular_basis(x['x'])
        pred = m.predict(paradigm=basis, parameters=b_df, weights=w)
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

        # ── fit slope + baseline (closed-form OLS on the circular basis) ───
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
            cv_r2_rival = (_rival_value_cv_r2(
                train_data, rival_val.loc[train_idx], *rival_val_range)
                if rival_val is not None else None)
            sel, status, msg = select_voxels(
                cv_r2, mixture_model='vonmises-linear',
                subject=sub.subject_id, bids_folder=bids_folder,
                smoothed=smoothed, fdr_alpha=fdr_alpha,
                p_signal_thr=p_signal_thr,
                fdr_fallback_n_voxels=fdr_fallback_n_voxels,
                cv_r2_rival=cv_r2_rival)
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
            # Nothing generalises in this fold. Record it and move on — the
            # fold contributes no trials rather than a posterior decoded from
            # voxels the encoding model does not explain.
            continue
        weights_sel    = weights[sel]
        baseline_sel   = baseline_df.loc[sel]
        train_data_sel = train_data[sel]
        test_data_sel  = test_data[sel]

        # ── fit noise model ────────────────────────────────────────────────
        n_iter_noise = 100 if debug else 1000
        train_basis = _circular_basis(train_paradigm['x'])
        residfit = ResidualFitter(model, train_data_sel, train_basis,
                                  parameters=baseline_sel, weights=weights_sel,
                                  lambd=lambd)
        omega, dof = residfit.fit(
            init_sigma2=0.1, init_dof=10.0, method='t',
            learning_rate=0.05, spherical=spherical_noise,
            max_n_iterations=n_iter_noise)
        print(f'    noise model: dof={float(dof):.1f}')

        # ── decode ─────────────────────────────────────────────────────────
        pdf = model.get_stimulus_pdf(test_data_sel, grid_basis.values,
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
        ], names=['session', 'run', 'trial_nr', 'true_orientation_rad'])

        all_pdfs.append(pdf)

    return all_pdfs, fold_meta


def main(subject, sessions=None, n_voxels=100, fdr_alpha=None,
         p_signal_thr=None, fdr_fallback_n_voxels=100,
         n_basis=8, kappa=2.0,
         weight_alpha=0.0, lambd=0.0,
         mask=None, mask_desc=None, spherical_noise=False,
         geodesic_noise=False, geodesic_hemi='R',
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         smoothed=False, debug=False, model_type='vonmises',
         rival_value=False):
    """If fdr_alpha is set, voxels are selected by FDR-thresholding the
    nested-CV R² using the whole-brain vonmises mixture.
    ``fdr_fallback_n_voxels`` is the top-N fallback when the mixture
    is flagged degenerate. Output filename uses ``nvoxels-fdrNN``.

    If p_signal_thr is set instead, voxels are selected by thresholding
    nested-CV R² at P(signal | r²) ≥ p_signal_thr (same whole-brain
    vonmises mixture). Same degenerate-fallback semantics. Output
    filename uses ``nvoxels-psigNN``. ``fdr_alpha`` and ``p_signal_thr``
    are mutually exclusive.

    ``model_type``: 'vonmises' (default, tuned bump — 8-function Von Mises
    basis set) or 'linear' (no tuning bump — signed response along a
    single graded orientation axis, cos(2x)/sin(2x) basis, closed-form
    fit). Output subdir: derivatives/decoding/{gabor,gabor-linear}/."""

    assert not (fdr_alpha is not None and p_signal_thr is not None), \
        "Pass at most one of --fdr-alpha / --p-signal-thr"

    # braincoder's ResidualFitter.get_omega() checks lambd>0 BEFORE D: when
    # both are set it silently routes to the sample-covariance shrinkage
    # Omega and never touches alpha/beta/D at all (confirmed empirically —
    # alpha/beta sit frozen at their init values, "Gradients do not exist"
    # UserWarning). Output would still say noise-geodesic while actually
    # being non-geodesic. Fail loudly instead of writing a misleadingly
    # labeled file.
    assert not (geodesic_noise and lambd > 0.0), \
        ('--geodesic-noise and --lambd > 0 are incompatible in the current '
         'braincoder ResidualFitter (lambd>0 silently overrides the '
         'geodesic Omega). Pass --lambd 0 with --geodesic-noise.')

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if sessions is None:
        sessions = sub.get_sessions()

    ses_dir    = f'ses-{sessions[0]}' if len(sessions) == 1 else ''
    ses_entity = f'_ses-{sessions[0]}' if len(sessions) == 1 else ''
    print(f'sub-{subject}  {ses_dir or "all-sessions"}  '
          f'[gabor orientation decoding  model={model_type}]')

    # ── paradigm + data ───────────────────────────────────────────────────────
    paradigm = get_gabor_paradigm(sub, sessions)
    # Selection rival: the same trials in value space (lazy import — see
    # _rival_value_cv_r2). Only meaningful with the nested-CV floor.
    rival_val = rival_val_range = None
    if rival_value:
        if n_voxels != 0:
            raise SystemExit(
                '--rival-value needs --n-voxels 0 (the nested-CV floor); '
                'top-N by training R2 has no per-voxel cv_r2 to compare.')
        from abstract_values.encoding_models.decode_value import (
            get_value_paradigm)
        rival_val = get_value_paradigm(sub, sessions)
        rival_val_range = (float(rival_val['x'].min()),
                           float(rival_val['x'].max()))
        print(f'  rival: log-Gaussian value basis (k={RIVAL_N_BASIS}, '
              f'alpha={RIVAL_ALPHA}, {rival_val_range[0]:.0f}-'
              f'{rival_val_range[1]:.0f} CHF) — keeping only voxels the '
              f'orientation model beats it on')
    print(f'  {len(paradigm)} trials')

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
    geo_snap = None
    if geodesic_noise:
        geo_snap = geodesic_snap_for_masker(
            masker, geodesic_hemi, subject, bids_folder, fmriprep_deriv)

    # ── stimulus grid ─────────────────────────────────────────────────────────
    stimulus_range = np.sort(paradigm['x'].unique()).astype(np.float32)
    print(f'  stimulus grid: {len(stimulus_range)} orientations '
          f'({np.rad2deg(stimulus_range[[0,-1]]).round(1)} deg)')

    # ── output dir ───────────────────────────────────────────────────────────
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
    else:
        nvox_tag = str(n_voxels)
    if rival_value:
        # Distinct tag: a winner-selected run answers a different question from
        # the null-gated one and must not overwrite it.
        nvox_tag = f'{nvox_tag}-vsval' 
    out_fn = (out_dir /
              f'sub-{subject}{ses_entity}_mask-{mask_desc}'
              f'_nvoxels-{nvox_tag}_noise-{noise_label}{smooth_label}{lambd_label}_pars.tsv')

    if model_type == 'linear':
        all_pdfs, fold_meta = _run_linear_folds(
            sub, sessions, paradigm, data, stimulus_range,
            n_voxels, fdr_alpha, p_signal_thr, fdr_fallback_n_voxels,
            weight_alpha, lambd, spherical_noise,
            smoothed, bids_folder, debug, rival_val=rival_val,
            rival_val_range=rival_val_range)
        warn_if_degraded(fold_meta, subject, mask_desc)
        pdfs = concat_posteriors(
            all_pdfs, stimulus_range,
            ['session', 'run', 'trial_nr', 'true_orientation_rad'])
        pdfs.to_csv(out_fn, sep='\t')
        print(f'\n  saved to {out_fn}')
        meta_fn = out_fn.with_name(out_fn.stem.replace('_pars', '_meta') + '.tsv')
        pd.DataFrame(fold_meta).to_csv(meta_fn, sep='\t', index=False)
        print(f'  meta  to {meta_fn}')
        return

    # ── basis parameters ──────────────────────────────────────────────────────
    basis_pars = make_basis_parameters(n_basis, kappa)
    print(f'  {n_basis} Von Mises basis functions  kappa={kappa}')

    # ── leave-one-run-out cross-validation ────────────────────────────────────
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

        # ── fit basis weights (ridge regression) ──────────────────────────────
        model = AxialVonMisesPRF(allow_neg_amplitudes=True)
        weights = WeightFitter(model, basis_pars, train_data, train_paradigm).fit(alpha=weight_alpha)
        # weights: DataFrame (n_basis × n_voxels)

        # ── voxel selection ────────────────────────────────────────────────────
        if n_voxels == 0 or fdr_alpha is not None or p_signal_thr is not None:
            # Nested CV: leave-one-run-out within training set to get unbiased R²
            inner_runs = list(zip(
                train_paradigm.index.get_level_values('session'),
                train_paradigm.index.get_level_values('run')))
            inner_runs = sorted(set(inner_runs))

            inner_r2s = []
            for inner_ses, inner_run in inner_runs:
                inner_test_idx = (
                    (train_paradigm.index.get_level_values('session') == inner_ses) &
                    (train_paradigm.index.get_level_values('run') == inner_run))
                inner_train_paradigm = train_paradigm.loc[~inner_test_idx]
                inner_test_paradigm  = train_paradigm.loc[inner_test_idx]
                inner_train_data     = train_data.loc[~inner_test_idx]
                inner_test_data      = train_data.loc[inner_test_idx]

                inner_model = AxialVonMisesPRF(allow_neg_amplitudes=True)
                inner_w = WeightFitter(inner_model, basis_pars,
                                       inner_train_data, inner_train_paradigm).fit(alpha=weight_alpha)
                inner_bp = inner_model.basis_predictions(inner_test_paradigm, basis_pars)
                inner_pred = pd.DataFrame(inner_bp @ inner_w.values,
                                          index=inner_test_data.index,
                                          columns=inner_test_data.columns)
                inner_r2s.append(get_rsq(inner_test_data, inner_pred))

            cv_r2 = pd.concat(inner_r2s, axis=1).mean(axis=1)
            cv_r2_rival = (_rival_value_cv_r2(
                train_data, rival_val.loc[train_idx], *rival_val_range)
                if rival_val is not None else None)
            sel, status, msg = select_voxels(
                cv_r2, mixture_model='vonmises', subject=subject,
                bids_folder=bids_folder, smoothed=smoothed,
                fdr_alpha=fdr_alpha, p_signal_thr=p_signal_thr,
                fdr_fallback_n_voxels=fdr_fallback_n_voxels,
                cv_r2_rival=cv_r2_rival)
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
            # No voxel generalises in this fold. Previously this raised
            # SystemExit, which left Snakemake with no sentinel and re-planned
            # the job on every driver generation. Record the fold as
            # undecodable and carry on — the emptiness is now visible in the
            # _meta sidecar instead of in an exit code.
            continue
        weights_sel    = weights[sel]
        train_data_sel = train_data[sel]
        test_data_sel  = test_data[sel]

        # ── fit noise model ───────────────────────────────────────────────────
        n_iter_noise = 100 if debug else 1000
        residfit = ResidualFitter(model, train_data_sel, train_paradigm,
                                  parameters=basis_pars, weights=weights_sel,
                                  lambd=lambd)
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
        pdf = model.get_stimulus_pdf(test_data_sel, stimulus_range,
                                     parameters=basis_pars,
                                     weights=weights_sel,
                                     omega=omega, dof=dof,
                                     normalize=False)
        # pdf: DataFrame (n_test_trials, n_orientations)
        pdf.columns = stimulus_range
        pdf.index = pd.MultiIndex.from_arrays([
            test_paradigm.index.get_level_values('session'),
            test_paradigm.index.get_level_values('run'),
            test_paradigm.index.get_level_values('trial_nr'),
            test_paradigm['x'].values,
        ], names=['session', 'run', 'trial_nr', 'true_orientation_rad'])

        all_pdfs.append(pdf)

    # ── save ──────────────────────────────────────────────────────────────────
    warn_if_degraded(fold_meta, subject, mask_desc)
    pdfs = concat_posteriors(
        all_pdfs, stimulus_range,
        ['session', 'run', 'trial_nr', 'true_orientation_rad'])
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
    parser.add_argument('--rival-value', action='store_true',
                        help="Keep only voxels whose nested-CV R2 beats a "
                             "log-Gaussian VALUE basis fit on the same inner "
                             "folds -- 'voxels the orientation model wins'. "
                             "The mirror of decode_value's "
                             "--rival-orientation; requires --n-voxels 0.")
    parser.add_argument('--n-voxels', type=int, default=100,
                        help='Top-N voxels by training R² (0 = nested CV R²>0)')
    parser.add_argument('--fdr-alpha', type=float, default=None,
                        help='If set, voxels are selected by FDR-thresholding '
                             'nested-CV R² with the whole-brain vonmises '
                             'mixture (see compute_r2_mixture). Output: nvoxels-fdrNN.')
    parser.add_argument('--p-signal-thr', type=float, default=None,
                        help='If set, voxels are selected by thresholding '
                             'nested-CV R² at P(signal | r²) ≥ p (whole-brain '
                             'vonmises mixture). Mutually exclusive with '
                             '--fdr-alpha. Output: nvoxels-psigNN.')
    parser.add_argument('--fdr-fallback-n-voxels', type=int, default=100,
                        help='Top-N voxels by cv-R² to use when the whole-brain '
                             'mixture is flagged degenerate (default: 100). '
                             'Applies to both --fdr-alpha and --p-signal-thr.')
    parser.add_argument('--n-basis', type=int, default=8,
                        help='Number of Von Mises basis functions (default: 8)')
    parser.add_argument('--kappa', type=float, default=2.0,
                        help='Von Mises concentration (default: 2.0)')
    parser.add_argument('--weight-alpha', type=float, default=0.0,
                        help='Ridge regression alpha for weight fitting (default: 0)')
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
                             'distance (default: R, for NPCr/BensonV1 hemi-R)')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                        choices=['fmriprep', 'fmriprep-t2w'])
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help='100 noise iterations (fast test)')
    parser.add_argument('--model', default='vonmises',
                        choices=['vonmises', 'linear'],
                        help="Tuning family. 'vonmises' (default) = tuned "
                             "bump, 8-function Von Mises basis set; "
                             "'linear' = no tuning bump, signed response "
                             "on cos(2x)/sin(2x), closed-form fit. Output "
                             "subdir: derivatives/decoding/{gabor,gabor-linear}/.")
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions, n_voxels=args.n_voxels,
         fdr_alpha=args.fdr_alpha,
         p_signal_thr=args.p_signal_thr,
         fdr_fallback_n_voxels=args.fdr_fallback_n_voxels,
         n_basis=args.n_basis, kappa=args.kappa,
         weight_alpha=args.weight_alpha, lambd=args.lambd,
         mask=args.mask, mask_desc=args.mask_desc,
         spherical_noise=args.spherical_noise,
         geodesic_noise=args.geodesic_noise, geodesic_hemi=args.geodesic_hemi,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed, debug=args.debug, model_type=args.model,
         rival_value=args.rival_value)
