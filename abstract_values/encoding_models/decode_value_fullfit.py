#!/usr/bin/env python3
"""
Bayesian decoding of abstract value (CHF) using the already-fit, full-data
(non-cross-validated) encoding model — decode every trial in one pass.

Overview
--------
Unlike decode_value.py (leave-one-run-out: encoding model refit per fold,
so each trial is decoded by a model that never saw it), this script:

  1. Loads the whole-brain parameter volumes already written by
     fit_aprf.py --model {standard,linear,session-shift} (no refitting —
     'standard'/'linear' don't need ParameterFitter at all here, and
     'session-shift' doesn't need it either since its params are loaded
     straight from derivatives/encoding_models/aprf-session-shift/).
  2. Selects voxels by that full-fit model's own R² (top-N within mask).
  3. Fits ONE Student-t noise model on ALL trials.
  4. Decodes ALL trials (posterior over CHF value) with that single,
     fixed, full-data model — every trial's own run contributed to the
     model that decodes it. This is circular/optimistic by design (it
     answers "what does the model think happened", not "how well does
     this generalize" — decode_value.py's LORO decoding is for that) but
     gives one decoded value per trial using the maximum available data
     and signal, e.g. for correlating decoded value/uncertainty against
     trial-by-trial behaviour.

Models
------
  loggauss      : standard aPRF (derivatives/encoding_models/aprf) —
                  mode, fwhm, amplitude, baseline.
  linear        : linear aPRF (derivatives/encoding_models/aprf-linear) —
                  amplitude (signed slope), baseline.
  session-shift : only the preferred value (mode) shifts between sessions;
                  fwhm/amplitude/baseline shared
                  (derivatives/encoding_models/aprf-session-shift) —
                  mode_1, mode_2, fwhm, amplitude, baseline. Decoding
                  splits trials by session so each is evaluated against
                  its own session's mode.

Output
------
  derivatives/decoding/value-fullfit/<model>/sub-<subject>/func/
    sub-<subject>_mask-<mask_desc>_nvoxels-<n>_noise-<spherical|full>[_smoothed]_pars.tsv

  One row per trial (ALL trials, not just held-out), columns = value grid (CHF).
  Row index: (session, run, trial_nr, true_value_chf).

Usage
-----
  python decode_value_fullfit.py pil01 --model loggauss --mask ... --mask-desc BensonV1
  python decode_value_fullfit.py pil01 --model linear --n-voxels 0   # all voxels in mask
  python decode_value_fullfit.py pil01 --model session-shift --spherical-noise
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import LogGaussianPRF
from braincoder.optimize import ResidualFitter
from braincoder.utils import get_rsq

from abstract_values.encoding_models.models import (
    LinearValuePRF, SessionShiftedLogGaussianPRF,
)
from abstract_values.utils.data import Subject, BIDS_FOLDER

MODEL_SPECS = {
    'loggauss': dict(
        cls=LogGaussianPRF, cls_kwargs={'parameterisation': 'mode_fwhm_natural'},
        enc_dir='aprf', params=['mode', 'fwhm', 'amplitude', 'baseline'],
        needs_session=False),
    'linear': dict(
        cls=LinearValuePRF, cls_kwargs={},
        enc_dir='aprf-linear', params=['amplitude', 'baseline'],
        needs_session=False),
    'session-shift': dict(
        cls=SessionShiftedLogGaussianPRF, cls_kwargs={},
        enc_dir='aprf-session-shift',
        params=['mode_1', 'mode_2', 'fwhm', 'amplitude', 'baseline'],
        needs_session=True),
}


def get_value_paradigm(sub, sessions, needs_session):
    """DataFrame indexed by (session, run, trial_nr), column 'x' (+ 'session'
    as 0-based float index if needs_session), in the same order as the gabor
    betas (session -> run -> events sorted by onset)."""
    rows = []
    for ses_idx, session in enumerate(sorted(sessions)):
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
                    'session_idx': np.float32(ses_idx),
                })
    df = pd.DataFrame(rows).set_index(['session', 'run', 'trial_nr'])
    if needs_session:
        return df[['x', 'session_idx']].rename(columns={'session_idx': 'session'})
    return df[['x']]


def load_full_fit_params(subject, enc_dir, params, masker, bids_folder, smoothed=False):
    """Load pre-fit whole-brain parameter (+ r2) volumes, transform through
    `masker` so rows line up with `masker`'s voxel order. Returns
    (pars_df, r2_series) both indexed 0..n_voxels-1 matching masker order."""
    smooth_label = '_smoothed' if smoothed else ''
    base = (Path(bids_folder) / 'derivatives' / 'encoding_models' / enc_dir
            / f'sub-{subject}' / 'func')
    fn = (f'sub-{subject}_task-abstractvalue_space-T1w_desc-{{desc}}'
          f'{smooth_label}_pe.nii.gz')
    cols = {}
    for p in params:
        img = nib.load(base / fn.format(desc=p))
        cols[p] = masker.transform(img).astype(np.float32)
    pars = pd.DataFrame(cols)
    r2 = pd.Series(masker.transform(nib.load(base / fn.format(desc='r2'))),
                   name='r2')
    return pars, r2


def main(subject, sessions=None, n_voxels=100, model_type='loggauss',
         lambd=0.0, mask=None, mask_desc=None, spherical_noise=False,
         n_stimulus_grid=50, bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         smoothed=False, debug=False):
    spec = MODEL_SPECS[model_type]

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)
    if sessions is None:
        sessions = sub.get_sessions()
    if spec['needs_session'] and len(sessions) < 2:
        raise ValueError(f"--model {model_type} requires >=2 sessions")

    print(f'sub-{subject}  all-sessions ({sessions})  '
          f'[full-fit value decoding  model={model_type}]')

    paradigm = get_value_paradigm(sub, sessions, spec['needs_session'])
    value_min = float(paradigm['x'].min())
    value_max = float(paradigm['x'].max())
    print(f'  {len(paradigm)} trials  value range: {value_min:.1f}-{value_max:.1f} CHF')

    betas_img = sub.get_single_trial_estimates(sessions, desc='gabor', smoothed=smoothed)
    assert betas_img.shape[3] == len(paradigm)

    if mask is None:
        raise ValueError('Please provide --mask and --mask-desc')
    masker = NiftiMasker(mask_img=mask, target_affine=betas_img.affine,
                         target_shape=betas_img.shape[:3]).fit()
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32),
                        index=paradigm.index)
    print(f'  {data.shape[1]} voxels in mask ({mask_desc})')

    # ── load pre-fit params (no refitting) ──────────────────────────────────
    pars_all, r2_all = load_full_fit_params(
        subject, spec['enc_dir'], spec['params'], masker, bids_folder, smoothed)

    # ── voxel selection: top-N by the full-fit model's own R² ──────────────
    # (n_voxels=0 means "use every voxel in the mask" here — there is no
    # cross-validated R² available in this fit-on-everything mode.)
    if n_voxels == 0:
        sel = r2_all.index
        print(f'  {len(sel)} voxels selected (all voxels in mask)')
    else:
        sel = r2_all.sort_values(ascending=False).index[:n_voxels]
        print(f'  {len(sel)} voxels selected (full-fit R² >= {r2_all.loc[sel].min():.3f})')

    pars_sel = pars_all.loc[sel]
    data_sel = data[sel]

    model = spec['cls'](**spec['cls_kwargs'])

    # ── fit ONE noise model on all trials ───────────────────────────────────
    n_iter_noise = 100 if debug else 5000
    fit_paradigm = paradigm[['x', 'session']] if spec['needs_session'] else paradigm[['x']]
    residfit = ResidualFitter(model, data_sel, fit_paradigm,
                              parameters=pars_sel, lambd=lambd)
    omega, dof = residfit.fit(
        init_sigma2=0.1, init_dof=10.0, method='t',
        learning_rate=0.05, spherical=spherical_noise,
        max_n_iterations=n_iter_noise)
    print(f'  noise model: dof={float(dof):.1f}')

    # ── stimulus grid ────────────────────────────────────────────────────────
    stimulus_range = np.linspace(value_min, value_max, n_stimulus_grid, dtype=np.float32)

    # ── decode every trial ──────────────────────────────────────────────────
    if not spec['needs_session']:
        pdf = model.get_stimulus_pdf(data_sel, stimulus_range,
                                     parameters=pars_sel, omega=omega, dof=dof,
                                     normalize=False)
        pdf.columns = stimulus_range
    else:
        # session-shift: each trial's mode depends on its (known) session, so
        # decode the two sessions' trials separately against a grid that
        # fixes the session column to match, then relabel columns back to
        # plain CHF value afterward (same post-hoc-relabel pattern used for
        # decode_gabor.py's circular-basis decoding).
        pdfs = []
        for ses_idx in sorted(paradigm['session'].unique()):
            ses_mask = paradigm['session'] == ses_idx
            grid = np.stack([stimulus_range,
                             np.full_like(stimulus_range, ses_idx)], axis=1)
            p = model.get_stimulus_pdf(data_sel.loc[ses_mask], grid,
                                       parameters=pars_sel, omega=omega, dof=dof,
                                       normalize=False)
            p.columns = stimulus_range
            pdfs.append(p)
        pdf = pd.concat(pdfs)

    pdf.index = pd.MultiIndex.from_arrays([
        paradigm.index.get_level_values('session'),
        paradigm.index.get_level_values('run'),
        paradigm.index.get_level_values('trial_nr'),
        paradigm['x'].values,
    ], names=['session', 'run', 'trial_nr', 'true_value_chf'])
    pdf = pdf.sort_index()

    pred = model.predict(parameters=pars_sel, paradigm=fit_paradigm)
    r2_check = get_rsq(data_sel, pred)
    print(f'  full-fit R² of selected voxels (sanity check): mean={float(r2_check.mean()):.3f}')

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = (bids_folder / 'derivatives' / 'decoding' / 'value-fullfit' / model_type
               / f'sub-{subject}' / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_label = 'spherical' if spherical_noise else 'full'
    smooth_label = '_smoothed' if smoothed else ''
    lambd_label = f'_lambda-{lambd}' if lambd != 0.0 else ''
    out_fn = (out_dir /
              f'sub-{subject}_mask-{mask_desc}_nvoxels-{n_voxels}'
              f'_noise-{noise_label}{smooth_label}{lambd_label}_pars.tsv')
    pdf.to_csv(out_fn, sep='\t')
    print(f'\n  saved to {out_fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-'")
    parser.add_argument('--sessions', type=int, nargs='+', default=None)
    parser.add_argument('--model', default='loggauss',
                        choices=list(MODEL_SPECS.keys()),
                        help="'loggauss' (standard aPRF), 'linear', or "
                             "'session-shift' (only mode shifts by session).")
    parser.add_argument('--n-voxels', type=int, default=100,
                        help='Top-N voxels by the full-fit model R² '
                             '(0 = all voxels in mask)')
    parser.add_argument('--n-stimulus-grid', type=int, default=50)
    parser.add_argument('--lambd', type=float, default=0.0,
                        help='Lambda regularization for noise model (default: 0)')
    parser.add_argument('--mask', default=None, required=True)
    parser.add_argument('--mask-desc', default=None, required=True)
    parser.add_argument('--spherical-noise', action='store_true')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                        choices=['fmriprep', 'fmriprep-t2w'])
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions, n_voxels=args.n_voxels,
         model_type=args.model, lambd=args.lambd, mask=args.mask,
         mask_desc=args.mask_desc, spherical_noise=args.spherical_noise,
         n_stimulus_grid=args.n_stimulus_grid, bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv, smoothed=args.smoothed,
         debug=args.debug)
