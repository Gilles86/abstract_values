"""Value or quantile code? lambda fitted directly to the betas, shared across a ROI.

The neural_priors approach applied to the value/quantile question: one braincoder
model (``models.MixLambdaLogGaussianPRF``, log-Gaussian tuning on the mix axis
x_lambda = (1 - lambda) v + lambda (2 + 40 F_session(v))) fitted to the
single-trial betas of all voxels of a ROI at once, with per-voxel mode, fwhm,
amplitude and baseline and ONE lambda for the ROI (``shared_pars=['lam']``,
gradient descent). Compared with the same model at fixed lambda = 0 (value
code), 0.5 and 1 (quantile code) by cross-validated R^2.

Per (participant, ROI), on run-demeaned betas of all ROI voxels:
  - fixed lambda in {0, 0.5, 1}: grid search (correlation cost) + amplitude /
    baseline refinement + Adam with lam fixed;
  - shared lambda: Adam from the best (training R^2) of the three fixed fits,
    lambda free but tied across voxels;
  - 8 folds: hold out run k of both sessions; cvR^2 per voxel = 1 - SSE / SS
    over the held-out trials, pooled over folds; plus the full-data shared
    lambda and the per-fold shared lambdas.

With --sim-lam / --noise-scale: the same on betas simulated with a known lambda
(``simulate_lambda_pipeline.simulated_betas``), for recovery.

    python -m abstract_values.encoding_models.fit_mix_lambda <subject> [--sim-lam 0 --noise-scale 2]
    -> derivatives/encoding_models/mix_lambda/sub-<S>/sub-<S>{_sim-lam<l>-noise<k>}{_width}_mixlambda.tsv

--model width: position and width follow the quantile code separately (lam_mu, lam_sigma,
``models.MixPositionWidthGaussianPRF``; Gaussian tuning), both shared across the ROI;
fixed corners (lam_mu, lam_sigma) in {(0,0), (1,0), (0,1), (1,1), (.5,.5)}. The
simulations (simulate_lambda_pipeline) move only the preferred value, so their lambda = 1
is (lam_mu, lam_sigma) = (1, 0).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from braincoder.optimize import ParameterFitter

from abstract_values.connective_fields.gates import demean_runs, get_paradigm, load_roi
from abstract_values.encoding_models.model_specs import value_quantiles
from abstract_values.encoding_models.models import (MixLambdaLogGaussianPRF,
                                                     MixPositionWidthGaussianPRF)
from abstract_values.utils.data import BIDS_FOLDER, Subject

TARGETS = {'V1': 'BensonV1ecc075-375', 'IPS0': 'IPS0', 'IPS12': 'IPS12', 'IPS345': 'IPS345',
           'V3ab': 'V3ab', 'hV4': 'hV4'}
FIXED = (0., .5, 1.)
# --model width: (lam_mu, lam_sigma) fixed corners; shared fit starts from the best
FIXED_W = ((0., 0.), (1., 0.), (0., 1.), (1., 1.), (.5, .5))
SHARED = {'mix': ['lam'], 'width': ['lam_mu', 'lam_sigma']}
MODES = np.linspace(2, 42, 15).astype(np.float32)
FWHMS = np.linspace(2, 40, 10).astype(np.float32)


def fit(Y, par, n_iter, lam=None, init=None, kind='mix'):
    """lam given (a number, or (lam_mu, lam_sigma) for kind 'width'): fixed fit
    (grid + Adam). Else: the shared mixing parameter(s), from ``init``."""
    model = MixLambdaLogGaussianPRF() if kind == 'mix' else MixPositionWidthGaussianPRF()
    f = ParameterFitter(model, Y, par)
    if lam is not None:
        lams = [np.array([l], np.float32) for l in np.atleast_1d(lam)]
        g = f.fit_grid(MODES, FWHMS, np.array([1.], np.float32), np.array([0.], np.float32),
                       *lams, use_correlation_cost=True)
        g = f.refine_baseline_and_amplitude(g)
        p = f.fit(max_n_iterations=n_iter, init_pars=g, fixed_pars=SHARED[kind],
                  progressbar=False)
    else:
        p = f.fit(max_n_iterations=n_iter, init_pars=init, shared_pars=SHARED[kind],
                  progressbar=False)
    return p, model


def fixed_name(l):
    return 'fixed' + '_'.join(f'{x:g}' for x in np.atleast_1d(l))


def fit_all(Y, par, n_iter, kind='mix'):
    """Fixed fits (FIXED or FIXED_W) and the shared fit; {name: (pars, model)}."""
    out = {fixed_name(l): fit(Y, par, n_iter, lam=l, kind=kind)
           for l in (FIXED if kind == 'mix' else FIXED_W)}

    def r2(name):
        p, m = out[name]
        pred = m.predict(parameters=p, paradigm=par).values
        return 1 - ((Y.values - pred) ** 2).sum() / ((Y.values - Y.values.mean(0)) ** 2).sum()
    best = max(out, key=r2)
    out['shared'] = fit(Y, par, n_iter, init=out[best][0], kind=kind)
    out['shared_init'] = best
    return out


def main(subject, bids_folder=BIDS_FOLDER, targets=tuple(TARGETS), n_iter=500,
         sim_lam=None, noise_scale=1., kind='mix', sim_corner=None):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    par_all = get_paradigm(sub, sessions)
    if sim_lam is None and sim_corner is None:
        betas = sub.get_single_trial_estimates(sessions, desc='gabor', smoothed=False)
    else:
        from abstract_values.encoding_models.simulate_lambda_pipeline import simulated_betas
        betas = simulated_betas(subject, sim_lam, bids_folder, noise_scale=noise_scale,
                                corner=sim_corner)
    v = par_all['value'].to_numpy()
    ses = par_all['session'].to_numpy()
    par = pd.DataFrame({'x': v, 'q': 2 + 40 * value_quantiles(v, ses)})
    if kind == 'width':
        par['m'] = (par_all['condition'] != 'cdf').astype(float).to_numpy()
    par = par.astype(np.float32)
    shared = SHARED[kind]
    runs = np.sort(par_all['run'].unique())
    folds = [(par_all['run'] == r).to_numpy() for r in runs]      # run k of both sessions
    rows = []
    for roi in targets:
        try:
            y, _ = load_roi(sub, betas, TARGETS[roi], 'aprf.cv', bids_folder, False,
                            selection='all')
        except Exception as e:
            print(f'  {roi}: skipped ({e})')
            continue
        Y = pd.DataFrame(demean_runs(y, par_all).astype(np.float32))
        full = fit_all(Y, par, n_iter, kind)
        names = [fixed_name(l) for l in (FIXED if kind == 'mix' else FIXED_W)] + ['shared']
        sse = {n: np.zeros(Y.shape[1]) for n in names}
        sst = np.zeros(Y.shape[1])
        fold_lam = {k: [] for k in shared}
        for k, te in enumerate(folds):
            tr = ~te
            m = fit_all(Y[tr].reset_index(drop=True), par[tr].reset_index(drop=True), n_iter, kind)
            yt = Y.values[te]
            sst += ((yt - yt.mean(0)) ** 2).sum(0)
            for n in names:
                p, model = m[n]
                pred = model.predict(parameters=p, paradigm=par[te].reset_index(drop=True)).values
                sse[n] += ((yt - pred) ** 2).sum(0)
            for key in shared:
                fold_lam[key].append(float(m['shared'][0][key].iloc[0]))
            print(f'  {roi} fold {k + 1}/{len(folds)}: shared '
                  + ', '.join(f'{key} {fold_lam[key][-1]:+.2f}' for key in shared), flush=True)
        cv = {n: 1 - sse[n] / sst for n in names}
        row = dict(roi=roi, n_voxels=Y.shape[1], shared_init=full['shared_init'])
        for key in shared:
            row.update({f'{key}_shared': float(full['shared'][0][key].iloc[0]),
                        f'{key}_fold_mean': np.mean(fold_lam[key]),
                        f'{key}_fold_sd': np.std(fold_lam[key]),
                        f'{key}_folds': ' '.join(f'{x:.3f}' for x in fold_lam[key])})
        for n in names:
            row[f'cvr2_{n}'] = float(np.mean(cv[n]))
            row[f'cvr2pooled_{n}'] = float(1 - sse[n].sum() / sst.sum())
        best_q, best_v = (('fixed1', 'fixed0') if kind == 'mix' else ('fixed1_1', 'fixed0_0'))
        row['frac_vox_quantile_better'] = float(np.mean(cv[best_q] > cv[best_v]))
        rows.append(row)
        print(f'  {roi}: {Y.shape[1]} voxels; shared '
              + ', '.join(f'{key} {row[f"{key}_shared"]:+.2f} (folds {row[f"{key}_fold_mean"]:+.2f}'
                          f' ± {row[f"{key}_fold_sd"]:.2f})' for key in shared)
              + '; cvR2 x1e3 ' + ', '.join(f'{n} {1e3 * row[f"cvr2_{n}"]:+.2f}' for n in names),
              flush=True)
    tag = '' if sim_lam is None else f'_sim-lam{sim_lam:g}-noise{noise_scale:g}'
    if sim_corner is not None:
        tag = f'_sim-corner{sim_corner[0]:g}_{sim_corner[1]:g}-noise{noise_scale:g}'
    tag += '_width' if kind == 'width' else ''
    out = bids_folder / 'derivatives' / 'encoding_models' / 'mix_lambda' / f'sub-{subject}'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).assign(subject=subject, sim_lam=sim_lam, noise_scale=noise_scale,
                              sim_corner=None if sim_corner is None
                              else f'{sim_corner[0]:g}_{sim_corner[1]:g}').to_csv(
        out / f'sub-{subject}{tag}_mixlambda.tsv', sep='\t', index=False)
    print(f'saved to {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    ap.add_argument('--targets', nargs='+', default=list(TARGETS))
    ap.add_argument('--n-iterations', type=int, default=500)
    ap.add_argument('--sim-lam', type=float, default=None)
    ap.add_argument('--noise-scale', type=float, default=1.)
    ap.add_argument('--model', default='mix', choices=['mix', 'width'],
                    help="'mix': one lambda (log-Gaussian on the mix axis); 'width': separate "
                         "lam_mu (position) and lam_sigma (width) (MixPositionWidthGaussianPRF)")
    ap.add_argument('--sim-corner', type=float, nargs=2, default=None,
                    metavar=('LAM_MU', 'LAM_SIGMA'),
                    help='betas simulated from the position/width model with these values')
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.targets, a.n_iterations, a.sim_lam, a.noise_scale, a.model,
         a.sim_corner)
