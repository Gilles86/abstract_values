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
    -> derivatives/encoding_models/mix_lambda/sub-<S>/sub-<S>{_sim-lam<l>-noise<k>}_mixlambda.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from braincoder.optimize import ParameterFitter

from abstract_values.connective_fields.gates import demean_runs, get_paradigm, load_roi
from abstract_values.encoding_models.model_specs import value_quantiles
from abstract_values.encoding_models.models import MixLambdaLogGaussianPRF
from abstract_values.utils.data import BIDS_FOLDER, Subject

TARGETS = {'V1': 'BensonV1ecc075-375', 'IPS0': 'IPS0', 'IPS12': 'IPS12', 'IPS345': 'IPS345',
           'V3ab': 'V3ab', 'hV4': 'hV4'}
FIXED = (0., .5, 1.)
MODES = np.linspace(2, 42, 15).astype(np.float32)
FWHMS = np.linspace(2, 40, 10).astype(np.float32)


def fit(Y, par, n_iter, lam=None, init=None):
    """lam given: fixed-lambda fit (grid + Adam). Else: shared lambda from ``init``."""
    model = MixLambdaLogGaussianPRF()
    f = ParameterFitter(model, Y, par)
    if lam is not None:
        g = f.fit_grid(MODES, FWHMS, np.array([1.], np.float32), np.array([0.], np.float32),
                       np.array([lam], np.float32), use_correlation_cost=True)
        g = f.refine_baseline_and_amplitude(g)
        p = f.fit(max_n_iterations=n_iter, init_pars=g, fixed_pars=['lam'], progressbar=False)
    else:
        p = f.fit(max_n_iterations=n_iter, init_pars=init, shared_pars=['lam'],
                  progressbar=False)
    return p, model


def fit_all(Y, par, n_iter):
    """Fixed fits at FIXED and the shared-lambda fit; {name: (pars, model)}."""
    out = {f'fixed{l:g}': fit(Y, par, n_iter, lam=l) for l in FIXED}

    def r2(name):
        p, m = out[name]
        pred = m.predict(parameters=p, paradigm=par).values
        return 1 - ((Y.values - pred) ** 2).sum() / ((Y.values - Y.values.mean(0)) ** 2).sum()
    best = max(out, key=r2)
    out['shared'] = fit(Y, par, n_iter, init=out[best][0])
    out['shared_init'] = best
    return out


def main(subject, bids_folder=BIDS_FOLDER, targets=tuple(TARGETS), n_iter=500,
         sim_lam=None, noise_scale=1.):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    par_all = get_paradigm(sub, sessions)
    if sim_lam is None:
        betas = sub.get_single_trial_estimates(sessions, desc='gabor', smoothed=False)
    else:
        from abstract_values.encoding_models.simulate_lambda_pipeline import simulated_betas
        betas = simulated_betas(subject, sim_lam, bids_folder, noise_scale=noise_scale)
    v = par_all['value'].to_numpy()
    par = pd.DataFrame({'x': v, 'q': 2 + 40 * value_quantiles(v, par_all['session'].to_numpy())}
                       ).astype(np.float32)
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
        full = fit_all(Y, par, n_iter)
        names = [f'fixed{l:g}' for l in FIXED] + ['shared']
        sse = {n: np.zeros(Y.shape[1]) for n in names}
        sst = np.zeros(Y.shape[1])
        fold_lam = []
        for k, te in enumerate(folds):
            tr = ~te
            m = fit_all(Y[tr].reset_index(drop=True), par[tr].reset_index(drop=True), n_iter)
            yt = Y.values[te]
            sst += ((yt - yt.mean(0)) ** 2).sum(0)
            for n in names:
                p, model = m[n]
                pred = model.predict(parameters=p, paradigm=par[te].reset_index(drop=True)).values
                sse[n] += ((yt - pred) ** 2).sum(0)
            fold_lam.append(float(m['shared'][0]['lam'].iloc[0]))
            print(f'  {roi} fold {k + 1}/{len(folds)}: shared lambda {fold_lam[-1]:+.2f}',
                  flush=True)
        cv = {n: 1 - sse[n] / sst for n in names}
        row = dict(roi=roi, n_voxels=Y.shape[1], lam_shared=float(full['shared'][0]['lam'].iloc[0]),
                   shared_init=full['shared_init'], lam_fold_mean=np.mean(fold_lam),
                   lam_fold_sd=np.std(fold_lam),
                   lam_folds=' '.join(f'{x:.3f}' for x in fold_lam))
        for n in names:
            row[f'cvr2_{n}'] = float(np.mean(cv[n]))
            row[f'cvr2pooled_{n}'] = float(1 - sse[n].sum() / sst.sum())
        row['frac_vox_quantile_better'] = float(np.mean(cv['fixed1'] > cv['fixed0']))
        rows.append(row)
        print(f'  {roi}: {Y.shape[1]} voxels; shared lambda {row["lam_shared"]:+.2f} '
              f'(folds {row["lam_fold_mean"]:+.2f} ± {row["lam_fold_sd"]:.2f}); cvR2 x1e3 '
              + ', '.join(f'{n} {1e3 * row[f"cvr2_{n}"]:+.2f}' for n in names), flush=True)
    tag = '' if sim_lam is None else f'_sim-lam{sim_lam:g}-noise{noise_scale:g}'
    out = bids_folder / 'derivatives' / 'encoding_models' / 'mix_lambda' / f'sub-{subject}'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).assign(subject=subject, sim_lam=sim_lam, noise_scale=noise_scale).to_csv(
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
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.targets, a.n_iterations, a.sim_lam, a.noise_scale)
