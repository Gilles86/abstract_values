#!/usr/bin/env python3
"""Tuning and V1 connectivity of IPS voxels, separately and in one model.

Source: V1 (0.75-3.75 deg) inverted linearly into orientation channels with
the von Mises encoding model (``test_coupling.iem_channels``), so every channel
time course is a fixed linear combination of V1 voxels. Its trial-specific part
c_resid(t) is what is left after the session's orientation means (and the
orientations of the +-1 neighbouring trials, ``gates.residualise``) are removed.

Target voxel i, trial t, session s, one tuning curve g(.; mode, fwhm) per voxel:

  coding        y(t) = b + a g(x_lam(v_t))                       a > 0
  connectivity  r(t) = b + beta h_lam'(s)' c_resid(t)            r = IPS residual
  joint         y(t) = b + a g(x_lam(v_t)) + beta h_lam'(s)' c_resid(t)

x_lam(v) = (1 - lam) v + lam (2 + 40 q(v)) is the mix-<lambda> axis
(lam = 0 fixed value, lam = 1 quantile = orientation rank). h_lam'(s) lays the
same tuning curve onto V1's orientation channels: channel phi carries value
v_s(phi) under session s's mapping, read on x_lam'. lam' = 1: the voxel listens
to the same orientations in both sessions (orientation-wired); lam' = 0: to
the orientations carrying its preferred value in that session (value-wired).

Fit: per (lam[, lam']) a grid over (mode, fwhm) with a, beta, b by least squares
on the training runs (shape chosen per voxel by training R^2), leave-one-run-out
cvR^2 on the held-out run. The c_resid control: the same models with c_resid
shuffled across trials of the same session and orientation (keeps the
stimulus, destroys V1's trial-specific deviations), K shuffles.

V1's projection and the residuals are estimated per session on all runs; they
use no IPS data, so they do not leak into the IPS cross-validation.

Output: derivatives/connective_fields/tuning_connectivity/sub-<S>/
  sub-<S>_roi-<R>{_smoothed}_cvr2.npz   per-voxel cvR^2 grids
  sub-<S>{_smoothed}_summary.tsv        ROI means

Usage:
  python -m abstract_values.connective_fields.fit_tuning_connectivity 05 --smoothed
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from abstract_values.connective_fields.gates import (demean_runs, get_paradigm,
                                                     load_mappings, load_roi,
                                                     residualise)
from abstract_values.connective_fields.test_coupling import (IEM_GRID,
                                                             fit_vonmises_weights,
                                                             iem_channels)
from abstract_values.encoding_models.model_specs import MIX_LAMBDAS, stimulus_axis
from abstract_values.utils.data import BIDS_FOLDER, Subject

TARGETS = ['IPS0', 'IPS12', 'IPS345', 'V3ab', 'hV4']
MODES = np.linspace(2, 42, 21)
FWHMS = np.array([4., 6., 8., 12., 16., 24., 32., 48.])
SIGMAS = FWHMS / (2 * np.sqrt(2 * np.log(2)))
N_BASIS, KAPPA = 24, 16.


def gauss(x, mode, sigma):
    return np.exp(-0.5 * ((x - mode) / sigma) ** 2)


def channel_axis(lam, maps, ori):
    """x_lam of every V1 channel under each mapping: {mapping: (n_channels,)}."""
    phi = IEM_GRID
    q = (phi / 7.5 - 0.5) / 23
    return {m: (1 - lam) * np.interp(phi, ori, v) + lam * (2 + 40 * q)
            for m, v in maps.items()}


def connectivity_regressor(lam, maps, ori, cres, cond):
    """(n_shapes, n_trials): h_lam'(s)' c_resid(t) for every grid shape."""
    ax = channel_axis(lam, maps, ori)
    out = np.empty((len(MODES) * len(SIGMAS), len(cres)))
    for m in maps:
        idx = cond == m
        h = gauss(ax[m][None, None, :], MODES[:, None, None], SIGMAS[None, :, None])
        h = h.reshape(-1, len(IEM_GRID))
        h = h - h.mean(1, keepdims=True)
        out[:, idx] = h @ cres[idx].T
    return out


def coding_regressor(x):
    """(n_shapes, n_trials)."""
    g = gauss(x[None, None, :], MODES[:, None, None], SIGMAS[None, :, None])
    return g.reshape(-1, len(x))


def cv_fit(Y, regs, folds, positive=(0,)):
    """Leave-one-run-out grid fit. regs: list of (n_shapes, n_trials) arrays, one
    per regressor (plus an intercept). Shape chosen per voxel by training SSE;
    coefficients listed in ``positive`` must be > 0. Returns cvR^2 per voxel.
    All shapes are solved at once (batched normal equations)."""
    n_shapes = regs[0].shape[0]
    pred = np.zeros_like(Y)
    vi = np.arange(Y.shape[1])
    for te in folds:
        tr = ~te
        Ytr = Y[tr]
        X = np.stack([np.ones((n_shapes, tr.sum()))] + [r[:, tr] for r in regs], -1)  # S,n,p
        xtx = np.einsum('snp,snq->spq', X, X)
        xty = np.einsum('snp,nv->spv', X, Ytr)
        p = xtx.shape[-1]
        xtx = xtx + 1e-9 * np.trace(xtx, axis1=1, axis2=2)[:, None, None] * np.eye(p)
        coef = np.linalg.solve(xtx, xty)                                        # S,p,V
        sse = ((Ytr ** 2).sum(0)[None] - 2 * np.einsum('spv,spv->sv', coef, xty)
               + np.einsum('spv,spq,sqv->sv', coef, xtx, coef))
        if positive:
            ok = np.all(coef[:, [1 + q for q in positive]] > 0, 1)
            sse = np.where(ok, sse, np.inf)
        k = np.argmin(sse, 0)                                                   # V
        cb = coef[k, :, vi]                                                     # V,p
        Xte = np.stack([np.ones((Y.shape[1], te.sum()))]
                       + [r[k][:, te] for r in regs], -1)                       # V,nte,p
        pred[te] = np.einsum('vnp,vp->nv', Xte, cb)
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ss_res / ss_tot


def shuffle_within(c, par, rng):
    out = c.copy()
    for _, idx in par.groupby(['session', 'orientation']).indices.items():
        out[idx] = c[rng.permutation(idx)]
    return out


def main(subject, bids_folder=BIDS_FOLDER, smoothed=False, targets=TARGETS,
         n_shuffle=3, lags=1, seed=0):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    par = get_paradigm(sub, sessions)
    betas = sub.get_single_trial_estimates(sessions, desc='gabor', smoothed=smoothed)
    ori, maps = load_mappings()
    cond = par['condition'].to_numpy()
    folds = [((par['session'] == s) & (par['run'] == r)).to_numpy()
             for (s, r), _ in par.groupby(['session', 'run'])]
    x_mix = {lam: stimulus_axis(par['value'].to_numpy(), par['session'].to_numpy(),
                                f'mix:{lam}') for lam in MIX_LAMBDAS}

    # V1 -> orientation channels (linear), trial-specific part per session
    y_v1, _ = load_roi(sub, betas, 'BensonV1ecc075-375', 'vonmises.cv', bids_folder,
                       smoothed, selection='all')
    w = fit_vonmises_weights(y_v1, par, N_BASIS, KAPPA)
    cres = np.zeros((len(par), len(IEM_GRID)))
    for s in sessions:
        ses = (par['session'] == s).to_numpy()
        p_ses = par[ses].reset_index(drop=True)
        c = iem_channels(demean_runs(y_v1[ses], p_ses), w, kappa=KAPPA)
        cres[ses] = residualise(c, p_ses, lags)
    rng = np.random.default_rng(seed)
    cres_sh = [shuffle_within(cres, par, rng) for _ in range(n_shuffle)]
    conn = {lam: connectivity_regressor(lam, maps, ori, cres, cond) for lam in MIX_LAMBDAS}
    conn_sh = {lam: [connectivity_regressor(lam, maps, ori, c, cond) for c in cres_sh]
               for lam in MIX_LAMBDAS}
    code = {lam: coding_regressor(x_mix[lam]) for lam in MIX_LAMBDAS}

    tag = '_smoothed' if smoothed else ''
    out = bids_folder / 'derivatives' / 'connective_fields' / 'tuning_connectivity' / f'sub-{subject}'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    L = MIX_LAMBDAS
    for roi in targets:
        try:
            y, _ = load_roi(sub, betas, roi, 'aprf.cv', bids_folder, smoothed,
                            selection='all')
        except Exception as e:                       # missing mask
            print(f'  {roi}: skipped ({e})')
            continue
        Y = demean_runs(y, par)
        R = np.zeros_like(Y)
        for s in sessions:
            ses = (par['session'] == s).to_numpy()
            R[ses] = residualise(y[ses], par[ses].reset_index(drop=True), lags)
        res = {'coding': np.stack([cv_fit(Y, [code[l]], folds) for l in L]),
               'connectivity': np.stack([cv_fit(R, [conn[l]], folds, ()) for l in L]),
               'connectivity_shuffled': np.stack([np.mean(
                   [cv_fit(R, [c], folds, ()) for c in conn_sh[l]], 0) for l in L]),
               'joint': np.stack([np.stack([cv_fit(Y, [code[a], conn[b]], folds)
                                            for b in L]) for a in L]),
               'joint_shuffled': np.stack([np.stack([np.mean(
                   [cv_fit(Y, [code[a], c], folds) for c in conn_sh[b]], 0)
                   for b in L]) for a in L])}
        np.savez_compressed(out / f'sub-{subject}_roi-{roi}{tag}_cvr2.npz',
                            lambdas=np.asarray(L), **res)
        for k, v in res.items():
            m = np.nanmean(v, -1)
            if v.ndim == 2:
                rows += [dict(roi=roi, model=k, lam=a, lam_conn=np.nan, cvr2=m[i],
                              n_voxels=v.shape[-1]) for i, a in enumerate(L)]
            else:
                rows += [dict(roi=roi, model=k, lam=a, lam_conn=b, cvr2=m[i, j],
                              n_voxels=v.shape[-1])
                         for i, a in enumerate(L) for j, b in enumerate(L)]
        jc = np.nanmean(res['joint'], -1) - np.nanmean(res['joint_shuffled'], -1)
        i, j = np.unravel_index(np.nanargmax(np.nanmean(res['joint'], -1)), jc.shape)
        print(f'  {roi}: {Y.shape[1]} voxels; best joint lam={L[i]:+.2f}, '
              f'lam_conn={L[j]:+.2f}; connectivity gain over shuffle at best '
              f'{1e3 * jc[i, j]:+.3f} x1e-3', flush=True)
    pd.DataFrame(rows).assign(subject=subject, smoothed=smoothed).to_csv(
        out / f'sub-{subject}{tag}_summary.tsv', sep='\t', index=False)
    print(f'saved to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject')
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--smoothed', action='store_true')
    p.add_argument('--targets', nargs='+', default=TARGETS)
    p.add_argument('--n-shuffle', type=int, default=3)
    p.add_argument('--lags', type=int, default=1)
    a = p.parse_args()
    main(a.subject, a.bids_folder, a.smoothed, a.targets, a.n_shuffle, a.lags)
