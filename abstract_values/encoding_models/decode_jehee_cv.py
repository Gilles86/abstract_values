#!/usr/bin/env python3
"""Production decoding with the van Bergen & Jehee (2015) noise model, next to spherical.

Companion to decode_lowrank_cv.py: same encoding models (vonmises orientation /
loggauss value, production settings), same leave-one-run-out folds, same top-N
voxel selection by training R², same GLMsingle gabor betas. In every outer fold,
on the same selected voxels, two noise models decode the held-out run:

  spherical   ResidualFitter(spherical=True): Omega = diag(tau²), Student-t.
              Repeated here so both arms share one encoding fit per fold.
  jehee       ResidualFitter(spherical=False), lambd = 0 (the production
              default of decode_value.py / decode_gabor.py):
              Omega = rho tau tau' + (1 - rho) diag(tau²) + sigma² W W',
              Student-t. W W' is the basis-weight product (orientation) or
              the pseudo-WW' on the stimulus grid (value).

Both posteriors are untempered (T = 1), as in production. N = 50, 250 and the
whole ROI ('all'), so the Jehee arm can be set against the low-rank decoder at
its cross-validated voxel count (which picks ~ the whole ROI).

Output (new tree; nothing else is touched):

  derivatives/decoding_jeheecv/{gabor,value}/sub-<s>/func/
    sub-<s>_mask-<roi>_trials.tsv    per-trial summary, every N x noise model
    sub-<s>_mask-<roi>_meta.tsv      per fold x N: rho, sigma², dof

    python decode_jehee_cv.py 10 --quantity value --roi NPCr
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from braincoder.optimize import ResidualFitter
from braincoder.utils import get_rsq

from abstract_values.encoding_models import lowrank_noise as port
from abstract_values.encoding_models.decode_gabor import get_gabor_paradigm
from abstract_values.encoding_models.decode_lowrank_cv import (
    ROI_HEMI, OrientationCode, ValueCode, summarise)
from abstract_values.encoding_models.decode_value import _build_model, get_value_paradigm
from abstract_values.utils.data import BIDS_FOLDER, Subject


def t_log_posterior(Y, P, omega, dof):
    """Normalised log posterior on the grid under a multivariate Student-t with
    full covariance omega (the normalising constant does not depend on s).
    Y: trials x N, P: grid x N."""
    c = cho_factor(omega, lower=True)
    ll = np.empty((len(Y), len(P)))
    for j, p in enumerate(P):
        r = Y - p
        q = np.einsum('ij,ji->i', r, cho_solve(c, r.T))
        ll[:, j] = -0.5 * (dof + Y.shape[1]) * np.log1p(q / dof)
    return ll - logsumexp(ll, axis=1, keepdims=True)


def fit_noise(code, circular, Y_sel, par_tr, th_sel, grid, spherical):
    if circular:
        model = code.model
        rf = ResidualFitter(model, Y_sel, par_tr, parameters=code.basis, weights=th_sel,
                            lambd=0.0)
    else:
        model = _build_model('loggauss')
        model.init_pseudoWWT(grid, th_sel)
        rf = ResidualFitter(model, Y_sel, par_tr, parameters=th_sel, lambd=0.0)
    omega, dof = rf.fit(init_sigma2=0.1, init_dof=10.0, method='t', learning_rate=0.05,
                        spherical=spherical, max_n_iterations=code.n_iter_noise)
    return model, rf, np.asarray(omega, dtype=np.float64), float(np.asarray(dof))


def main(subject, quantity, roi, n_voxels=(50, 250, 'all'), bids_folder=BIDS_FOLDER,
         out_tree='decoding_jeheecv', debug=False, check_production=True):
    t0 = time.time()
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sub.get_sessions()
    circular = quantity == 'orientation'
    paradigm = (get_gabor_paradigm if circular else get_value_paradigm)(sub, sessions)
    if circular:
        code = OrientationCode(debug=debug)
        grid = np.sort(paradigm['x'].unique()).astype(np.float32)
    else:
        vmin, vmax = float(paradigm['x'].min()), float(paradigm['x'].max())
        code = ValueCode(vmin, vmax, debug=debug)
        grid = np.linspace(vmin, vmax, 50, dtype=np.float32)
    grid_df = pd.DataFrame({'x': grid})
    true_idx_all = port.nearest_grid_index(paradigm['x'].values, grid)

    betas = sub.get_single_trial_estimates(sessions, desc='gabor', smoothed=False)
    assert betas.shape[3] == len(paradigm)
    masker = NiftiMasker(mask_img=sub.get_roi_mask(roi, hemi=ROI_HEMI[roi]),
                         target_affine=betas.affine, target_shape=betas.shape[:3]).fit()
    data = pd.DataFrame(masker.transform(betas).astype(np.float32), index=paradigm.index)
    data = data.loc[:, data.std() > 0]
    n_roi = data.shape[1]
    print(f'sub-{subject} {quantity} {roi}: {len(paradigm)} trials, {n_roi} voxels', flush=True)

    out_dir = bids_folder / 'derivatives' / out_tree / code.subdir / f'sub-{subject}' / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'sub-{subject}_mask-{roi}'
    Ns = [(n, n) for n in n_voxels if n != 'all' and int(n) < n_roi]
    if 'all' in n_voxels:
        Ns.append(('all', n_roi))

    ses = paradigm.index.get_level_values('session')
    run = paradigm.index.get_level_values('run')
    folds = sorted(set(zip(ses, run)))
    if debug:
        folds = folds[:2]
    rng = np.random.default_rng(int(''.join(c for c in str(subject) if c.isdigit()) or 0))
    trial_rows, meta_rows = [], []

    for fs, fr in folds:
        tf = time.time()
        te = (ses == fs) & (run == fr)
        tr = ~te
        par_tr, par_te = paradigm.loc[tr], paradigm.loc[te]
        Y_tr, Y_te = data.loc[tr], data.loc[te]
        theta = code.fit(Y_tr, par_tr)
        pred_tr = pd.DataFrame(code.predict(theta, par_tr), index=Y_tr.index, columns=Y_tr.columns)
        rank = get_rsq(Y_tr, pred_tr).sort_values(ascending=False).index

        for label, N in Ns:
            sel = rank[:int(N)]
            th_sel = code.subset(theta, sel)
            Yte_s = Y_te[sel].values.astype(np.float64)
            P_outer = code.predict(th_sel, grid_df)
            meta = dict(session=fs, run=fr, n_voxels=label, n_selected=int(N))
            for noise, spherical in (('spherical', True), ('jehee', False)):
                model, rf, omega, dof = fit_noise(code, circular, Y_tr[sel], par_tr, th_sel,
                                                  grid, spherical)
                lp = t_log_posterior(Yte_s, P_outer, omega, dof)
                meta[f'dof_{noise}'] = dof
                if not spherical:
                    tau = np.sqrt(np.diag(omega))
                    corr = omega / np.outer(tau, tau)
                    meta['mean_offdiag_corr'] = float(corr[np.triu_indices(len(tau), 1)].mean())
                if check_production and (fs, fr) == folds[0] and label == Ns[0][0]:
                    kw = (dict(parameters=code.basis, weights=th_sel) if circular
                          else dict(parameters=th_sel))
                    pdf = model.get_stimulus_pdf(Y_te[sel], grid, omega=omega.astype(np.float32),
                                                 dof=dof, normalize=False, **kw)
                    pb = pdf.values / pdf.values.sum(1, keepdims=True)
                    diff = np.abs(pb - np.exp(lp)).max()
                    print(f'    [check] {noise} posterior vs get_stimulus_pdf: max |dp| = '
                          f'{diff:.2e}', flush=True)
                    assert diff < 0.1, f'numpy {noise} likelihood does not match braincoder'
                s = summarise(lp, grid, true_idx_all[te], par_te['x'].values, circular, rng)
                idx = par_te.index
                for c, v in (('trial_nr', idx.get_level_values('trial_nr')), ('run', fr),
                             ('session', fs), ('noise', noise), ('n_voxels', label)):
                    s.insert(0, c, v)
                trial_rows.append(s)
            meta_rows.append(meta)
            print(f'  ses-{fs} run-{fr} N={label}({N}): nu_sph={meta["dof_spherical"]:.1f} '
                  f'nu_jehee={meta["dof_jehee"]:.1f} mean corr {meta["mean_offdiag_corr"]:.3f}',
                  flush=True)
        print(f'  fold ses-{fs} run-{fr} done in {time.time() - tf:.0f}s', flush=True)

    trials = pd.concat(trial_rows, ignore_index=True)
    trials.insert(0, 'roi', roi)
    trials.insert(0, 'subject', str(subject))
    trials.to_csv(out_dir / f'{stem}_trials.tsv', sep='\t', index=False)
    meta = pd.DataFrame(meta_rows)
    meta.insert(0, 'roi', roi)
    meta.insert(0, 'subject', str(subject))
    meta.to_csv(out_dir / f'{stem}_meta.tsv', sep='\t', index=False)
    print(f'saved to {out_dir}  ({time.time() - t0:.0f}s)')
    g = trials.groupby(['n_voxels', 'noise'])
    print(g[['logp_true', 'post_sd']].mean().join(g.error.apply(lambda e: e.abs().mean()))
          .join(g.hpd.apply(lambda h: (h < .8).mean()).rename('cov80')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject')
    p.add_argument('--quantity', required=True, choices=['orientation', 'value'])
    p.add_argument('--roi', required=True, choices=list(ROI_HEMI))
    p.add_argument('--n-voxels', nargs='+', default=['50', '250', 'all'])
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--out-tree', default='decoding_jeheecv')
    p.add_argument('--debug', action='store_true', help='2 folds, 100 iterations')
    a = p.parse_args()
    main(a.subject, a.quantity, a.roi, [n if n == 'all' else int(n) for n in a.n_voxels],
         a.bids_folder, a.out_tree, a.debug)
