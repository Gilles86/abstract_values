#!/usr/bin/env python3
"""Production decoding with the noise model swapped: spherical vs low-rank (nested CV).

Mirrors the production decoders exactly up to the noise model:

  orientation  decode_gabor.py  --model vonmises (8-channel axial von Mises
               basis, kappa 2, ridge alpha = project default), stimulus grid =
               the 23 presented orientations
  value        decode_value.py  --model loggauss (per-voxel log-Gaussian PRF,
               20 x 15 correlation grid -> amplitude/baseline -> 1000 Adam
               iterations), 50-point CHF grid between the subject's min and max

Same leave-one-run-out folds, same top-N voxel selection by training R², same
GLMsingle (unsmoothed) gabor betas. In every outer fold, on the SAME selected
voxels, three noise models decode the held-out run:

  spherical   production: ResidualFitter(spherical=True), Student-t, per-voxel
              tau, 1000 (orientation) / 5000 (value) iterations. (lambd is
              irrelevant on the spherical path.) Posterior = likelihood, T = 1.
  lowrankcv   braincoder.noise.LowRankNoise (native; braincoder v0.6.0 +
              branch lowrank-inner-weights): Omega = D(FF' + Psi)D with k
              factors of the residual correlation, Student-t, tempered
              posterior p(y|s)^(1/T); k and T jointly by leave-one-run-out CV
              inside the 15 training runs on the held-out log score, inner
              encoding parameters (or, for the von Mises basis, inner WEIGHTS)
              refit without the inner run (retinonumeral decode_fdr.py
              --noise_model lowrank_cv --inner_params loro). The numpy port in
              lowrank_noise.py is run on the first fold as a cross-check.
  diagcv      the k = 0 row of the same CV table at its best T: diagonal
              Student-t + temperature only. Separates "correlations" from
              "temperature".

Needs braincoder >= v0.6.0 with the (parameters, weights) inner-fold support:
put a checkout of branch lowrank-inner-weights first on PYTHONPATH (the SLURM
script does this).

Output (new tree; production derivatives are never touched):

  derivatives/decoding_lowrankcv/{gabor,value}/sub-<s>/func/
    sub-<s>_mask-<roi>_nvoxels-<N>_noise-<model>_pars.tsv   normalised posterior,
        production layout (index session, run, trial_nr, true_*)
    sub-<s>_mask-<roi>_trials.tsv     per-trial summary, every N x noise model
    sub-<s>_mask-<roi>_meta.tsv       per fold x N: k, T, dof, CV scores

    python decode_lowrank_cv.py 10 --quantity value --roi NPCr
    python decode_lowrank_cv.py 10 --quantity orientation --roi BensonV1 --debug
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import AxialVonMisesPRF
from braincoder.optimize import ParameterFitter, ResidualFitter, WeightFitter
from braincoder.utils import get_rsq

from abstract_values.encoding_models.decode_gabor import (
    get_gabor_paradigm, make_basis_parameters)
from abstract_values.encoding_models.decode_value import (
    _build_model, _grid_ranges, get_value_paradigm)
from braincoder.noise import LowRankNoise
from braincoder.utils.calibration import hpd_level, pit
from scipy.special import logsumexp

from abstract_values.encoding_models import lowrank_noise as port
from abstract_values.encoding_models.ridge_alpha import DEFAULT_RIDGE_ALPHA
from abstract_values.utils.data import BIDS_FOLDER, Subject

#: ROI -> hemi entity used by Subject.get_roi_mask
ROI_HEMI = {'BensonV1': 'LR', 'BensonV2': 'LR', 'BensonV3': 'LR',
            'NPCr': None, 'IPS0': None, 'IPS12': None, 'IPS345': None}


# ── encoding models (production settings) ────────────────────────────────────

class OrientationCode:
    """decode_gabor.py --model vonmises."""
    subdir, true_col = 'gabor', 'true_orientation_rad'
    n_iter_noise = 1000

    def __init__(self, n_basis=8, kappa=2.0, alpha=DEFAULT_RIDGE_ALPHA, debug=False):
        self.basis = make_basis_parameters(n_basis, kappa)
        self.alpha = alpha
        self.model = AxialVonMisesPRF(allow_neg_amplitudes=True)
        if debug:
            self.n_iter_noise = 100

    def fit(self, data, paradigm):
        return WeightFitter(self.model, self.basis, data, paradigm).fit(alpha=self.alpha)

    def predict(self, w, paradigm):
        bp = np.asarray(self.model.basis_predictions(paradigm, self.basis), dtype=np.float64)
        return bp @ np.asarray(w.values, dtype=np.float64)

    def lowrank_args(self, theta_sel):
        """(parameters, weights) in LowRankNoise.fit / get_stimulus_pdf terms."""
        return self.basis, theta_sel

    @staticmethod
    def subset(w, cols):
        return w[cols]


class ValueCode:
    """decode_value.py --model loggauss."""
    subdir, true_col = 'value', 'true_value_chf'
    n_iter_noise = 5000

    def __init__(self, value_min, value_max, n_iterations=1000, debug=False):
        self.mus, self.sds = _grid_ranges('loggauss', value_min, value_max, 20, 15)
        self.n_iterations = 100 if debug else n_iterations
        if debug:
            self.n_iter_noise = 100

    def fit(self, data, paradigm):
        model = _build_model('loggauss')
        fitter = ParameterFitter(model, data, paradigm)
        g = fitter.fit_grid(self.mus, self.sds, np.array([1.0], np.float32),
                            np.array([0.0], np.float32), use_correlation_cost=True)
        g = fitter.refine_baseline_and_amplitude(g)
        return fitter.fit(max_n_iterations=self.n_iterations, init_pars=g)

    def predict(self, pars, paradigm):
        return np.asarray(_build_model('loggauss').predict(parameters=pars, paradigm=paradigm))

    def lowrank_args(self, theta_sel):
        return theta_sel, None

    @staticmethod
    def subset(pars, cols):
        return pars.loc[cols]


def log_posterior(noise, Y, P, temperature=None):
    """Tempered, normalised log posterior on the grid from a LowRankNoise."""
    ll = noise.grid_log_likelihood(Y, P)
    T = noise.temperature if temperature is None else temperature
    if np.isinf(T):
        return np.full_like(ll, -np.log(ll.shape[1]))
    a = ll / T
    return a - logsumexp(a, axis=1, keepdims=True)


# ── posterior summaries ──────────────────────────────────────────────────────

def summarise(logpost, grid, true_idx, true_val, circular, rng):
    p = np.exp(logpost)
    p /= p.sum(1, keepdims=True)
    tg = grid[true_idx]                       # truth snapped to the grid
    out = dict(logp_true=logpost[np.arange(len(p)), true_idx] + np.log(len(grid)),
               hpd=hpd_level(p, tg, grid=grid, rng=rng))
    if circular:
        z = p @ np.exp(2j * grid)
        mean = (np.angle(z) / 2) % np.pi
        R = np.clip(np.abs(z), 1e-12, 1)
        out.update(post_mean=np.rad2deg(mean), post_sd=np.rad2deg(np.sqrt(-2 * np.log(R)) / 2),
                   post_map=np.rad2deg(grid[p.argmax(1)]), true=np.rad2deg(true_val))
        out['error'] = (out['post_mean'] - out['true'] + 90) % 180 - 90
        out['pit'] = np.nan
    else:
        mean = p @ grid
        out.update(post_mean=mean, post_sd=np.sqrt(np.clip(p @ grid ** 2 - mean ** 2, 0, None)),
                   post_map=grid[p.argmax(1)], true=true_val, pit=pit(p, tg, grid=grid, rng=rng))
        out['error'] = out['post_mean'] - out['true']
    return pd.DataFrame(out)


# ── main ─────────────────────────────────────────────────────────────────────

def main(subject, quantity, roi, n_voxels=(50, 250), bids_folder=BIDS_FOLDER,
         out_tree='decoding_lowrankcv', debug=False, check_production=True):
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
    mask = sub.get_roi_mask(roi, hemi=ROI_HEMI[roi])
    masker = NiftiMasker(mask_img=mask, target_affine=betas.affine,
                         target_shape=betas.shape[:3]).fit()
    data = pd.DataFrame(masker.transform(betas).astype(np.float32), index=paradigm.index)
    print(f'sub-{subject} {quantity} {roi}: {len(paradigm)} trials, {data.shape[1]} voxels, '
          f'grid {len(grid)}', flush=True)

    out_dir = bids_folder / 'derivatives' / out_tree / code.subdir / f'sub-{subject}' / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'sub-{subject}_mask-{roi}'

    n_voxels = sorted(int(n) for n in n_voxels)
    n_max = min(max(n_voxels), data.shape[1])
    ses = paradigm.index.get_level_values('session')
    run = paradigm.index.get_level_values('run')
    folds = sorted(set(zip(ses, run)))
    if debug:
        folds = folds[:2]
    rng = np.random.default_rng(int(''.join(c for c in str(subject) if c.isdigit()) or 0))

    trial_rows, meta_rows = [], []
    posts = {}  # (N, noise) -> list of DataFrames

    for fs, fr in folds:
        tf = time.time()
        te = (ses == fs) & (run == fr)
        tr = ~te
        par_tr, par_te = paradigm.loc[tr], paradigm.loc[te]
        Y_tr, Y_te = data.loc[tr], data.loc[te]

        # outer encoding fit on the whole ROI + top-N ranking by training R²
        theta = code.fit(Y_tr, par_tr)
        pred_tr = pd.DataFrame(code.predict(theta, par_tr), index=Y_tr.index, columns=Y_tr.columns)
        r2 = get_rsq(Y_tr, pred_tr)
        rank = r2.sort_values(ascending=False).index
        top = rank[:n_max]

        # inner leave-one-run-out encoding fits (only on the top voxels)
        g_tr = np.array([f'{a}-{b}' for a, b in zip(ses[tr], run[tr])])
        inner = {}
        for g in pd.unique(g_tr):
            keep = g_tr != g
            inner[g] = code.fit(Y_tr[top].loc[keep], par_tr.loc[keep])
        t_inner = time.time() - tf

        for N in n_voxels:
            if N > data.shape[1]:
                continue
            sel = rank[:N]
            th_sel = code.subset(theta, sel)
            Ytr_s, Yte_s = Y_tr[sel].values.astype(np.float64), Y_te[sel].values.astype(np.float64)
            P_outer = code.predict(th_sel, grid_df)                      # S x N
            R_all = Ytr_s - code.predict(th_sel, par_tr)
            ti_tr, ti_te = true_idx_all[tr], true_idx_all[te]

            # --- spherical: production ResidualFitter ---
            if circular:
                rf = ResidualFitter(code.model, Y_tr[sel], par_tr, parameters=code.basis,
                                    weights=th_sel, lambd=0.1)
            else:
                m_sel = _build_model('loggauss')
                m_sel.init_pseudoWWT(grid, th_sel)
                rf = ResidualFitter(m_sel, Y_tr[sel], par_tr, parameters=th_sel, lambd=0.1)
            omega, dof = rf.fit(init_sigma2=0.1, init_dof=10.0, method='t', learning_rate=0.05,
                                spherical=True, max_n_iterations=code.n_iter_noise)
            omega = np.asarray(omega, dtype=np.float64)
            dof = float(np.asarray(dof))
            sph = LowRankNoise(np.sqrt(np.diag(omega)), np.zeros((N, 0)), np.ones(N), dof=dof)
            lp_sph = log_posterior(sph, Yte_s, P_outer)

            if check_production and (fs, fr) == folds[0]:
                # numpy likelihood == braincoder get_stimulus_pdf (production path)
                if circular:
                    pdf = code.model.get_stimulus_pdf(Y_te[sel], grid, parameters=code.basis,
                                                      weights=th_sel, omega=omega.astype(np.float32),
                                                      dof=dof, normalize=False)
                else:
                    pdf = m_sel.get_stimulus_pdf(Y_te[sel], grid, parameters=th_sel,
                                                 omega=omega.astype(np.float32), dof=dof,
                                                 normalize=False)
                pb = pdf.values / pdf.values.sum(1, keepdims=True)
                diff = np.abs(pb - np.exp(lp_sph)).max()
                print(f'    [check] spherical posterior vs get_stimulus_pdf: max |dp| = {diff:.2e}')
                # braincoder evaluates this likelihood in float32: ~1e-3 on sharp posteriors
                assert diff < 2e-2, 'numpy spherical likelihood does not match braincoder'

            # --- low-rank, k and T by nested LORO (native braincoder) ---
            model = code.model if circular else m_sel
            pars_lr, w_lr = code.lowrank_args(th_sel)
            if circular:
                inner_fn = lambda d, p, g, sel=sel: (code.basis, inner[g][sel])
            else:
                inner_fn = lambda d, p, g, sel=sel: inner[g].loc[sel]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                lr = LowRankNoise.fit(model, Y_tr[sel], par_tr, pars_lr, weights=w_lr,
                                      stimulus_range=grid, groups=g_tr,
                                      inner_parameters=inner_fn)
            cv = lr.cv_scores_
            lp_lr = log_posterior(lr, Yte_s, P_outer)
            T0 = float(cv.loc[0].idxmax()) if 0 in cv.index else np.nan
            dg = LowRankNoise.from_residuals(R_all, k=0, temperature=T0)
            lp_dg = log_posterior(dg, Yte_s, P_outer)

            if check_production and (fs, fr) == folds[0]:
                # (1) grid predictions == braincoder's own; (2) our log posterior ==
                # get_stimulus_pdf(noise_model=...); (3) numpy port == native CV table
                P_bc = model.predict_stimulus_grid(grid, parameters=pars_lr, weights=w_lr)
                d1 = np.abs(np.asarray(P_bc) - P_outer).max()
                pdf = model.get_stimulus_pdf(Y_te[sel], grid, parameters=pars_lr, weights=w_lr,
                                             noise_model=lr, normalize=False)
                pb = pdf.values / pdf.values.sum(1, keepdims=True)
                d2 = np.abs(pb - np.exp(lp_lr)).max()

                def port_inner(g, sel=sel):
                    keep = g_tr != g
                    th_g = code.subset(inner[g], sel)
                    return (Ytr_s[keep] - code.predict(th_g, par_tr.loc[keep]),
                            code.predict(th_g, grid_df))
                _, cv_port = port.fit_lowrank_cv(Ytr_s, R_all, ti_tr, g_tr, port_inner)
                fin = np.isfinite(cv.values)
                d3 = np.abs(cv.values[fin] - cv_port.values[fin]).max()
                print(f'    [check] grid pred |d| = {d1:.1e}, pdf |dp| = {d2:.1e}, '
                      f'port vs native CV |d| = {d3:.1e}')
                assert d1 < 1e-3 and d2 < 1e-3 and d3 < 1e-3

            meta_rows.append(dict(
                session=fs, run=fr, n_voxels=N, k=lr.k, temperature=lr.temperature,
                dof_lowrank=lr.dof, dof_spherical=dof, temperature_diag=T0, dof_diag=dg.dof,
                cv_best=float(cv.values.max()),
                cv_k0_best=float(cv.loc[0].max()) if 0 in cv.index else np.nan,
                cv_bestk_T1=float(cv.loc[lr.k, 1.0]) if 1.0 in cv.columns else np.nan,
                cv_k0_T1=float(cv.loc[0, 1.0]) if 0 in cv.index else np.nan,
                n_train=len(Ytr_s), n_test=len(Yte_s)))

            idx = par_te.index
            for noise, lp in (('spherical', lp_sph), ('lowrankcv', lp_lr), ('diagcv', lp_dg)):
                s = summarise(lp, grid, ti_te, par_te['x'].values, circular, rng)
                s.insert(0, 'trial_nr', idx.get_level_values('trial_nr'))
                s.insert(0, 'run', fr)
                s.insert(0, 'session', fs)
                s.insert(0, 'noise', noise)
                s.insert(0, 'n_voxels', N)
                trial_rows.append(s)
                pdf = pd.DataFrame(np.exp(lp), columns=grid, index=pd.MultiIndex.from_arrays(
                    [idx.get_level_values('session'), idx.get_level_values('run'),
                     idx.get_level_values('trial_nr'), par_te['x'].values],
                    names=['session', 'run', 'trial_nr', code.true_col]))
                posts.setdefault((N, noise), []).append(pdf)
            m = meta_rows[-1]
            print(f'  ses-{fs} run-{fr} N={N}: k={m["k"]} T={m["temperature"]:g} '
                  f'nu={m["dof_lowrank"]} | T(k=0)={T0:g} | cv {m["cv_best"]:.3f} '
                  f'(k=0 {m["cv_k0_best"]:.3f}, T=1 {m["cv_bestk_T1"]:.3f})', flush=True)
        print(f'  fold ses-{fs} run-{fr} done in {time.time() - tf:.0f}s '
              f'(inner fits {t_inner:.0f}s)', flush=True)

    trials = pd.concat(trial_rows, ignore_index=True)
    trials.insert(0, 'roi', roi)
    trials.insert(0, 'subject', str(subject))
    trials.to_csv(out_dir / f'{stem}_trials.tsv', sep='\t', index=False)
    meta = pd.DataFrame(meta_rows)
    meta.insert(0, 'roi', roi)
    meta.insert(0, 'subject', str(subject))
    meta.to_csv(out_dir / f'{stem}_meta.tsv', sep='\t', index=False)
    for (N, noise), lst in posts.items():
        pd.concat(lst).to_csv(out_dir / f'{stem}_nvoxels-{N}_noise-{noise}_pars.tsv', sep='\t')
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
    p.add_argument('--n-voxels', type=int, nargs='+', default=[50, 250])
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--out-tree', default='decoding_lowrankcv')
    p.add_argument('--debug', action='store_true', help='2 folds, 100 iterations')
    a = p.parse_args()
    main(a.subject, a.quantity, a.roi, a.n_voxels, a.bids_folder, a.out_tree, a.debug)
