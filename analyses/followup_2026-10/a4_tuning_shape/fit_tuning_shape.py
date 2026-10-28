#!/usr/bin/env python3
"""A4: tuning-function shape near the cardinals (one subject, cluster job).

Compares, in orientation space, on the ROI-union voxels (permnullunion: V1-3,
hV4, V3a/b, IPS0, IPS1-2, IPS3-5, NPCl/r), with leave-one-run-out CV:

  null       training-mean prediction (= aprf-null.cv)
  vm         single axial von Mises PRF (= vonmises-prf spec; grid + Adam)
  gvm        exponential-power ("generalized") von Mises, 1 shape parameter,
             warm-started from vm with shape = 1 (nests vm)
  warp       von Mises on a cardinal-warped axis, warp free per voxel,
             warm-started from vm with warp = 0 (nests vm)
  basis_a*   8-channel von Mises basis (kappa 2, ridge alpha 10, = vonmises.cv
             at a = 0) with channels evenly spaced on the warped axis
             F_a(x) = x + a/4 sin 4x; a > 0 = denser/narrower channels at the
             cardinals, a < 0 at the obliques (control)

Outputs (derivatives/followup_2026-10/a4_tuning_shape/sub-<S>/):
  *_voxels.tsv.gz    per voxel: ROI membership, mean-over-folds cvR2 per model,
                     out-of-sample SSE per orientation bin per model, full-data
                     parameters of vm / gvm / warp
  *_decoding.tsv.gz  per (ROI, decoder, test trial): posterior mean, circular SD,
                     error, log density at the truth, HPD rank of the truth
  *_index.tsv        model-implied categorical index (A3) per ROI and model

Usage: python fit_tuning_shape.py 05 [--debug]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image as nimage
from nilearn.maskers import NiftiMasker

from braincoder.optimize import ParameterFitter, ResidualFitter, WeightFitter

from abstract_values.encoding_models.fit_aprf_cv import get_paradigm
from abstract_values.encoding_models.fit_pipeline import fit_one_model
from abstract_values.encoding_models.model_specs import get_spec
from abstract_values.encoding_models.models import (
    CardinalWarpedVonMisesBasis, GeneralizedAxialVonMisesPRF,
    WarpedAxialVonMisesPRF)
from abstract_values.encoding_models.permnull_select import roi_masks
from abstract_values.encoding_models.ridge_alpha import DEFAULT_RIDGE_ALPHA
from abstract_values.utils.data import BIDS_FOLDER, Subject

BASIS_WARPS = (-0.5, 0.0, 0.25, 0.5, 0.75)
N_BASIS, BASIS_KAPPA = 8, 2.0
DECODE_ROIS = ('BensonV1', 'BensonV2', 'BensonV3', 'IPS0', 'IPS12', 'IPS345')
DECODERS = ('vm', 'gvm', 'warp', 'basis_a+0.00', 'basis_a+0.50', 'basis_a-0.50')
GRID = np.deg2rad(np.arange(0, 180, 0.5)).astype(np.float32)
BINS = {'near90': lambda d: np.abs(d - 90) <= 15,
        'near0': lambda d: (d <= 15) | (d >= 165)}


def rsq(y, yhat):
    """Per-voxel R2 = 1 - SS_res / SS_tot (numpy; braincoder's get_rsq wants pandas)."""
    y, yhat = np.asarray(y, np.float64), np.asarray(yhat, np.float64)
    ss = ((y - y.mean(0)) ** 2).sum(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(ss > 0, 1 - ((y - yhat) ** 2).sum(0) / ss, np.nan)


def btag(a):
    return f'basis_a{a:+.2f}'


def orientation_bin(deg):
    out = np.full(len(deg), 'oblique', dtype=object)
    for k, f in BINS.items():
        out[f(deg)] = k
    return out


def basis_pars():
    return pd.DataFrame({
        'mu': np.linspace(0, np.pi, N_BASIS, endpoint=False).astype(np.float32),
        'kappa': np.full(N_BASIS, BASIS_KAPPA, np.float32),
        'amplitude': np.ones(N_BASIS, np.float32),
        'baseline': np.zeros(N_BASIS, np.float32)})


def fit_prfs(d, p, n_iter, debug):
    """vm by grid + Adam; gvm and warp warm-started from it (nested)."""
    xr = (float(p['x'].min()), float(p['x'].max()))
    vm, _ = fit_one_model(get_spec('vonmises-prf'), d, p, *xr,
                          n_iterations=n_iter, debug=debug, log_prefix='    ')
    vm = vm.astype(np.float32)
    out = {'vm': vm}
    for name, cls, col, init in (('gvm', GeneralizedAxialVonMisesPRF, 'shape', 1.0),
                                 ('warp', WarpedAxialVonMisesPRF, 'warp', 0.0)):
        m = cls()
        init_pars = vm.assign(**{col: np.float32(init)})[m.parameter_labels]
        out[name] = ParameterFitter(m, d, p).fit(
            max_n_iterations=50 if debug else n_iter, init_pars=init_pars,
            progressbar=False).astype(np.float32)
    return out


PRF_CLS = {'vm': lambda: get_spec('vonmises-prf').cls(**get_spec('vonmises-prf').cls_kwargs),
           'gvm': GeneralizedAxialVonMisesPRF, 'warp': WarpedAxialVonMisesPRF}


def predict_prf(name, pars, p):
    return np.asarray(PRF_CLS[name]().predict(parameters=pars, paradigm=p), np.float32)


def fit_bases(d, p):
    bp = basis_pars()
    return {btag(a): WeightFitter(CardinalWarpedVonMisesBasis(warp=a), bp, d, p)
            .fit(alpha=DEFAULT_RIDGE_ALPHA) for a in BASIS_WARPS}


def predict_basis(tag, w, p):
    a = float(tag.split('_a')[1])
    return np.asarray(CardinalWarpedVonMisesBasis(warp=a).basis_predictions(p, basis_pars())
                      @ w.values, np.float32)


def posterior_metrics(post, true_rad):
    """post: (n, G) normalised on GRID. Axial circular summaries in degrees."""
    z = post @ np.exp(2j * GRID)
    mean = (np.angle(z) / 2) % np.pi
    R = np.clip(np.abs(z), 1e-12, 1)
    sd = np.rad2deg(np.sqrt(-2 * np.log(R)) / 2)
    err = (np.rad2deg(mean - true_rad) + 90) % 180 - 90
    ti = np.argmin(np.abs(((GRID[None] - true_rad[:, None]) + np.pi / 2) % np.pi
                          - np.pi / 2), 1)
    pt = post[np.arange(len(post)), ti]
    rank = np.where(post > pt[:, None], post, 0).sum(1)  # HPD mass above the truth
    step = np.rad2deg(GRID[1] - GRID[0])
    return dict(post_mean=np.rad2deg(mean), post_sd=sd, error=err,
                logdens_true=np.log(np.clip(pt / step, 1e-300, None)), hpd_rank=rank,
                map=np.rad2deg(GRID[post.argmax(1)]))


def decode(name, model_obj, train_d, train_p, test_d, pars=None, weights=None,
           bpars=None, n_noise=1000):
    kw = dict(parameters=pars) if weights is None else dict(parameters=bpars,
                                                            weights=weights)
    rf = ResidualFitter(model_obj, train_d, train_p, **kw)
    omega, dof = rf.fit(init_sigma2=0.1, init_dof=10.0, method='t', learning_rate=0.05,
                        spherical=True, max_n_iterations=n_noise, progressbar=False)
    pdf = model_obj.get_stimulus_pdf(test_d, GRID, omega=omega, dof=dof,
                                     normalize=False, **kw)
    pdf = np.asarray(pdf, np.float64)
    return pdf / pdf.sum(1, keepdims=True), float(dof)


def categorical_index(pred, sigma):
    """pred: (23 orientations, V) predicted responses at 7.5..172.5 deg; sigma: (V,).
    Model-implied distance as in A3 (compute_rdms.model_rdm): D_ij = mean_v
    ((f_v(i) - f_v(j)) / sigma_v)^2. d90 = mean D of the two adjacent pairs touching
    90 deg, dadj = mean D of the other 20 adjacent pairs, dall = mean over all 253
    pairs. ratio = d90 / dadj (A3 'ratio'), CI = (d90 - dadj) / dall (A3 'CI').
    Same for 45 and 135 deg as oblique controls."""
    Z = pred / sigma
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).mean(-1)
    adj = np.diag(D, 1)                                   # 22 adjacent pairs
    i, j = np.triu_indices(len(D), 1)
    dall = D[i, j].mean()
    ori = np.arange(1, 24) * 7.5
    out = {'dall': dall, 'mean_adjacent': adj.mean()}
    for x in (90, 45, 135):
        k = np.flatnonzero(ori == x)[0]
        d_x = adj[[k - 1, k]].mean()
        d_o = np.delete(adj, [k - 1, k]).mean()
        out[f'ratio_{x}'] = d_x / d_o
        out[f'ci_{x}'] = (d_x - d_o) / dall
    return out


def main(subject, bids_folder=BIDS_FOLDER, n_iterations=1000, n_voxels=100,
         out_root=None, debug=False):
    t0 = time.time()
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    par = get_paradigm(sub, sessions, False, space='orientation')
    deg = np.rad2deg(par['x'].to_numpy())
    obin = orientation_bin(np.round(deg, 3))

    betas = sub.get_single_trial_estimates(sessions, desc='gabor')
    ref = nib.Nifti1Image(np.zeros(betas.shape[:3], np.uint8), betas.affine)
    brain = nimage.resample_to_img(sub.get_brain_mask(sessions[0]), ref,
                                   interpolation='nearest', force_resample=True,
                                   copy_header=True).get_fdata() > 0
    rois = roi_masks(sub, ref)
    union = np.zeros(ref.shape, bool)
    for m in rois.values():
        union |= m
    union &= brain
    masker = NiftiMasker(mask_img=nib.Nifti1Image(union.astype(np.float32),
                                                  ref.affine)).fit()
    data = masker.transform(betas).astype(np.float32)
    flat = np.flatnonzero(union.ravel())
    ok = np.isfinite(data).all(0) & (data.std(0) > 0)
    data, flat = data[:, ok], flat[ok]
    if debug:
        keep = np.sort(np.random.default_rng(0).choice(data.shape[1], 400, replace=False))
        data, flat = data[:, keep], flat[keep]
    V = data.shape[1]
    inroi = {r: m.ravel()[flat] for r, m in rois.items()}
    data = pd.DataFrame(data, index=par.index)
    print(f'sub-{subject}: {len(par)} trials, {V} voxels, ROIs {list(rois)}', flush=True)

    out_dir = Path(out_root or bids_folder / 'derivatives' / 'followup_2026-10'
                   / 'a4_tuning_shape') / f'sub-{subject}'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'sub-{subject}_task-abstractvalue_space-T1w'

    models = ['null', 'vm', 'gvm', 'warp'] + [btag(a) for a in BASIS_WARPS]
    pred = {m: np.zeros(data.shape, np.float32) for m in models}
    cvr2 = {m: [] for m in models}
    folds = sorted(set(zip(par.index.get_level_values('session'),
                           par.index.get_level_values('run'))))
    if debug:
        folds = folds[:2]
    dec_rows = []
    for fi, (ses, run) in enumerate(folds):
        te = ((par.index.get_level_values('session') == ses)
              & (par.index.get_level_values('run') == run))
        tr_p = par.loc[~te, ['x']].reset_index(drop=True)
        te_p = par.loc[te, ['x']].reset_index(drop=True)
        tr_d = data.loc[~te].reset_index(drop=True)
        te_d = data.loc[te].reset_index(drop=True)
        print(f'\n[fold {fi + 1}/{len(folds)}] ses-{ses} run-{run}  '
              f'{time.time() - t0:.0f}s', flush=True)
        prf = fit_prfs(tr_d, tr_p, n_iterations, debug)
        bw = fit_bases(tr_d, tr_p)
        fold_pred = {'null': np.tile(tr_d.mean(0).to_numpy(np.float32), (te.sum(), 1))}
        fold_pred.update({k: predict_prf(k, v, te_p) for k, v in prf.items()})
        fold_pred.update({k: predict_basis(k, v, te_p) for k, v in bw.items()})
        for m in models:
            pred[m][te] = fold_pred[m]
            cvr2[m].append(np.asarray(rsq(te_d.to_numpy(), fold_pred[m]), np.float32))
        print('  mean cvR2 - null: ' + '  '.join(
            f'{m} {np.nanmean(cvr2[m][-1] - cvr2["null"][-1]):+.4f}' for m in models[1:]),
            flush=True)

        # ── decoding: same top-N voxels for every decoder ──────────────────
        r2_vm = np.asarray(rsq(tr_d.to_numpy(), predict_prf('vm', prf['vm'], tr_p)))
        r2_b0 = np.asarray(rsq(tr_d.to_numpy(),
                                   predict_basis(btag(0.), bw[btag(0.)], tr_p)))
        score = np.nan_to_num((r2_vm + r2_b0) / 2, nan=-np.inf)
        true_rad = par.loc[te, 'x'].to_numpy()
        idx = par.loc[te].index
        for roi in DECODE_ROIS:
            if roi not in inroi or inroi[roi].sum() < 20:
                continue
            cand = np.flatnonzero(inroi[roi])
            sel = cand[np.argsort(-score[cand])[:n_voxels]]
            for dname in DECODERS:
                try:
                    if dname in prf:
                        pars = prf[dname].iloc[sel]
                        post, dof = decode(dname, PRF_CLS[dname](), tr_d[sel], tr_p,
                                           te_d[sel], pars=pars,
                                           n_noise=100 if debug else 1000)
                    else:
                        a = float(dname.split('_a')[1])
                        post, dof = decode(dname, CardinalWarpedVonMisesBasis(warp=a),
                                           tr_d[sel], tr_p, te_d[sel],
                                           weights=bw[dname][sel], bpars=basis_pars(),
                                           n_noise=100 if debug else 1000)
                except Exception as e:                       # record, don't die
                    print(f'  decode {roi} {dname} failed: {e!r}', flush=True)
                    continue
                met = posterior_metrics(post, true_rad)
                dec_rows.append(pd.DataFrame(dict(
                    roi=roi, decoder=dname, session=idx.get_level_values('session'),
                    run=idx.get_level_values('run'), trial=idx.get_level_values('trial'),
                    true_deg=np.rad2deg(true_rad), n_vox=len(sel), dof=dof, **met)))
        print(f'  decoded  {time.time() - t0:.0f}s', flush=True)
        pd.concat(dec_rows).to_csv(out_dir / f'{stem}_desc-a4_decoding.tsv.gz',
                                   sep='\t', index=False)          # running save

    # ── voxel table: cvR2 + binned out-of-sample SSE ───────────────────────
    vox = pd.DataFrame({'flat_index': flat})
    for r, m in inroi.items():
        vox[f'in_{r}'] = m
    for m in models:
        vox[f'cvr2_{m}'] = np.nanmean(np.stack(cvr2[m]), 0)
        res2 = (data.to_numpy() - pred[m]) ** 2
        for b in ('near90', 'near0', 'oblique'):
            vox[f'sse_{b}_{m}'] = res2[obin == b].sum(0)

    # ── full-data fits: parameters + model-implied categorical index ───────
    print('\n[full-data fits]', flush=True)
    p_all = par[['x']].reset_index(drop=True)
    d_all = data.reset_index(drop=True)
    prf = fit_prfs(d_all, p_all, n_iterations, debug)
    bw = fit_bases(d_all, p_all)
    for k, v in prf.items():
        for c in v.columns:
            vox[f'{k}_{c}'] = v[c].to_numpy()
        vox[f'{k}_r2'] = np.asarray(rsq(d_all.to_numpy(), predict_prf(k, v, p_all)))
    vox['warp_a'] = 0.95 * np.tanh(vox['warp_warp'])
    vox.to_csv(out_dir / f'{stem}_desc-a4_voxels.tsv.gz', sep='\t', index=False)

    grid23 = pd.DataFrame({'x': np.deg2rad(np.arange(1, 24) * 7.5).astype(np.float32)})
    full_pred = {k: (predict_prf(k, v, grid23), predict_prf(k, v, p_all))
                 for k, v in prf.items()}
    full_pred.update({k: (predict_basis(k, v, grid23), predict_basis(k, v, p_all))
                      for k, v in bw.items()})
    good = ((vox.cvr2_vm > vox.cvr2_null) | (vox[f'cvr2_{btag(0.)}'] > vox.cvr2_null)
            ).to_numpy()
    idx_rows = []
    for roi in DECODE_ROIS:
        if roi not in inroi:
            continue
        sel = np.flatnonzero(inroi[roi] & good)
        if len(sel) < 10:
            continue
        for k, (g, fitted) in full_pred.items():
            sigma = (d_all.to_numpy()[:, sel] - fitted[:, sel]).std(0)
            idx_rows.append(dict(roi=roi, model=k, n_vox=len(sel),
                                 **categorical_index(g[:, sel], sigma)))
    pd.DataFrame(idx_rows).to_csv(out_dir / f'{stem}_desc-a4_index.tsv', sep='\t',
                                  index=False)
    print(f'saved to {out_dir}  ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    ap.add_argument('--n-iterations', type=int, default=1000)
    ap.add_argument('--n-voxels', type=int, default=100)
    ap.add_argument('--out-root', default=None)
    ap.add_argument('--debug', action='store_true',
                    help='400 voxels, 2 folds, 50 iterations')
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.n_iterations, a.n_voxels, a.out_root, a.debug)
