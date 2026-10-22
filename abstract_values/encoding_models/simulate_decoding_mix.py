"""Simulated decoding precision of the fitted mix-<lambda> codes, per value and mapping.

After neural_priors' expected-uncertainty analysis (ideal Bayesian decoder on
data simulated from the fitted encoding model), rather than Fisher information,
which is local and spiky. For one participant, smoothing level and every
lambda in ``MIX_LAMBDAS``:

  1. Per ROI, voxels whose mean cvR² over the lambda grid beats the null cvR²
     (the same voxels for every lambda; symmetric in lambda).
  2. Each voxel's full-data mix-<lambda> tuning (Gaussian on
     x = (1 - lambda) v + lambda (2 + 40 q(v)), ``model_specs.stimulus_axis``).
  3. Noise: low-rank (k <= 40 factors) + diagonal covariance of the
     in-sample residuals with Student-t scale and dof by ML
     (``noise_models``, ported from retinonumeral, where it was the
     calibrated choice). No temperature: the error of the posterior mean of a
     decoder that knows the generative noise does not need one.
  4. For each mapping and each of the 23 presented values: N patterns
     simulated from the code under THAT mapping, decoded on a continuous CHF
     grid (2-42, flat prior, so the mapping's value prior plays no role) by
     the posterior mean. Error, bias, SD of the estimates, posterior SD.

An efficient (quantile-like) code is most precise where that mapping's values
are dense (Misaligned: 12 and 32 CHF; Aligned: 2, 22, 42); a fixed-value code
has the same precision profile under both mappings.

Output: derivatives/encoding_models/simulated_decoding_mix/sub-<S>/
        sub-<S>_desc-simdecode{_smoothed}.tsv

Usage:
    python -m abstract_values.encoding_models.simulate_decoding_mix 05 --smoothed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image as nimage
from nilearn.maskers import NiftiMasker
from scipy.special import logsumexp

from abstract_values.encoding_models.aggregate_cvr2 import find_per_run_cvr2
from abstract_values.encoding_models.fit_aprf_cv import get_paradigm
from abstract_values.encoding_models.model_specs import (MIX_LAMBDAS, get_spec,
                                                          mix_tag, stimulus_axis)
from abstract_values.encoding_models.noise_models import (WhitenedMVT, fit_scale_dof,
                                                          lowrank_corr_cov,
                                                          safe_cholesky_np)
from abstract_values.encoding_models.permnull_select import roi_masks
from abstract_values.utils.data import BIDS_FOLDER, Subject

ROIS = ['BensonV1', 'BensonV2', 'BensonV3', 'hV4', 'V3ab', 'IPS0', 'IPS12', 'IPS345']
ORI_TABLE = np.arange(0, 180.1, 7.5)
GRID = np.linspace(2, 42, 161)
FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))
MIN_VOX = 10


def mapping_table(sub, sessions):
    """{session: (orientations of the 23 stimuli, their values)} from the events."""
    out = {}
    for s in sessions:
        ev = sub.get_events(s, sub.get_runs(s))
        g = ev[ev['event_type'] == 'gabor'][['orientation', 'value']].drop_duplicates()
        g = g.sort_values('orientation')
        out[s] = (g['orientation'].to_numpy(float), g['value'].to_numpy(float))
    return out


def grid_axis(v, oris, vals, lam):
    """Mixture axis for continuous values v under one mapping (monotone table)."""
    theta = np.interp(v, vals, oris)
    q = (theta / 7.5 - 0.5) / 23                 # continuous mid-rank quantile
    return (1 - lam) * v + lam * (2 + 40 * q)


def predict(x, p):
    sigma = p['fwhm'] * FWHM_TO_SIGMA
    return p['amplitude'] * np.exp(-0.5 * ((x[:, None] - p['mode']) / sigma) ** 2) + p['baseline']


def mean_cv(subject, model_dir, smoothed, bids, masker):
    files = find_per_run_cvr2(subject, model_dir, smoothed, bids)
    if not files:
        return None
    return np.mean([masker.transform(nib.load(str(f))).ravel() for f in files], 0)


def main(subject, smoothed, bids=BIDS_FOLDER, n_sim=1000, k=40, seed=0):
    bids = Path(bids)
    sub = Subject(subject, bids_folder=bids)
    sessions = sorted(sub.get_sessions())
    betas = sub.get_single_trial_estimates(sessions, desc='gabor', smoothed=smoothed)
    ref = nib.Nifti1Image(np.zeros(betas.shape[:3], np.uint8), betas.affine)
    brain = nimage.resample_to_img(sub.get_brain_mask(sessions[0]), ref,
                                   interpolation='nearest', force_resample=True,
                                   copy_header=True).get_fdata() > 0
    rois = {r: m & brain for r, m in roi_masks(sub, ref).items() if r in ROIS}
    union = np.any(list(rois.values()), 0)
    masker = NiftiMasker(nib.Nifti1Image(union.astype(np.uint8), ref.affine)).fit()
    Y = masker.transform(betas).astype(np.float64)
    in_roi = {r: m[union] for r, m in rois.items()}
    maps = mapping_table(sub, sessions)
    cond = {s: sub.get_mapping(s) for s in sessions}
    tag = '_smoothed' if smoothed else ''

    cv_null = mean_cv(subject, 'aprf-null.cv', smoothed, bids, masker)
    cv = {lam: mean_cv(subject, get_spec(f'mix-{mix_tag(lam)}').cv_out_subdir,
                       smoothed, bids, masker) for lam in MIX_LAMBDAS}
    lams = [l for l in MIX_LAMBDAS if cv[l] is not None]
    signal = np.mean([cv[l] for l in lams], 0) > cv_null
    rng = np.random.default_rng(seed)
    rows = []
    for lam in lams:
        spec = get_spec(f'mix-{mix_tag(lam)}')
        d = bids / 'derivatives' / 'encoding_models' / spec.out_subdir / f'sub-{subject}' / 'func'
        fn = lambda p: d / f'sub-{subject}_task-abstractvalue_space-T1w_desc-{p}{tag}_pe.nii.gz'
        if not fn('mode').exists():
            print(f'  lambda {lam}: no full fit, skipped')
            continue
        P = {p: masker.transform(nib.load(str(fn(p)))).ravel()
             for p in ('mode', 'fwhm', 'amplitude', 'baseline')}
        par = get_paradigm(sub, sessions, False, space=spec.stimulus_space)
        x_tr = par['x'].to_numpy(float)
        for roi, m in in_roi.items():
            vox = m & signal & np.isfinite(P['mode'])
            if vox.sum() < MIN_VOX:
                continue
            p = {kk: vv[vox] for kk, vv in P.items()}
            R = Y[:, vox] - predict(x_tr, p)
            kk = int(min(k, vox.sum() - 1, len(R) - 1))
            S = lowrank_corr_cov(R, kk)
            c, nu, _ = fit_scale_dof(R, S)
            omega = c * S
            L = safe_cholesky_np(omega)
            wm = WhitenedMVT(omega, nu)
            for s in sessions:
                oris, vals = maps[s]
                Pg = wm.whiten(predict(grid_axis(GRID, oris, vals, lam), p))
                for v in vals:
                    mu = predict(grid_axis(np.array([v]), oris, vals, lam), p)[0]
                    z = rng.standard_normal((n_sim, vox.sum())) @ L.T
                    z *= np.sqrt(nu / rng.chisquare(nu, n_sim))[:, None]
                    ll = wm.loglik(wm.whiten(mu + z), Pg)
                    post = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
                    est = post @ GRID
                    psd = np.sqrt(np.clip(post @ GRID ** 2 - est ** 2, 0, None))
                    rows.append(dict(subject=subject, smoothed=smoothed, roi=roi,
                                     lam=lam, session=s, mapping=cond[s], value=v,
                                     n_vox=int(vox.sum()), k=kk, nu=nu,
                                     mae=np.mean(np.abs(est - v)), bias=np.mean(est - v),
                                     sd=np.std(est), post_sd=np.mean(psd)))
            print(f'  lambda {lam:+.2f} {roi}: {vox.sum()} voxels, k={kk}, nu={nu:g}, '
                  f'MAE {np.mean([r["mae"] for r in rows[-46:]]):.2f}', flush=True)
    out = bids / 'derivatives' / 'encoding_models' / 'simulated_decoding_mix' / f'sub-{subject}'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f'sub-{subject}_desc-simdecode{tag}.tsv', sep='\t',
                              index=False, float_format='%.5g')
    print(f'saved {len(rows)} rows to {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--smoothed', action='store_true')
    ap.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    ap.add_argument('--n-sim', type=int, default=1000)
    a = ap.parse_args()
    main(a.subject, a.smoothed, a.bids_folder, a.n_sim)
