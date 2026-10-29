#!/usr/bin/env python3
"""Simulate, then decode: predicted precision profiles per value and mapping.

One participant x ROI x generative code. Generative codes:

  vonmises  the participant's own fitted orientation code: the decoder's
            8-channel von Mises basis, weights fitted on ALL trials (positive
            control -- must reproduce V1's mapping x position interaction)
  lam0      mix-lambda=0 full fit: Gaussian tuning on VALUE (fixed-value code)
  lam1      mix-lambda=1 full fit: Gaussian tuning on the within-session value
            QUANTILE = orientation rank (efficient/quantile code)

The mix codes use model_specs' axis x = (1 - lam) v + lam (2 + 40 q(v)); voxels
without a mix fit get zero amplitude (pure noise). Noise: braincoder
LowRankNoise.from_residuals on the residuals of that code on the real data, k =
the participant's median k chosen by the CV-N decoder (decoding_lowrankcv_nvcv),
Student-t dof by ML. Simulated data: one draw per REAL trial (same stimuli,
sessions/mappings, runs), so trial counts per value and mapping are matched.

Two decoders:
  ideal     true generative tuning + true noise, T = 1, flat prior on the
            decoder's grid (value codes: 50 CHF points; orientation code: the 23
            orientations). n_rep replicate datasets; also a noise-scale sweep
            (diagnostic: does the code separate the mappings once noise is low?).
  pipeline  the real decoder (decode_lowrank_cv.py --select-voxels: refit
            encoding model, CV-selected N, low-rank noise with nested-CV k, T)
            run on one simulated dataset. Value codes use the value decoder,
            the orientation code the orientation decoder.

Outputs (new tree, nothing else touched):
  derivatives/decoding_simideal/ideal/sub-<s>/sub-<s>_roi-<roi>_code-<code>.tsv.gz
  derivatives/decoding_simideal/pipeline_<code>/{value,gabor}/sub-<s>/func/  (decoder format)

    python simulate_ideal_decoding.py 10 --roi IPS12 --code lam1 --mode ideal
"""
import argparse
import sys
import time
import zlib
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from scipy.special import logsumexp

from braincoder.noise import LowRankNoise

from abstract_values.encoding_models.decode_gabor import get_gabor_paradigm
from abstract_values.encoding_models.decode_lowrank_cv import (ROI_HEMI, OrientationCode,
                                                                main as decode_main)
from abstract_values.encoding_models.decode_value import get_value_paradigm
from abstract_values.encoding_models.model_specs import get_spec, mix_tag
from abstract_values.encoding_models.simulate_decoding_mix import grid_axis, predict
from abstract_values.utils.data import BIDS_FOLDER, Subject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapping_profiles_collect import mapping_tables, summarise_posterior  # noqa: E402

NOISE_SCALES = (1.0, 0.5, 0.25, 0.1)


def load_roi(sub, roi, n_trials):
    sessions = sub.get_sessions()
    betas = sub.get_single_trial_estimates(sessions, desc='gabor', smoothed=False)
    assert betas.shape[3] == n_trials
    masker = NiftiMasker(mask_img=sub.get_roi_mask(roi, hemi=ROI_HEMI[roi]),
                         target_affine=betas.affine, target_shape=betas.shape[:3]).fit()
    return masker, masker.transform(betas).astype(np.float64)


def mix_params(sub, masker, lam, Y):
    spec = get_spec(f'mix-{mix_tag(lam)}')
    d = (sub.bids_folder / 'derivatives' / 'encoding_models' / spec.out_subdir /
         f'sub-{sub.subject_id}' / 'func')
    p = {k: masker.transform(nib.load(str(
        d / f'sub-{sub.subject_id}_task-abstractvalue_space-T1w_desc-{k}_pe.nii.gz'))).ravel()
         for k in ('mode', 'fwhm', 'amplitude', 'baseline')}
    bad = ~(np.isfinite(p['mode']) & np.isfinite(p['fwhm']) & (p['fwhm'] > 0)
            & np.isfinite(p['amplitude']) & np.isfinite(p['baseline']))
    p['mode'][bad], p['fwhm'][bad], p['amplitude'][bad] = 22.0, 10.0, 0.0
    p['baseline'][bad] = Y[:, bad].mean(0)
    return p, int((~bad).sum())


def chosen_k(bids, subject, roi, quantity):
    fn = bids / 'derivatives' / 'decoding_lowrankcv_nvcv' / 'lowrank_meta.tsv.gz'
    if not fn.exists():
        return 5
    m = pd.read_csv(fn, sep='\t', dtype={'subject': str})
    m = m[(m.subject == str(subject)) & (m.roi == roi) & (m.quantity == quantity)]
    return int(round(m.k.median())) if len(m) else 5


def sample_noise(nm, n, rng, scale=1.0):
    z = rng.standard_normal((n, nm.k)) @ nm.factors.T + \
        rng.standard_normal((n, nm.n_voxels)) * np.sqrt(nm.psi)
    if nm.dof is not None:
        z *= np.sqrt(nm.dof / rng.chisquare(nm.dof, n))[:, None]
    return z * nm.scale * scale


def main(subject, roi, code, mode, bids=BIDS_FOLDER, n_rep=10, seed=0):
    t0 = time.time()
    bids = Path(bids)
    sub = Subject(subject, bids_folder=bids)
    vpar = get_value_paradigm(sub, sub.get_sessions())
    opar = get_gabor_paradigm(sub, sub.get_sessions())
    maps = mapping_tables(sub)
    masker, Y = load_roi(sub, roi, len(vpar))
    ses = vpar.index.get_level_values('session').to_numpy()
    rng = np.random.default_rng(seed + zlib.crc32(f'{subject}{roi}{code}'.encode()))

    if code == 'vonmises':
        quantity, kind = 'orientation', 'gabor'
        oc = OrientationCode()
        W = oc.fit(pd.DataFrame(Y.astype(np.float32), index=opar.index), opar)
        mean = oc.predict(W, opar)
        grid = np.sort(opar['x'].unique()).astype(np.float64)
        P_by_ses = {s: oc.predict(W, pd.DataFrame({'x': grid.astype(np.float32)}))
                    for s in maps}
        true = opar['x'].to_numpy(float)
        n_fit = Y.shape[1]
    else:
        quantity, kind = 'value', 'value'
        lam = {'lam0': 0.0, 'lam1': 1.0}[code]
        p, n_fit = mix_params(sub, masker, lam, Y)
        mean = np.zeros_like(Y)
        v = vpar['x'].to_numpy(float)
        for s, (_, oris, vals) in maps.items():
            m = ses == s
            mean[m] = predict(grid_axis(v[m], oris, vals, lam), p)
        vmin, vmax = float(v.min()), float(v.max())
        grid = np.linspace(vmin, vmax, 50).astype(np.float32).astype(np.float64)
        P_by_ses = {s: predict(grid_axis(grid, oris, vals, lam), p)
                    for s, (_, oris, vals) in maps.items()}
        true = v

    k = chosen_k(bids, subject, roi, quantity)
    nm = LowRankNoise.from_residuals(Y - mean, k=k, dof='fit')
    print(f'sub-{subject} {roi} {code} {mode}: {Y.shape[1]} voxels ({n_fit} with a fit), '
          f'k={nm.k}, nu={nm.dof}', flush=True)

    if mode == 'pipeline':
        Ysim = mean + sample_noise(nm, len(Y), rng)
        decode_main(subject, quantity, roi, bids_folder=bids,
                    out_tree=f'decoding_simideal/pipeline_{code}',
                    check_production=False, select_voxels=True,
                    data=pd.DataFrame(Ysim, index=vpar.index))
        print(f'done ({time.time() - t0:.0f}s)')
        return

    rows = []
    for scale in NOISE_SCALES:
        reps = n_rep if scale == 1.0 else max(2, n_rep // 3)
        nms = LowRankNoise(nm.scale * scale, nm.factors, nm.psi, dof=nm.dof)
        for r in range(reps):
            Ysim = mean + sample_noise(nm, len(Y), rng, scale)
            for s, (mapping, oris, vals) in maps.items():
                m = ses == s
                ll = nms.grid_log_likelihood(Ysim[m], P_by_ses[s])
                post = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
                d = summarise_posterior(post, grid, kind, true[m], oris, vals)
                d.insert(0, 'trial_nr', vpar.index.get_level_values('trial_nr')[m])
                d.insert(0, 'run', vpar.index.get_level_values('run')[m])
                d.insert(0, 'mapping', mapping)
                d.insert(0, 'session', s)
                d.insert(0, 'rep', r)
                d.insert(0, 'noise_scale', scale)
                rows.append(d)
        print(f'  scale {scale}: {reps} reps', flush=True)
    out = pd.concat(rows)
    for c, val in (('n_fit', n_fit), ('n_vox', Y.shape[1]), ('k', nm.k),
                   ('decoder', kind), ('code', code), ('roi', roi), ('subject', str(subject))):
        out.insert(0, c, val)
    od = bids / 'derivatives' / 'decoding_simideal' / 'ideal' / f'sub-{subject}'
    od.mkdir(parents=True, exist_ok=True)
    fn = od / f'sub-{subject}_roi-{roi}_code-{code}.tsv.gz'
    out.to_csv(fn, sep='\t', index=False, float_format='%.5g')
    g = out[out.noise_scale == 1.0].groupby('mapping')
    print(g.v_err.apply(lambda e: e.abs().mean()).round(2).to_string())
    print(f'saved {fn} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--roi', required=True, choices=list(ROI_HEMI))
    ap.add_argument('--code', required=True, choices=['vonmises', 'lam0', 'lam1'])
    ap.add_argument('--mode', default='ideal', choices=['ideal', 'pipeline'])
    ap.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    ap.add_argument('--n-rep', type=int, default=10)
    a = ap.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        main(a.subject, a.roi, a.code, a.mode, a.bids_folder, a.n_rep)
