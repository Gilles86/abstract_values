"""End-to-end check of the hierarchical lambda: simulate betas with a KNOWN code, refit.

Per participant and voxel (ROI-union mask), the ground truth is the voxel's real
session-shift aPRF (fwhm, amplitude, baseline), but with preferred values set by
a fixed lambda: the voxel keeps its real Misaligned (cdf) preferred value v and
its preferred value under the Aligned (inverse cdf) mapping is the one at the
same position on the lambda mix axis (``hierarchical_lambda``): lambda = 0 keeps
v, lambda = 1 keeps v's quantile. Simulated betas = prediction + Gaussian noise
with the voxel's real residual SD (the real session-shift fit). The simulated
betas are then fitted exactly as the real ones (session-shift aPRF; optionally
only odd or even runs), so the hierarchical model can be run on the result:
any departure of its estimate from the true lambda is a bias of the pipeline
(fitting noise, grid, censoring), not of the brain.

    python -m abstract_values.encoding_models.simulate_lambda_pipeline <subject> \
        --lam 0 [--runs odd|even] [--bids-folder ...]
    -> derivatives/encoding_models/sim-lam<lam>[-noise<k>]/aprf-session-shift[.runs-<half>]/sub-<S>/func/

--noise-scale k multiplies the noise SD: Gaussian noise at the residual SD gives
preferred values that scatter much less than the observed ones (real noise is
structured over runs and sessions), so k = 2, 3 bracket the observed scatter.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from abstract_values.encoding_models import hierarchical_lambda as hl
from abstract_values.encoding_models.fit_aprf import get_value_paradigm, save_f32
from abstract_values.encoding_models.fit_pipeline import fit_one_model
from abstract_values.encoding_models.model_specs import get_spec
from abstract_values.utils.data import BIDS_FOLDER, Subject


def aligned_mode(v, lam):
    """Inverse-cdf preferred value of a voxel whose cdf preferred value is v."""
    Fc = np.interp(v, np.r_[hl.LO, hl.VALUES['cdf'], hl.HI], np.r_[0, (np.arange(23) + .5) / 23, 1])
    pos = (1 - lam) * v + lam * (2 + 40 * Fc)
    ax = (1 - lam) * hl.V + lam * (2 + 40 * hl.quantile_fn('inv'))
    return hl.V[np.argmin(np.abs(ax[None, :] - pos[:, None]), 1)]


def simulate(subject, lam, bids_folder=BIDS_FOLDER, seed=0, noise_scale=1.):
    """Simulated betas (all trials, ROI-union voxels) with a known lambda.
    Returns sim (trials x voxels), paradigm (x, session), run, masker, truth, (k_cdf, k_inv)."""
    bids = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids)
    sessions = sorted(sub.get_sessions())
    spec = get_spec('session-shift')
    paradigm = get_value_paradigm(sub, sessions, True, with_run=True)
    run = paradigm.pop('run').to_numpy()
    mask = (bids / 'derivatives' / 'masks' / f'sub-{subject}' / 'anat'
            / f'sub-{subject}_space-T1w_desc-permnullunion_mask.nii.gz')
    masker = NiftiMasker(mask_img=str(mask)).fit()
    data = pd.DataFrame(masker.transform(
        sub.get_single_trial_estimates(sessions, desc='gabor', smoothed=False)).astype(np.float32))
    real = bids / 'derivatives' / 'encoding_models' / spec.out_subdir / f'sub-{subject}' / 'func'
    pars = pd.DataFrame({p: masker.transform(str(
        real / f'sub-{subject}_task-abstractvalue_space-T1w_desc-{p}_pe.nii.gz')).ravel()
        for p in spec.save_params}).astype(np.float32)
    pars.index = data.columns
    model = spec.cls(**spec.cls_kwargs)
    resid_sd = (data - model.predict(parameters=pars, paradigm=paradigm).values).std(0).values

    first = sub.get_mapping(sessions[0])
    k_cdf, k_inv = ('mode_1', 'mode_2') if first == 'cdf' else ('mode_2', 'mode_1')
    v = np.clip(pars[k_cdf].to_numpy(), hl.LO, hl.HI)
    truth = pars.copy()
    truth[k_cdf], truth[k_inv] = v, aligned_mode(v, lam)
    rng = np.random.default_rng([seed, int(''.join(c for c in subject if c.isdigit()))])
    sim = model.predict(parameters=truth, paradigm=paradigm).values
    sim = pd.DataFrame((sim + rng.standard_normal(sim.shape) * resid_sd * noise_scale)
                       .astype(np.float32))
    return sim, paradigm, run, masker, truth, (k_cdf, k_inv)


def simulate_corner(subject, lam_mu, lam_sigma, bids_folder=BIDS_FOLDER, seed=0,
                    noise_scale=1.):
    """Simulated betas from the position/width model (``MixPositionWidthGaussianPRF``) with
    known (lam_mu, lam_sigma), Gaussian tuning. Each voxel keeps its real Misaligned
    preferred value v and its real width in CHF at v; its response is scaled to the
    variance and mean of its real session-shift prediction; noise as in ``simulate``."""
    from abstract_values.encoding_models.models import MixPositionWidthGaussianPRF
    from abstract_values.encoding_models.model_specs import value_quantiles
    bids = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids)
    sessions = sorted(sub.get_sessions())
    spec = get_spec('session-shift')
    paradigm = get_value_paradigm(sub, sessions, True, with_run=True)
    run = paradigm.pop('run').to_numpy()
    mask = (bids / 'derivatives' / 'masks' / f'sub-{subject}' / 'anat'
            / f'sub-{subject}_space-T1w_desc-permnullunion_mask.nii.gz')
    masker = NiftiMasker(mask_img=str(mask)).fit()
    data = masker.transform(sub.get_single_trial_estimates(
        sessions, desc='gabor', smoothed=False)).astype(np.float32)
    real = bids / 'derivatives' / 'encoding_models' / spec.out_subdir / f'sub-{subject}' / 'func'
    pars = pd.DataFrame({p: masker.transform(str(
        real / f'sub-{subject}_task-abstractvalue_space-T1w_desc-{p}_pe.nii.gz')).ravel()
        for p in spec.save_params}).astype(np.float32)
    model = spec.cls(**spec.cls_kwargs)
    pred_real = model.predict(parameters=pars, paradigm=paradigm).values
    resid_sd = (data - pred_real).std(0)

    first = sub.get_mapping(sessions[0])
    k_cdf = 'mode_1' if first == 'cdf' else 'mode_2'
    v = np.clip(pars[k_cdf].to_numpy(), hl.LO, hl.HI)
    kv = np.r_[hl.LO, hl.VALUES['cdf'], hl.HI]
    kq = 2 + 40 * np.r_[0, (np.arange(23) + .5) / 23, 1]
    q_v = np.interp(v, kv, kq)
    slope = np.interp(v, kv, np.gradient(kq, kv))
    ses = paradigm['session'].to_numpy()
    mapping = np.array([sub.get_mapping(s) for s in sessions])[ses.astype(int)]
    par = pd.DataFrame({'x': paradigm['x'].to_numpy(),
                        'q': 2 + 40 * value_quantiles(paradigm['x'].to_numpy(), ses),
                        'm': (mapping != 'cdf').astype(float)}).astype(np.float32)
    truth = pd.DataFrame({'mode': (1 - lam_mu) * v + lam_mu * q_v,
                          'fwhm': pars['fwhm'].to_numpy() * ((1 - lam_sigma) + lam_sigma * slope),
                          'amplitude': 1., 'baseline': 0., 'lam_mu': lam_mu,
                          'lam_sigma': lam_sigma}).astype(np.float32)
    unit = MixPositionWidthGaussianPRF().predict(parameters=truth, paradigm=par).values
    sd_u = unit.std(0)
    amp = np.sign(pars['amplitude'].to_numpy()) * pred_real.std(0) / np.where(sd_u > 0, sd_u, 1)
    base = pred_real.mean(0) - amp * unit.mean(0)
    rng = np.random.default_rng([seed, int(''.join(c for c in subject if c.isdigit())), 7])
    sim = unit * amp + base + rng.standard_normal(unit.shape) * resid_sd * noise_scale
    return pd.DataFrame(sim.astype(np.float32)), masker


def simulated_betas(subject, lam, bids_folder=BIDS_FOLDER, seed=0, noise_scale=1.,
                    corner=None):
    """The simulated betas as a 4-D image (zeros outside the ROI-union mask), in GLMsingle
    beta order, as a drop-in for ``Subject.get_single_trial_estimates``. With ``corner``
    = (lam_mu, lam_sigma): from ``simulate_corner`` instead (``lam`` ignored)."""
    if corner is not None:
        sim, masker = simulate_corner(subject, *corner, bids_folder, seed, noise_scale)
    else:
        sim, _, _, masker, _, _ = simulate(subject, lam, bids_folder, seed, noise_scale)
    return masker.inverse_transform(sim.to_numpy())


def main(subject, lam, runs=None, bids_folder=BIDS_FOLDER, seed=0, n_iterations=1000,
         noise_scale=1.):
    bids = Path(bids_folder)
    em = bids / 'derivatives' / 'encoding_models'
    spec = get_spec('session-shift')
    sim, paradigm, run, masker, truth, (k_cdf, k_inv) = simulate(subject, lam, bids, seed,
                                                                  noise_scale)
    keep = np.ones(len(run), bool) if runs is None else run % 2 == (runs == 'odd')
    sim, par = sim[keep].reset_index(drop=True), paradigm[keep].reset_index(drop=True)
    print(f'sub-{subject} lambda {lam:g} runs {runs or "all"}: {len(par)} trials, '
          f'{sim.shape[1]} voxels', flush=True)
    fit, r2 = fit_one_model(spec, sim, par, float(par.x.min()), float(par.x.max()),
                            n_iterations=n_iterations)
    tag = f'sim-lam{lam:g}' + (f'-noise{noise_scale:g}' if noise_scale != 1 else '')
    out = (em / tag / (spec.out_subdir + (f'.runs-{runs}' if runs else ''))
           / f'sub-{subject}' / 'func')
    out.mkdir(parents=True, exist_ok=True)
    fn = f'sub-{subject}_task-abstractvalue_space-T1w_desc-{{}}_pe.nii.gz'
    for p in spec.save_params:
        save_f32(masker.inverse_transform(fit[p].to_numpy()), out / fn.format(p))
    save_f32(masker.inverse_transform(r2.to_numpy()), out / fn.format('r2'))
    for p in (k_cdf, k_inv):
        save_f32(masker.inverse_transform(truth[p].to_numpy()), out / fn.format(f'{p}_true'))
    print(f'  saved to {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--lam', type=float, default=0.)
    ap.add_argument('--runs', default=None, choices=['odd', 'even'])
    ap.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    ap.add_argument('--n-iterations', type=int, default=1000)
    ap.add_argument('--noise-scale', type=float, default=1.,
                    help='multiply the residual SD (output dir sim-lam<lam>-noise<scale>)')
    a = ap.parse_args()
    main(a.subject, a.lam, a.runs, a.bids_folder, n_iterations=a.n_iterations,
         noise_scale=a.noise_scale)
