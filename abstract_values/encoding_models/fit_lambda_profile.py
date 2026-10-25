"""Fine cvR^2 profile over the mix-lambda tuning axis, with a per-participant CI.

The coding model of ``connective_fields/fit_tuning_connectivity.py``
(y = b + a g(x_lambda(v)), Gaussian g, shape chosen per voxel on the training
runs, leave-one-run-out) on a fine lambda grid (-1 to 2, step 0.05) and a finer
shape grid. Per-voxel, per-run held-out SSE are kept, so the ROI-mean cvR^2
profile can be bootstrapped over runs: lambda-hat is the profile's argmax
(parabola through the peak and its neighbours), with a 95% bootstrap interval.

Output: derivatives/encoding_models/lambda_profile/sub-<S>/
  sub-<S>{_smoothed}_profile.tsv   ROI-mean cvR^2 per lambda
  sub-<S>{_smoothed}_lamhat.tsv    lambda-hat, bootstrap CI, profile depth

With --whole-brain: the same coding model in every brain-mask voxel on a
coarser grid (lambda -0.5 to 1.5, step 0.25), written as NIfTIs:
  sub-<S>{_smoothed}_desc-lambdacvr2_pe.nii.gz   4-D, one volume per lambda
  sub-<S>{_smoothed}_desc-lambdahat_pe.nii.gz    argmax (parabola through the peak)
  sub-<S>{_smoothed}_desc-lambdagain_pe.nii.gz   max cvR^2 - cvR^2(lambda = 0)

With --power: the same profile over a power (Box-Cox) value axis instead,
x = b_gamma(v) rescaled so that 2 and 42 CHF map to 2 and 42, with
b_gamma(v) = (v^gamma - 1) / gamma and b_0 = log v. gamma = 1 is the linear
value axis (= mix lambda 0), gamma = 0 a log (Weber) axis, gamma < 0 more
compressive still. Output sub-<S>{_smoothed}_gammaprofile.tsv / _gammahat.tsv.

With --bids: reported value instead. Only trials with a valid bid, and the
same Gaussian coding model on five stimulus axes:
  value      objective value (= mix lambda 0)
  quantile   orientation rank (= mix lambda 1)
  bid        the trial's own bid
  mean-bid   mean bid for that session and orientation (the behavioural
             mapping: biased, but stimulus-locked like value and quantile)
  bid-rank   mid-rank of the bid among the session's bids (2 + 40 q)
bid vs mean-bid asks whether the voxel follows the trial-to-trial deviation of
the report from its stimulus mean. Output sub-<S>{_smoothed}_bidaxes.tsv: ROI
mean cvR^2 per axis, plus the per-run bootstrap SD of each axis' difference
from mean-bid.

Usage:
  python -m abstract_values.encoding_models.fit_lambda_profile 05 --smoothed
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import abstract_values.connective_fields.fit_tuning_connectivity as tc
from abstract_values.connective_fields.gates import demean_runs, get_paradigm, load_roi
from abstract_values.encoding_models.model_specs import stimulus_axis
from abstract_values.utils.data import BIDS_FOLDER, Subject

LAMBDAS = np.round(np.arange(-1, 2.0001, .05), 2)
TARGETS = ['IPS0', 'IPS12', 'IPS345', 'V3ab', 'hV4']
tc.MODES = np.linspace(2, 42, 41)
tc.SIGMAS = np.array([3., 4., 5., 6., 8., 10., 12., 16., 24., 32.]) / (2 * np.sqrt(2 * np.log(2)))


def lam_hat(prof):
    """Argmax with a parabola through the peak and its neighbours."""
    k = int(np.argmax(prof))
    if 0 < k < len(prof) - 1:
        y0, y1, y2 = prof[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        off = .5 * (y0 - y2) / den if den < 0 else 0.
        return LAMBDAS[k] + off * (LAMBDAS[1] - LAMBDAS[0])
    return LAMBDAS[k]


def power_axis(v, gamma, lo=2., hi=42.):
    """Box-Cox value axis, rescaled so ``lo`` and ``hi`` keep their CHF position."""
    b = (lambda x: np.log(x)) if abs(gamma) < 1e-9 else (lambda x: (x ** gamma - 1) / gamma)
    return lo + (hi - lo) * (b(np.asarray(v, float)) - b(lo)) / (b(hi) - b(lo))


def bid_axes(par):
    """{axis name: x per trial} for the valid-bid trials in ``par``."""
    from scipy.stats import rankdata
    v, ses = par['value'].to_numpy(), par['session'].to_numpy()
    mean_bid = par.groupby(['session', 'orientation'])['bid'].transform('mean').to_numpy()
    rank = np.empty(len(par))
    for s in np.unique(ses):
        i = ses == s
        rank[i] = (rankdata(par['bid'].to_numpy()[i]) - .5) / i.sum()
    return {'value': v, 'quantile': stimulus_axis(v, ses, 'mix:1.0'),
            'bid': par['bid'].to_numpy(), 'mean-bid': mean_bid, 'bid-rank': 2 + 40 * rank}


LAMBDAS_BRAIN = np.round(np.arange(-.5, 1.5001, .25), 2)


def main_brain(subject, bids_folder=BIDS_FOLDER, smoothed=False, chunk=4000):
    import nibabel as nib
    from nilearn import image as nimage
    from nilearn.maskers import NiftiMasker
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    par = get_paradigm(sub, sessions)
    betas = sub.get_single_trial_estimates(sessions, desc='gabor', smoothed=smoothed)
    ref = nimage.index_img(betas, 0)
    mask = nimage.resample_to_img(sub.get_brain_mask(sessions[0]), ref,
                                  interpolation='nearest', force_resample=True,
                                  copy_header=True)
    mask = nimage.new_img_like(ref, (mask.get_fdata() > 0).astype(np.uint8))
    masker = NiftiMasker(mask).fit()
    Y = masker.transform(betas).astype(np.float64)
    ok = np.isfinite(Y).all(0) & (Y.std(0) > 0)
    Y = demean_runs(np.where(ok, Y, 0), par)
    folds = [((par['session'] == s) & (par['run'] == r)).to_numpy()
             for (s, r), _ in par.groupby(['session', 'run'])]
    code = {lam: tc.coding_regressor(stimulus_axis(par['value'].to_numpy(),
                                                   par['session'].to_numpy(),
                                                   f'mix:{lam}'))
            for lam in LAMBDAS_BRAIN}
    cv = np.full((len(LAMBDAS_BRAIN), Y.shape[1]), np.nan)
    idx = np.flatnonzero(ok)
    for c0 in range(0, len(idx), chunk):
        v = idx[c0:c0 + chunk]
        for k, lam in enumerate(LAMBDAS_BRAIN):
            cv[k, v] = tc.cv_fit(Y[:, v], [code[lam]], folds)
        print(f'  voxels {c0 + len(v)}/{len(idx)}', flush=True)
    k0 = int(np.argmin(np.abs(LAMBDAS_BRAIN)))
    step = LAMBDAS_BRAIN[1] - LAMBDAS_BRAIN[0]
    kk = np.nanargmax(np.where(np.isfinite(cv), cv, -np.inf), 0)
    hat = LAMBDAS_BRAIN[kk].astype(float)
    inner = (kk > 0) & (kk < len(LAMBDAS_BRAIN) - 1)
    j = np.flatnonzero(inner)
    y0, y1, y2 = cv[kk[j] - 1, j], cv[kk[j], j], cv[kk[j] + 1, j]
    den = y0 - 2 * y1 + y2
    hat[j] += np.where(den < 0, .5 * (y0 - y2) / np.where(den < 0, den, -1), 0) * step
    gain = np.nanmax(cv, 0) - cv[k0]
    hat[~ok] = np.nan
    tag = '_smoothed' if smoothed else ''
    out = bids_folder / 'derivatives' / 'encoding_models' / 'lambda_profile' / f'sub-{subject}'
    out.mkdir(parents=True, exist_ok=True)
    for name, arr in (('lambdacvr2', cv), ('lambdahat', hat[None]), ('lambdagain', gain[None])):
        img = masker.inverse_transform(arr.astype(np.float32))
        if arr.shape[0] == 1:
            img = nimage.index_img(img, 0)
        img.set_data_dtype(np.float32)
        img.header.set_slope_inter(slope=1, inter=0)
        img.to_filename(out / f'sub-{subject}{tag}_desc-{name}_pe.nii.gz')
    print(f'saved to {out}')


def main_bids(subject, bids_folder=BIDS_FOLDER, smoothed=False, targets=TARGETS,
              n_boot=1000, seed=0):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    par_all = get_paradigm(sub, sessions)
    keep = np.isfinite(par_all['bid'].to_numpy())
    par = par_all[keep].reset_index(drop=True)
    betas = sub.get_single_trial_estimates(sessions, desc='gabor', smoothed=smoothed)
    folds = [((par['session'] == s) & (par['run'] == r)).to_numpy()
             for (s, r), _ in par.groupby(['session', 'run'])]
    axes = bid_axes(par)
    code = {k: tc.coding_regressor(x) for k, x in axes.items()}
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(folds), (n_boot, len(folds)))
    rows = []
    for roi in targets:
        try:
            y, _ = load_roi(sub, betas, roi, 'aprf.cv', bids_folder, smoothed,
                            selection='all')
        except Exception as e:                       # missing mask
            print(f'  {roi}: skipped ({e})')
            continue
        Y = demean_runs(y[keep], par)
        dev2 = (Y - Y.mean(0)) ** 2
        tot = np.stack([dev2[f].sum(0) for f in folds])
        res = {k: np.stack([((Y - P)[f] ** 2).sum(0) for f in folds])
               for k, P in ((k, tc.cv_predict(Y, [c], folds)) for k, c in code.items())}
        cv = {k: np.mean(1 - r.sum(0) / tot.sum(0)) for k, r in res.items()}
        bcv = {k: np.array([np.mean(1 - r[b].sum(0) / tot[b].sum(0)) for b in boot])
               for k, r in res.items()}
        rows += [dict(roi=roi, axis=k, cvr2=cv[k],
                      boot_sd_vs_meanbid=np.std(bcv[k] - bcv['mean-bid']),
                      n_trials=len(par), n_dropped=int((~keep).sum()),
                      n_voxels=Y.shape[1]) for k in axes]
        print(f'  {roi}: ' + ', '.join(f'{k} {1e3 * c:+.2f}' for k, c in cv.items())
              + ' (x1e-3)', flush=True)
    tag = '_smoothed' if smoothed else ''
    out = bids_folder / 'derivatives' / 'encoding_models' / 'lambda_profile' / f'sub-{subject}'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).assign(subject=subject, smoothed=smoothed).to_csv(
        out / f'sub-{subject}{tag}_bidaxes.tsv', sep='\t', index=False)
    print(f'saved to {out}')


def main(subject, bids_folder=BIDS_FOLDER, smoothed=False, targets=TARGETS,
         n_boot=1000, seed=0, power=False, sim_lam=None, noise_scale=1.):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    par = get_paradigm(sub, sessions)
    if sim_lam is None:
        betas = sub.get_single_trial_estimates(sessions, desc='gabor', smoothed=smoothed)
    else:       # betas simulated with a known lambda (simulate_lambda_pipeline), unsmoothed
        from abstract_values.encoding_models.simulate_lambda_pipeline import simulated_betas
        betas = simulated_betas(subject, sim_lam, bids_folder, noise_scale=noise_scale)
    folds = [((par['session'] == s) & (par['run'] == r)).to_numpy()
             for (s, r), _ in par.groupby(['session', 'run'])]
    if power:
        code = {g: tc.coding_regressor(power_axis(par['value'].to_numpy(), g))
                for g in LAMBDAS}
    else:
        code = {lam: tc.coding_regressor(stimulus_axis(par['value'].to_numpy(),
                                                       par['session'].to_numpy(),
                                                       f'mix:{lam}'))
                for lam in LAMBDAS}
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(folds), (n_boot, len(folds)))
    tag = '_smoothed' if smoothed else ''
    if sim_lam is not None:
        tag += f'_sim-lam{sim_lam:g}-noise{noise_scale:g}'
    out = bids_folder / 'derivatives' / 'encoding_models' / 'lambda_profile' / f'sub-{subject}'
    out.mkdir(parents=True, exist_ok=True)
    prof_rows, hat_rows = [], []
    for roi in targets:
        try:
            y, _ = load_roi(sub, betas, roi, 'aprf.cv', bids_folder, smoothed,
                            selection='all')
        except Exception as e:                       # missing mask
            print(f'  {roi}: skipped ({e})')
            continue
        Y = demean_runs(y, par)
        dev2 = (Y - Y.mean(0)) ** 2
        tot = np.stack([dev2[f].sum(0) for f in folds])                  # F,V
        res = np.stack([np.stack([((Y - P)[f] ** 2).sum(0) for f in folds])
                        for P in (tc.cv_predict(Y, [code[l]], folds) for l in LAMBDAS)])
        prof = np.mean(1 - res.sum(1) / tot.sum(0), -1)                  # L
        bp = np.stack([np.mean(1 - res[:, b].sum(1) / tot[b].sum(0), -1) for b in boot])
        bh = np.array([lam_hat(p) for p in bp])
        k0 = int(np.argmin(np.abs(LAMBDAS - (1 if power else 0))))   # linear value axis
        hat_rows.append(dict(roi=roi, lam_hat=lam_hat(prof),
                             ci_lo=np.percentile(bh, 2.5), ci_hi=np.percentile(bh, 97.5),
                             boot_sd=bh.std(), depth=prof.max() - prof[k0],
                             cvr2_max=prof.max(), n_voxels=Y.shape[1]))
        prof_rows += [dict(roi=roi, lam=l, cvr2=c) for l, c in zip(LAMBDAS, prof)]
        h = hat_rows[-1]
        print(f'  {roi}: {Y.shape[1]} voxels; lambda-hat {h["lam_hat"]:+.2f} '
              f'[{h["ci_lo"]:+.2f}, {h["ci_hi"]:+.2f}]', flush=True)
    meta = dict(subject=subject, smoothed=smoothed)
    prof, hat = pd.DataFrame(prof_rows).assign(**meta), pd.DataFrame(hat_rows).assign(**meta)
    if power:       # same columns, gamma in place of lambda; depth is relative to gamma = 1
        prof = prof.rename(columns={'lam': 'gamma'})
        hat = hat.rename(columns={'lam_hat': 'gamma_hat'})
    kind = ('gammaprofile', 'gammahat') if power else ('profile', 'lamhat')
    prof.to_csv(out / f'sub-{subject}{tag}_{kind[0]}.tsv', sep='\t', index=False)
    hat.to_csv(out / f'sub-{subject}{tag}_{kind[1]}.tsv', sep='\t', index=False)
    print(f'saved to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject')
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--smoothed', action='store_true')
    p.add_argument('--targets', nargs='+', default=TARGETS)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--bids', action='store_true',
                   help='reported-value axes instead of the lambda profile')
    p.add_argument('--power', action='store_true',
                   help='profile over a power (Box-Cox) value axis instead of mix lambda')
    p.add_argument('--whole-brain', action='store_true',
                   help='per-voxel lambda profile in the whole brain mask (NIfTIs)')
    p.add_argument('--sim-lam', type=float, default=None,
                   help='profile of betas simulated with this lambda (simulate_lambda_pipeline); '
                        'output sub-<S>_sim-lam<lam>-noise<k>_{profile,lamhat}.tsv')
    p.add_argument('--noise-scale', type=float, default=1.)
    a = p.parse_args()
    if a.whole_brain:
        main_brain(a.subject, a.bids_folder, a.smoothed)
    elif a.bids:
        main_bids(a.subject, a.bids_folder, a.smoothed, a.targets, a.n_boot)
    else:
        main(a.subject, a.bids_folder, a.smoothed, a.targets, a.n_boot, power=a.power,
             sim_lam=a.sim_lam, noise_scale=a.noise_scale)
