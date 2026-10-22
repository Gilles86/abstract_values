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


def main(subject, bids_folder=BIDS_FOLDER, smoothed=False, targets=TARGETS,
         n_boot=1000, seed=0):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    par = get_paradigm(sub, sessions)
    betas = sub.get_single_trial_estimates(sessions, desc='gabor', smoothed=smoothed)
    folds = [((par['session'] == s) & (par['run'] == r)).to_numpy()
             for (s, r), _ in par.groupby(['session', 'run'])]
    code = {lam: tc.coding_regressor(stimulus_axis(par['value'].to_numpy(),
                                                   par['session'].to_numpy(),
                                                   f'mix:{lam}'))
            for lam in LAMBDAS}
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(folds), (n_boot, len(folds)))
    tag = '_smoothed' if smoothed else ''
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
        k0 = int(np.argmin(np.abs(LAMBDAS)))
        hat_rows.append(dict(roi=roi, lam_hat=lam_hat(prof),
                             ci_lo=np.percentile(bh, 2.5), ci_hi=np.percentile(bh, 97.5),
                             boot_sd=bh.std(), depth=prof.max() - prof[k0],
                             cvr2_max=prof.max(), n_voxels=Y.shape[1]))
        prof_rows += [dict(roi=roi, lam=l, cvr2=c) for l, c in zip(LAMBDAS, prof)]
        h = hat_rows[-1]
        print(f'  {roi}: {Y.shape[1]} voxels; lambda-hat {h["lam_hat"]:+.2f} '
              f'[{h["ci_lo"]:+.2f}, {h["ci_hi"]:+.2f}]', flush=True)
    meta = dict(subject=subject, smoothed=smoothed)
    pd.DataFrame(prof_rows).assign(**meta).to_csv(out / f'sub-{subject}{tag}_profile.tsv',
                                                  sep='\t', index=False)
    pd.DataFrame(hat_rows).assign(**meta).to_csv(out / f'sub-{subject}{tag}_lamhat.tsv',
                                                 sep='\t', index=False)
    print(f'saved to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject')
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--smoothed', action='store_true')
    p.add_argument('--targets', nargs='+', default=TARGETS)
    p.add_argument('--n-boot', type=int, default=1000)
    a = p.parse_args()
    main(a.subject, a.bids_folder, a.smoothed, a.targets, a.n_boot)
