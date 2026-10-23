"""Sample the whole-brain mix-lambda cvR^2 profile to fsaverage.

Input: ``sub-<S>{_smoothed}_desc-lambdacvr2_pe.nii.gz`` (4-D, one volume per
lambda in ``fit_lambda_profile.LAMBDAS_BRAIN``) from
``fit_lambda_profile.py --whole-brain``. Each volume is sampled to fsnative
(pial/white) and transformed to fsaverage; the per-subject result is one
array (n_lambda, n_vertices L + R):

    derivatives/encoding_models/lambda_profile/sub-<S>/
        sub-<S>{_smoothed}_space-fsaverage_desc-lambdacvr2.npy

Then ``--collect`` reduces all subjects to a group file (small; plot locally):
mean profile per vertex, the group lambda-hat (argmax of the mean profile),
the mean and SD of the per-participant lambda-hat, the participant t of cvR^2 at
every lambda and of cvR^2(lambda = 1) - cvR^2(lambda = 0). (The gain over
lambda = 0 is a max over the grid, so >= 0 by construction: its t is not a
test.)

    python -m abstract_values.surface.sample_lambda_to_surface 05
    python -m abstract_values.surface.sample_lambda_to_surface --collect \
        --out notes/data/lambda_brain_fsaverage.npz
"""
import argparse
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import image as nimage

from abstract_values.encoding_models.fit_lambda_profile import LAMBDAS_BRAIN
from abstract_values.surface.sampling import load_surface_data, sample_to_surfaces
from abstract_values.utils.data import BIDS_FOLDER


def lam_dir(bids):
    return Path(bids) / 'derivatives' / 'encoding_models' / 'lambda_profile'


def sample(subject, bids, smoothed):
    tag = '_smoothed' if smoothed else ''
    d = lam_dir(bids) / f'sub-{subject}'
    vol = nib.load(str(d / f'sub-{subject}{tag}_desc-lambdacvr2_pe.nii.gz'))
    out = []
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
        for k in range(vol.shape[-1]):
            fn = Path(tmp) / f'sub-{subject}_space-T1w_desc-lam{k}.nii.gz'
            img = nimage.index_img(vol, k)
            img.set_data_dtype(np.float32)
            img.header.set_slope_inter(slope=1, inter=0)
            img.to_filename(fn)
            surf = sample_to_surfaces(fn, subject, 1, Path(bids))
            out.append(np.concatenate([load_surface_data(surf[(h, 'fsaverage')])
                                       for h in ('L', 'R')]))
    out = np.stack(out).astype(np.float32)
    np.save(d / f'sub-{subject}{tag}_space-fsaverage_desc-lambdacvr2.npy', out)
    print(f'sub-{subject} smoothed={smoothed}: {out.shape}', flush=True)


def argmax_parabola(prof):
    """lambda at the profile maximum (axis 0), parabola through the peak."""
    L, step = LAMBDAS_BRAIN, LAMBDAS_BRAIN[1] - LAMBDAS_BRAIN[0]
    ok = np.isfinite(prof).all(0)
    kk = np.argmax(np.where(np.isfinite(prof), prof, -np.inf), 0)
    hat = L[kk].astype(float)
    j = np.flatnonzero(ok & (kk > 0) & (kk < len(L) - 1))
    y0, y1, y2 = prof[kk[j] - 1, j], prof[kk[j], j], prof[kk[j] + 1, j]
    den = y0 - 2 * y1 + y2
    hat[j] += np.where(den < 0, .5 * (y0 - y2) / np.where(den < 0, den, -1), 0) * step
    hat[~ok] = np.nan
    return hat


def collect(bids, out):
    k0 = int(np.argmin(np.abs(LAMBDAS_BRAIN)))
    res = dict(lambdas=LAMBDAS_BRAIN)
    for smoothed in (False, True):
        tag = '_smoothed' if smoothed else ''
        fs = sorted(lam_dir(bids).glob(f'sub-*/sub-*{tag}_space-fsaverage_desc-lambdacvr2.npy'))
        if not smoothed:
            fs = [f for f in fs if '_smoothed' not in f.name]
        P = np.stack([np.load(f) for f in fs])                   # S, L, V
        hats = np.stack([argmax_parabola(p) for p in P])          # S, V
        gain = P.max(1) - P[:, k0]
        n = np.isfinite(gain).sum(0)
        key = 'smoothed' if smoothed else 'unsmoothed'
        mean = np.nanmean(P, 0)
        tstat = lambda x: np.nanmean(x, 0) / (np.nanstd(x, 0, ddof=1) / np.sqrt(n))
        k1 = int(np.argmin(np.abs(LAMBDAS_BRAIN - 1)))
        res.update({
            f'{key}_mean_profile': mean.astype(np.float32),
            f'{key}_group_hat': argmax_parabola(mean).astype(np.float32),
            f'{key}_hat_mean': np.nanmean(hats, 0).astype(np.float32),
            f'{key}_hat_median': np.nanmedian(hats, 0).astype(np.float32),
            f'{key}_hat_sd': np.nanstd(hats, 0).astype(np.float32),
            f'{key}_gain_mean': np.nanmean(gain, 0).astype(np.float32),
            f'{key}_gain_t': (np.nanmean(gain, 0) / (np.nanstd(gain, 0, ddof=1)
                                                     / np.sqrt(n))).astype(np.float32),
            # unbiased (no per-subject max): cvR^2 > 0 per lambda, and quantile vs value
            f'{key}_profile_t': tstat(P).astype(np.float32),
            f'{key}_quantile_vs_value_t': tstat(P[:, k1] - P[:, k0]).astype(np.float32),
            f'{key}_maxcvr2_mean': np.nanmean(P.max(1), 0).astype(np.float32),
            f'{key}_n': n.astype(np.int16),
            f'{key}_subjects': np.array([f.name.split('_')[0][4:] for f in fs]),
        })
        print(f'{key}: {len(fs)} subjects', flush=True)
    np.savez_compressed(out, **res)
    print(f'wrote {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', nargs='?')
    ap.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    ap.add_argument('--collect', action='store_true')
    ap.add_argument('--out', default='notes/data/lambda_brain_fsaverage.npz')
    a = ap.parse_args()
    if a.collect:
        collect(a.bids_folder, a.out)
    else:
        for sm in (False, True):
            sample(a.subject, a.bids_folder, sm)
