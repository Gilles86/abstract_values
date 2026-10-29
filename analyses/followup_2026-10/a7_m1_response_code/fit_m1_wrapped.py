#!/usr/bin/env python3
"""Wrapped (axial von Mises) PRF on value and on bid in M1, with parameters kept.

The A1 motor fits (fit_motor_cv.py) saved cvR² only, and their wrapped model
wraps the orientation rank, not value. Here the value / bid axis itself is
wrapped, x = pi * v / 42, so 0 and 42 CHF are neighbours. A voxel that tracks
expected slider travel ((b² + (42 - b)²) / 84, U-shaped in the bid because the
marker starts uniformly on [0, 42]) is then a von Mises centred on the seam
(mu = 0 = pi), and the preferred-value distribution should pile up there in
contralateral (left) M1 and not in ipsilateral (right) M1.

Per subject x epoch (GLMsingle 'gabor' or 'response' betas), on the trials
with a valid bid, for space in (value, bid):

    wrapped   full-data fit -> mu, kappa, amplitude, baseline, r2
              + leave-one-run-out cvR²
    gaussian  full-data fit on the unwrapped CHF axis -> mode, fwhm, ... r2
    null      leave-one-run-out cvR² of the training mean (shared by both)

Output, one TSV per subject x epoch, one row per (voxel, space):
``derivatives/followup_2026-10/a7_m1_response_code/fits/
sub-<S>_epoch-<E>_m1wrapped.tsv``. Columns: voxel, hemi (M1l / M1r), hand
(top 25 % mean response-epoch beta within the hemisphere, as in A1), space,
wrapped_*, gaussian_*, cvr2_wrapped, cvr2_null. No existing derivative is
written to.

    python fit_m1_wrapped.py 10 --epoch response
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.utils import get_rsq

from abstract_values.encoding_models.fit_aprf_cv import get_paradigm
from abstract_values.encoding_models.fit_pipeline import fit_one_model
from abstract_values.encoding_models.model_specs import MODEL_SPECS
from abstract_values.utils.data import Subject, BIDS_FOLDER

V_MAX = 42.0
HAND_Q = 0.75
OUT = Path('derivatives') / 'followup_2026-10' / 'a7_m1_response_code' / 'fits'
WRAPPED = MODEL_SPECS['quantile-wrapped']   # AxialVonMisesPRF, log-kappa grid
GAUSSIAN = MODEL_SPECS['gaussian']


def load_hemi(sub, hemi, betas):
    mask = Path(sub.bids_folder) / 'derivatives' / 'masks' / f'sub-{sub.subject_id}' / \
        'anat' / f'sub-{sub.subject_id}_space-T1w_desc-{hemi}_mask.nii.gz'
    masker = NiftiMasker(mask_img=str(mask), target_affine=betas.affine,
                         target_shape=betas.shape[:3]).fit()
    return masker.transform(betas).astype(np.float32)


def fit(spec, data, x, **kw):
    par = pd.DataFrame({'x': x.astype(np.float32)})
    return fit_one_model(spec, data, par, value_min=float(x.min()),
                         value_max=float(x.max()), **kw)


def loro_cvr2(data, x, folds, n_iterations):
    cv_w, cv_0 = [], []
    for f in np.unique(folds):
        te = folds == f
        tr_d = data[~te].reset_index(drop=True)
        te_d = data[te].reset_index(drop=True)
        pars, _ = fit(WRAPPED, tr_d, x[~te], n_iterations=n_iterations,
                      log_prefix='    ')
        model = WRAPPED.cls(**WRAPPED.cls_kwargs)
        pred = np.asarray(model.predict(parameters=pars,
                                        paradigm=pd.DataFrame({'x': x[te].astype(np.float32)})))
        cv_w.append(get_rsq(te_d, pd.DataFrame(pred, columns=te_d.columns)).to_numpy())
        null = np.tile(tr_d.mean(0).to_numpy(), (len(te_d), 1))
        cv_0.append(get_rsq(te_d, pd.DataFrame(null, columns=te_d.columns)).to_numpy())
    return np.mean(cv_w, 0), np.mean(cv_0, 0)


def main(subject, epoch, bids_folder, n_iterations, skip_cv=False):
    bids_folder = Path(bids_folder)
    out = bids_folder / OUT / f'sub-{subject}_epoch-{epoch}_m1wrapped.tsv'
    out.parent.mkdir(parents=True, exist_ok=True)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())

    # get_paradigm clips bids below at 1 CHF (log-Gaussian legacy); undo nothing,
    # the 0-1 CHF difference is 0.07 rad on the wrapped axis
    p = get_paradigm(sub, sessions, needs_session=False, space='value')
    keep = p['bid'].notna().to_numpy()
    folds = (p.index.get_level_values('session').to_numpy() * 100
             + p.index.get_level_values('run').to_numpy())[keep]
    axes = {'value': p['x'].to_numpy()[keep], 'bid': p['bid'].to_numpy()[keep]}

    betas = sub.get_single_trial_estimates(sessions, desc=epoch)
    resp = sub.get_single_trial_estimates(sessions, desc='response')
    assert betas.shape[-1] == len(p), (betas.shape, len(p))
    blocks = []
    for hemi in ('M1l', 'M1r'):
        Y = load_hemi(sub, hemi, betas)
        R = load_hemi(sub, hemi, resp).mean(0)
        ok = np.isfinite(Y).all(0) & (Y.std(0) > 0)
        hand = R >= np.quantile(R[ok], HAND_Q)
        blocks.append((hemi, Y[:, ok], hand[ok]))
    data = pd.DataFrame(np.concatenate([b[1] for b in blocks], 1)[keep])
    meta = pd.DataFrame({'voxel': np.arange(data.shape[1]),
                         'hemi': np.concatenate([[b[0]] * b[1].shape[1] for b in blocks]),
                         'hand': np.concatenate([b[2] for b in blocks])})
    print(f'sub-{subject} epoch={epoch}: {len(data)} trials, {data.shape[1]} voxels '
          f'({(meta.hemi == "M1l").sum()} M1l)', flush=True)

    rows = []
    for space, v in axes.items():
        xw = np.pi * np.clip(v, 0, V_MAX) / V_MAX
        print(f'\n== {space}: wrapped full fit', flush=True)
        pw, r2w = fit(WRAPPED, data, xw, n_iterations=n_iterations)
        print(f'== {space}: gaussian full fit', flush=True)
        pg, r2g = fit(GAUSSIAN, data, v, n_iterations=n_iterations)
        df = meta.assign(space=space)
        for c in pw.columns:
            df[f'wrapped_{c}'] = pw[c].to_numpy()
        df['wrapped_mu_chf'] = (pw['mu'].to_numpy() % np.pi) / np.pi * V_MAX
        df['wrapped_r2'] = r2w.to_numpy()
        for c in pg.columns:
            df[f'gaussian_{c}'] = pg[c].to_numpy()
        df['gaussian_r2'] = r2g.to_numpy()
        if not skip_cv:
            print(f'== {space}: wrapped LORO cv', flush=True)
            df['cvr2_wrapped'], df['cvr2_null'] = loro_cvr2(data, xw, folds, n_iterations)
        rows.append(df)
    pd.concat(rows).assign(subject=subject, epoch=epoch).to_csv(out, sep='\t', index=False)
    print(f'saved {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--epoch', default='response', choices=['gabor', 'response'])
    ap.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    ap.add_argument('--n-iterations', type=int, default=1000)
    ap.add_argument('--skip-cv', action='store_true')
    a = ap.parse_args()
    main(a.subject, a.epoch, a.bids_folder, a.n_iterations, a.skip_cv)
