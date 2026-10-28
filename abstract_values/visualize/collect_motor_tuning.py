#!/usr/bin/env python3
"""Motor-cortex tuning: trial table + voxelwise cvR² summary (run on the cluster).

Per MRI subject this writes, from the GLMsingle single-trial betas of both
epochs (gabor, response):

1. ``--trials-tsv``: one row per (subject, trial, epoch, roi) with the ROI-mean
   beta and the trial's value, orientation, bid, RT and mapping.
2. ``--cv-tsv``: one row per (subject, epoch, roi, model) with the mean over ROI
   voxels of leave-one-run-out cvR² for closed-form linear regressions
   (numpy, same folds as fit_aprf_cv), all on the bid-valid trials:

       null          intercept
       value         objective CHF
       bid           reported CHF (≈ P(rightward movement) = bid / 42)
       extent        expected |movement| = E|bid − U(0, 42)| = (b² + (42 − b)²) / 84
                     (the slider start is uniform on [0, 42] and not logged)
       bid+extent
       rt            response time
       bid+rt
       orientation   cos 2θ, sin 2θ
       rank          orientation rank (= within-session quantile)

3. ``motor_tuning_prf_cv.tsv``: mean over ROI voxels of the PRF-model cvR² from
   ``derivatives/followup_2026-10/a1_motor_stage/encoding_models[_preSMA]/``
   (fit_motor_cv.py), when present.
4. ``motor_tuning_decoding.tsv``: leave-one-run-out ridge decoding (RidgeCV,
   nested alpha) of bid, value and expected travel from each ROI's voxels;
   per (subject, epoch, roi, target) Pearson r and RMSE of decoded vs true.

ROIs: M1l, M1r (pycortex fsaverage M1 projected to T1w), their 'hand' subsets
(the 25 % of ROI voxels with the largest mean response-epoch beta: a movement
localiser that ignores trial-to-trial modulation), pre-SMA (control; see
analyses/followup_2026-10/a1_motor_stage/make_presma_mask.py), bilateral Benson V1,
IPS1–2 and IPS3–5.

    python collect_motor_tuning.py --out-dir ~/git/abstract_values/notes/data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image

from abstract_values.utils.data import Subject

BIDS = Path('/shares/zne.uzh/gdehol/ds-abstractvalue')
A1 = BIDS / 'derivatives' / 'followup_2026-10' / 'a1_motor_stage'
SUBJECTS = ['pil01', 'pil02'] + [f'{i:02d}' for i in range(3, 36)]
HAND_Q = 0.75

PRF_MODELS = {  # name -> (cv dir, null dir)
    'gaussian': ('aprf-gauss.cv', 'aprf-null.cv'),
    'linear': ('aprf-linear.cv', 'aprf-null.cv'),
    'quantile': ('aprf-quantile.cv', 'aprf-null.cv'),
    'quantile-wrapped': ('aprf-quantile-wrapped.cv', 'aprf-null.cv'),
    'vonmises-prf': ('vonmises-prf.cv', 'aprf-null.cv'),
    'gaussian-bid': ('aprf-gauss-bid.cv', 'aprf-null-bidmatched.cv'),
    'gaussian-bidmatched': ('aprf-gauss-bidmatched.cv', 'aprf-null-bidmatched.cv'),
    'linear-bid': ('aprf-linear-bid.cv', 'aprf-null-bidmatched.cv'),
    'linear-bidmatched': ('aprf-linear-bidmatched.cv', 'aprf-null-bidmatched.cv'),
    'quantile-bidmatched': ('aprf-quantile-bidmatched.cv', 'aprf-null-bidmatched.cv'),
    'quantile-wrapped-bidmatched': ('aprf-quantile-wrapped-bidmatched.cv',
                                    'aprf-null-bidmatched.cv'),
}


def paradigm(sub):
    rows = []
    for session in sorted(sub.get_sessions()):
        runs = sub.get_runs(session)
        ev = sub.get_events(session, runs)
        mapping = sub.get_mapping(session)
        for run in runs:
            r = ev.loc[run].reset_index().sort_values('onset')
            resp = r[r.event_type == 'response_bar'].set_index('trial_nr')
            for _, g in r[r.event_type == 'gabor'].iterrows():
                t = g.trial_nr
                bid = float(resp.bid.get(t, np.nan))
                rt = float(resp.duration.get(t, np.nan)) if np.isfinite(bid) else np.nan
                rows.append(dict(session=session, run=run, trial_nr=t, mapping=mapping,
                                 value=float(g.value), orientation=float(g.orientation),
                                 bid=bid, rt=rt))
    return pd.DataFrame(rows)


def masks(sub_id, ref):
    anat = BIDS / 'derivatives' / 'masks' / f'sub-{sub_id}' / 'anat'
    files = {'M1l': f'sub-{sub_id}_space-T1w_desc-M1l_mask.nii.gz',
             'M1r': f'sub-{sub_id}_space-T1w_desc-M1r_mask.nii.gz',
             'V1': f'sub-{sub_id}_space-T1w_hemi-LR_desc-BensonV1_mask.nii.gz',
             'IPS12': f'sub-{sub_id}_space-T1w_desc-IPS12_mask.nii.gz',
             'IPS345': f'sub-{sub_id}_space-T1w_desc-IPS345_mask.nii.gz'}
    files = {k: anat / f for k, f in files.items()}
    a1 = A1 / 'masks' / f'sub-{sub_id}' / 'anat'
    for k in ('preSMA', 'preSMAl', 'preSMAr'):
        files[k] = a1 / f'sub-{sub_id}_space-T1w_desc-{k}_mask.nii.gz'
    out = {}
    for k, f in files.items():
        if not f.exists():
            print(f'  missing mask {f}')
            continue
        m = image.resample_to_img(str(f), ref, interpolation='nearest',
                                  force_resample=True, copy_header=True)
        out[k] = np.asarray(m.dataobj) > 0
    return out


def extent(b):
    """E|b − U(0, 42)|: expected slider travel from the (unlogged) uniform start."""
    return (b ** 2 + (42 - b) ** 2) / 84


def design(p, model):
    one = np.ones(len(p))
    b, v = p.bid.to_numpy(), p.value.to_numpy()
    ext = extent(b)
    th = np.deg2rad(p.orientation.to_numpy()) * 2
    rank = p.groupby('session').orientation.rank().to_numpy()
    cols = {'null': [], 'value': [v], 'bid': [b], 'extent': [ext], 'bid+extent': [b, ext],
            'rt': [p.rt.to_numpy()], 'bid+rt': [b, p.rt.to_numpy()],
            'orientation': [np.cos(th), np.sin(th)], 'rank': [rank]}[model]
    return np.column_stack([one] + cols)


def decode(Y, y, folds):
    """LORO ridge decoding (voxels standardised on the training runs, nested RidgeCV)."""
    from sklearn.linear_model import RidgeCV
    pred = np.full(len(y), np.nan)
    for f in np.unique(folds):
        te = folds == f
        mu, sd = Y[~te].mean(0), Y[~te].std(0) + 1e-6
        r = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit((Y[~te] - mu) / sd, y[~te])
        pred[te] = r.predict((Y[te] - mu) / sd)
    return pred


def cv_r2(X, Y, folds):
    """Mean over folds of per-voxel test R² (1 − SSres / SStot about the test mean)."""
    out = []
    for f in np.unique(folds):
        te = folds == f
        beta = np.linalg.lstsq(X[~te], Y[~te], rcond=None)[0]
        res = Y[te] - X[te] @ beta
        tot = Y[te] - Y[te].mean(0)
        out.append(1 - (res ** 2).sum(0) / (tot ** 2).sum(0))
    return np.mean(out, 0)


def main(out_dir):
    out_dir = Path(out_dir)
    trial_rows, cv_rows, prf_rows, dec_rows = [], [], [], []
    for s in SUBJECTS:
        sub = Subject(s, bids_folder=BIDS)
        try:
            p = paradigm(sub)
            betas = {d: sub.get_single_trial_estimates(sorted(sub.get_sessions()), desc=d)
                     for d in ('gabor', 'response')}
        except Exception as e:  # noqa: BLE001
            print(f'sub-{s}: skipped ({e})')
            continue
        ref = image.index_img(betas['gabor'], 0)
        assert betas['gabor'].shape[-1] == len(p) == betas['response'].shape[-1], s
        rois = masks(s, ref)
        data = {d: np.asarray(b.dataobj, dtype=np.float32) for d, b in betas.items()}
        mean_resp = data['response'].mean(-1)
        for h in ('M1l', 'M1r'):
            m = rois[h]
            thr = np.quantile(mean_resp[m], HAND_Q)
            rois[f'{h}hand'] = m & (mean_resp >= thr)
        valid = p.bid.notna().to_numpy() & p.rt.notna().to_numpy()
        pv = p[valid].reset_index(drop=True)
        folds = (pv.session * 100 + pv.run).to_numpy()
        for d, arr in data.items():
            for roi, m in rois.items():
                Y = arr[m].T                                      # trials x voxels
                Y = Y[:, np.isfinite(Y).all(0) & (Y.std(0) > 0)]
                trial_rows.append(p.assign(subject=s, epoch=d, roi=roi,
                                           beta=Y.mean(1), n_vox=Y.shape[1]))
                Yv = Y[valid]
                for model in ('null', 'value', 'bid', 'extent', 'bid+extent', 'rt',
                              'bid+rt', 'orientation', 'rank'):
                    r2 = cv_r2(design(pv, model), Yv, folds)
                    cv_rows.append(dict(subject=s, epoch=d, roi=roi, model=model,
                                        cvr2=float(np.nanmean(r2)), n_vox=Y.shape[1]))
                for target in ('bid', 'value', 'ext'):
                    yt = (pv.bid.to_numpy() if target == 'bid' else pv.value.to_numpy()
                          if target == 'value' else extent(pv.bid.to_numpy()))
                    yp = decode(Yv, yt, folds)
                    dec_rows.append(dict(subject=s, epoch=d, roi=roi, target=target,
                                         r=float(np.corrcoef(yp, yt)[0, 1]),
                                         rmse=float(np.sqrt(np.mean((yp - yt) ** 2))),
                                         sd_true=float(yt.std()), n_vox=Y.shape[1]))
                root = A1 / ('encoding_models_preSMA' if roi.startswith('preSMA')
                             else 'encoding_models')
                for model, (cvd, nulld) in PRF_MODELS.items():
                    sfx = '' if d == 'gabor' else '-response'
                    fn = lambda dd: (root / (dd + sfx)
                                     / f'sub-{s}' / 'func'
                                     / f'sub-{s}_task-abstractvalue_space-T1w_desc-cvr2_pe.nii.gz')
                    if not (fn(cvd).exists() and fn(nulld).exists()):
                        continue
                    a = np.asarray(nib.load(str(fn(cvd))).dataobj, dtype=np.float32)
                    n = np.asarray(nib.load(str(fn(nulld))).dataobj, dtype=np.float32)
                    mm = m & (a != 0)
                    prf_rows.append(dict(subject=s, epoch=d, roi=roi, model=model,
                                         cvr2=float(a[mm].mean()),
                                         cvr2_null=float(n[mm].mean()),
                                         n_vox=int(mm.sum()), n_unique=len(np.unique(a[mm]))))
        print(f'sub-{s}: done ({len(p)} trials, {valid.sum()} valid)', flush=True)

    pd.concat(trial_rows).to_csv(out_dir / 'motor_tuning_trials.tsv', sep='\t', index=False)
    pd.DataFrame(cv_rows).to_csv(out_dir / 'motor_tuning_linear_cv.tsv', sep='\t', index=False)
    pd.DataFrame(prf_rows).to_csv(out_dir / 'motor_tuning_prf_cv.tsv', sep='\t', index=False)
    pd.DataFrame(dec_rows).to_csv(out_dir / 'motor_tuning_decoding.tsv', sep='\t', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args().out_dir)
