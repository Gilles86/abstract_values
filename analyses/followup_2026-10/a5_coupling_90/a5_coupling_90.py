#!/usr/bin/env python3
"""A5 (exploratory): is V1 -> IPS residual coupling stronger for 90-deg-preferring V1 voxels?

Builds on the residual-coupling analysis in ``abstract_values/connective_fields``
(``gates.residualise``, ``gates.load_roi``): per session, every voxel's single-trial
GLMsingle betas are residualised on one dummy per orientation (23), one per run and
the orientations of the +-1 neighbouring trials (HRF overlap). What remains is
trial-to-trial fluctuation around the stimulus-locked means.

V1 (Benson V1, 0.75-3.75 deg annulus) voxels are binned by preferred orientation,
taken from the joint (both-session) axial von Mises PRF fit
(``derivatives/encoding_models/vonmises-prf``, ``desc-mu``). 8 bins of 22.5 deg,
centred on 0, 22.5, ..., 157.5; the 90-deg bin is [78.75, 101.25). Labels come from
one fit on both sessions, so bin membership is identical under both mappings.

Coupling of bin k with an IPS segment, per session (= mapping):
  raw       c_k = mean residual over the bin's V1 voxels. Coupling = mean over IPS
            voxels of Fisher-z r(c_k, IPS voxel residual). Includes the V1-wide
            shared fluctuation.
  specific  d_k = c_k - mean residual over ALL V1 voxels in the selection (removes
            the shared fluctuation, as the channels in test_coupling do). Coupling =
            mean over IPS voxels of r(d_k, IPS voxel)^2 x 100 (sign of a voxel's
            coupling to a channel is free, so squared). Under no coupling
            E[r^2] = 1/(T-1) for every bin, whatever its size.

Contrasts: delta90 = C[90 bin] - mean C[other 7 bins]; delta0 = same for the 0-deg
(horizontal) bin, to ask whether anything at 90 is specific to vertical or cardinal.

Nulls (Saurabh's point: a bin with more voxels gets a less noisy average and so
more coupling by sampling alone):
  perm  preferred orientations permuted across the V1 voxels (marginal distribution,
        hence bin counts, preserved), N_PERM times; delta under each permutation.
  sub   every bin subsampled to the smallest bin's voxel count, N_SUB draws;
        coupling averaged over draws.

Outputs: derivatives/followup_2026-10/a5_coupling_90/sub-<S>/
  sub-<S>_desc-bins.tsv        coupling per (session, v1_sel, segment, ips_sel, metric, bin)
  sub-<S>_desc-contrasts.tsv   delta observed / subsampled / perm mean, sd, p
  sub-<S>_desc-permnull.tsv.gz the permutation deltas (for the group null)
  sub-<S>_desc-counts.tsv      voxels per bin

Usage:
  python analyses/followup_2026-10/a5_coupling_90/a5_coupling_90.py 12 \
      --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.maskers import NiftiMasker

from abstract_values.connective_fields.gates import get_paradigm, load_roi, residualise, zscore
from abstract_values.utils.data import Subject, BIDS_FOLDER

N_BINS = 8
WIDTH = 180. / N_BINS
CENTRES = np.arange(N_BINS) * WIDTH
BIN90, BIN0 = int(round(90 / WIDTH)), 0
SEGMENTS = {'IPS0': 'IPS0', 'IPS12': 'IPS12', 'IPS345': 'IPS345',
            'IPS0-5': 'IPS0+IPS12+IPS345'}
V1_ROI = 'BensonV1ecc075-375'


def bin_index(pref_deg):
    return np.floor(((pref_deg + WIDTH / 2) % 180) / WIDTH).astype(int)


def load_pref(sub, betas, bids_folder):
    """Preferred orientation (deg, 0-180) of every usable V1 voxel, from vonmises-prf mu."""
    mask = sub.get_roi_mask(V1_ROI, hemi=None)
    mask = image.resample_to_img(mask, image.index_img(betas, 0), interpolation='nearest')
    masker = NiftiMasker(mask_img=mask).fit()
    y = masker.transform(betas)
    ok = np.isfinite(y).all(0) & (y.std(0) > 0)          # same rule as gates.load_roi
    fn = (Path(bids_folder) / 'derivatives' / 'encoding_models' / 'vonmises-prf'
          / f'sub-{sub.subject_id}' / 'func'
          / f'sub-{sub.subject_id}_task-abstractvalue_space-T1w_desc-mu_pe.nii.gz')
    mu = masker.transform(image.load_img(str(fn))).ravel()[ok]
    print(f'  vonmises-prf mu range {np.nanmin(mu):.3f} .. {np.nanmax(mu):.3f} (rad)')
    return np.rad2deg(mu) % 180.


def coupling(res_v1, idx_mat, res_ips_z, metric, v1_mean=None):
    """Bin coupling for a stack of bin assignments.

    res_v1: T x V; idx_mat: P x V integer bins (one row per assignment);
    res_ips_z: T x N z-scored IPS residuals. Returns P x N_BINS."""
    T = res_v1.shape[0]
    out = np.empty((len(idx_mat), N_BINS))
    for p, idx in enumerate(idx_mat):
        ind = np.zeros((len(idx), N_BINS))
        ind[np.arange(len(idx)), idx] = 1
        ind /= np.maximum(ind.sum(0, keepdims=True), 1)
        c = res_v1 @ ind                                          # T x bins
        if metric == 'specific':
            c = c - v1_mean[:, None]
        r = zscore(c).T @ res_ips_z / T                           # bins x N
        if metric == 'raw':
            out[p] = np.arctanh(np.clip(r, -.999, .999)).mean(1)
        else:
            out[p] = 100 * (r ** 2).mean(1)
    return out


def subsample_assignments(idx, n_sub, n_draw, rng):
    """Assignments where every bin keeps n_sub random members, the rest get -1."""
    rows = []
    members = [np.flatnonzero(idx == k) for k in range(N_BINS)]
    for _ in range(n_draw):
        a = np.full(len(idx), -1)
        for k, m in enumerate(members):
            a[rng.choice(m, n_sub, replace=False)] = k
        rows.append(a)
    return np.array(rows)


def coupling_sub(res_v1, assign, res_ips_z, metric, v1_mean):
    """``coupling`` for assignments with excluded (-1) voxels."""
    T = res_v1.shape[0]
    out = np.empty((len(assign), N_BINS))
    for p, a in enumerate(assign):
        keep = a >= 0
        ind = np.zeros((keep.sum(), N_BINS))
        ind[np.arange(keep.sum()), a[keep]] = 1
        ind /= ind.sum(0, keepdims=True)
        c = res_v1[:, keep] @ ind
        if metric == 'specific':
            c = c - v1_mean[:, None]
        r = zscore(c).T @ res_ips_z / T
        out[p] = (np.arctanh(np.clip(r, -.999, .999)).mean(1) if metric == 'raw'
                  else 100 * (r ** 2).mean(1))
    return out


def contrast(c, k):
    others = [j for j in range(N_BINS) if j != k]
    return c[..., k] - c[..., others].mean(-1)


def main(subject, bids_folder=BIDS_FOLDER, n_perm=1000, n_sub=200, lags=1, seed=0,
         min_bin=10, out_root=None):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    if len(sessions) != 2:
        raise SystemExit(f'sub-{subject}: needs 2 sessions, has {sessions}')
    par = get_paradigm(sub, sessions)
    betas = sub.get_single_trial_estimates(sessions, desc='gabor')
    if betas.shape[3] != len(par):
        raise SystemExit(f'{betas.shape[3]} betas vs {len(par)} trials')

    y_v1, sel_v1 = load_roi(sub, betas, V1_ROI, 'vonmises-prf.cv', bids_folder, False)
    pref = load_pref(sub, betas, bids_folder)
    assert len(pref) == y_v1.shape[1]
    ips = {}
    for seg, roi in SEGMENTS.items():
        y, sel = load_roi(sub, betas, roi, 'aprf.cv', bids_folder, False)
        ips[seg] = (y, sel)

    v1_sets = {'tuned': sel_v1 & np.isfinite(pref), 'all': np.isfinite(pref)}
    rng = np.random.default_rng(seed + sum(map(ord, str(subject))))
    rows_bins, rows_con, rows_null, rows_cnt = [], [], [], []

    for v1_name, keep in v1_sets.items():
        idx = bin_index(pref[keep])
        counts = np.bincount(idx, minlength=N_BINS)
        for k in range(N_BINS):
            rows_cnt.append({'v1_sel': v1_name, 'bin': k, 'centre': CENTRES[k],
                             'n_vox': int(counts[k])})
        n_min = int(counts.min())
        perms = np.array([rng.permutation(idx) for _ in range(n_perm)])
        subs = (subsample_assignments(idx, n_min, n_sub, rng) if n_min >= min_bin else None)
        print(f'  V1 {v1_name}: {keep.sum()} voxels, bins {counts.tolist()}, '
              f'subsample n = {n_min if subs is not None else "skipped"}')

        for s in sessions:
            ses = (par['session'] == s).to_numpy()
            p_ses = par[ses].reset_index(drop=True)
            cond = p_ses['condition'].iloc[0]
            res_v1 = residualise(y_v1[ses][:, keep], p_ses, lags)
            v1_mean = res_v1.mean(1)
            for seg, (y_ips, sel_ips) in ips.items():
                res_ips = residualise(y_ips[ses], p_ses, lags)
                for ips_name, cols in (('all', np.ones(res_ips.shape[1], bool)),
                                       ('aprf', sel_ips)):
                    if cols.sum() < 10:
                        continue
                    rz = zscore(res_ips[:, cols])
                    for metric in ('raw', 'specific'):
                        obs = coupling(res_v1, idx[None], rz, metric, v1_mean)[0]
                        null = coupling(res_v1, perms, rz, metric, v1_mean)
                        sb = (coupling_sub(res_v1, subs, rz, metric, v1_mean).mean(0)
                              if subs is not None else np.full(N_BINS, np.nan))
                        key = dict(session=s, condition=cond, v1_sel=v1_name, segment=seg,
                                   ips_sel=ips_name, metric=metric)
                        for k in range(N_BINS):
                            rows_bins.append({**key, 'bin': k, 'centre': CENTRES[k],
                                              'n_vox': int(counts[k]), 'coupling': obs[k],
                                              'coupling_sub': sb[k],
                                              'coupling_perm': null[:, k].mean()})
                        for cname, kb in (('delta90', BIN90), ('delta0', BIN0)):
                            d_obs, d_null, d_sub = (contrast(obs, kb), contrast(null, kb),
                                                    contrast(sb, kb))
                            rows_con.append({**key, 'contrast': cname, 'delta_obs': d_obs,
                                             'delta_perm_mean': d_null.mean(),
                                             'delta_perm_sd': d_null.std(),
                                             'delta_obs_minus_perm': d_obs - d_null.mean(),
                                             'p_perm': (1 + (d_null >= d_obs).sum()) / (1 + n_perm),
                                             'delta_sub': d_sub, 'n_sub': n_min,
                                             'n_ips': int(cols.sum()), 'n_v1': int(keep.sum())})
                            rows_null.append(pd.DataFrame({**key, 'contrast': cname,
                                                           'perm': np.arange(n_perm),
                                                           'delta': d_null}))
            print(f'    ses-{s} ({cond}) done')

    out = (Path(out_root) if out_root else
           bids_folder / 'derivatives' / 'followup_2026-10' / 'a5_coupling_90') / f'sub-{subject}'
    out.mkdir(parents=True, exist_ok=True)
    pre = out / f'sub-{subject}'
    pd.DataFrame(rows_bins).assign(subject=subject).to_csv(f'{pre}_desc-bins.tsv', sep='\t', index=False)
    con = pd.DataFrame(rows_con).assign(subject=subject)
    con.to_csv(f'{pre}_desc-contrasts.tsv', sep='\t', index=False)
    pd.concat(rows_null).assign(subject=subject).to_csv(f'{pre}_desc-permnull.tsv.gz', sep='\t',
                                                        index=False)
    pd.DataFrame(rows_cnt).assign(subject=subject).to_csv(f'{pre}_desc-counts.tsv', sep='\t', index=False)
    print(con.query("v1_sel == 'tuned' and ips_sel == 'all' and contrast == 'delta90'")
          [['condition', 'segment', 'metric', 'delta_obs', 'delta_obs_minus_perm', 'p_perm',
            'delta_sub']].to_string())
    print(f'saved to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject')
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--n-perm', type=int, default=1000)
    p.add_argument('--n-sub', type=int, default=200)
    p.add_argument('--lags', type=int, default=1)
    p.add_argument('--out-root', default=None)
    a = p.parse_args()
    main(a.subject, a.bids_folder, a.n_perm, a.n_sub, a.lags, out_root=a.out_root)
