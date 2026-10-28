#!/usr/bin/env python3
"""A5 (exploratory): is V1 -> IPS residual coupling stronger for 90-deg-preferring V1 voxels?

Builds on the residual-coupling analysis in ``abstract_values/connective_fields``
(``gates.residualise``, ``gates.load_roi``): per session, every voxel's single-trial
GLMsingle betas are residualised on one dummy per orientation (23), one per run and
the orientations of the +-1 neighbouring trials (HRF overlap). What remains is
trial-to-trial fluctuation around the stimulus-locked means.

V1 voxels are binned by preferred orientation from the joint (both-session) axial von
Mises PRF fit (``derivatives/encoding_models/vonmises-prf``: desc-mu, +90 deg where the
fitted amplitude is negative, i.e. mu is the trough). Labels come from one fit on both
sessions, so bin membership is identical under both mappings. Bins are centred on
0, 180/K, ... (K = 6 or 8), so one bin is centred on 90 deg.

V1 voxel sets:
  ann-all    Benson V1 0.75-3.75 deg annulus (the stimulated region; the voxels the
             existing coupling analysis uses), all usable voxels
  ann-tuned  annulus voxels with cvR2(vonmises-prf.cv) > cvR2(aprf-null.cv)
  v1-tuned   whole Benson V1 (both hemis), same cvR2 selection

Coupling of bin k with an IPS segment (Wang IPS0, IPS1-2, IPS3-5, union), per session
(= mapping), IPS voxels either all or aPRF-selected (aprf.cv > null):
  raw       c_k = mean residual over the bin's V1 voxels. Coupling = mean over IPS
            voxels of Fisher-z r(c_k, IPS voxel residual). Includes the V1-wide
            shared fluctuation.
  specific  d_k = c_k - mean residual over all V1 voxels of the set (removes the shared
            fluctuation, as the channels in test_coupling do). Coupling = mean over IPS
            voxels of r(d_k, IPS voxel)^2 x 100 (a voxel's sign of coupling to a channel
            is free, so squared). With no coupling E[r^2] = 1/(T-1) for any bin size.

Contrasts: delta90 = C[90 bin] - mean C[other bins]; delta0 = same for the 0-deg
(horizontal) bin, to see whether anything at 90 is vertical-specific or cardinal.

Nulls (Saurabh: a bin with more voxels gets a less noisy average, hence more coupling
by sampling alone):
  perm  preferred orientations permuted across V1 voxels (marginal distribution, hence
        bin counts, preserved), N_PERM times; delta under each permutation.
  sub   every bin subsampled to the smallest bin's voxel count (>= MIN_BIN), N_SUB draws;
        coupling averaged over draws.

Outputs: derivatives/followup_2026-10/a5_coupling_90/sub-<S>/
  sub-<S>_desc-bins.tsv        coupling per (n_bins, v1_sel, session, segment, ips_sel, metric, bin)
  sub-<S>_desc-contrasts.tsv   delta: observed, subsampled, permutation mean/sd/p
  sub-<S>_desc-permnull.tsv.gz permutation deltas (for the group-level null)
  sub-<S>_desc-counts.tsv      V1 voxels per bin
  sub-<S>_desc-v1pref.tsv      per V1 voxel: set membership, preferred orientation, cvR2 gain

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

SEGMENTS = {'IPS0': 'IPS0', 'IPS12': 'IPS12', 'IPS345': 'IPS345',
            'IPS0-5': 'IPS0+IPS12+IPS345'}
BIN_SCHEMES = (6, 8)
MIN_BIN = 3


def bin_index(pref_deg, n_bins):
    w = 180. / n_bins
    return np.floor(((pref_deg + w / 2) % 180) / w).astype(int)


def load_v1(sub, betas, bids_folder, which):
    """Betas (trials x voxels), preferred orientation (deg), cvR2 gain over null."""
    if which == 'ann':
        mask = sub.get_roi_mask('BensonV1ecc075-375', hemi=None)
    else:
        mask = sub.get_roi_mask('BensonV1', hemi='LR')
    if len(mask.shape) == 4:
        mask = image.index_img(mask, 0)
    mask = image.resample_to_img(mask, image.index_img(betas, 0), interpolation='nearest')
    masker = NiftiMasker(mask_img=mask).fit()
    y = masker.transform(betas).astype(np.float64)
    ok = np.isfinite(y).all(0) & (y.std(0) > 0)          # same rule as gates.load_roi
    em = Path(bids_folder) / 'derivatives' / 'encoding_models'
    sid = sub.subject_id

    def par(model, desc):
        fn = em / model / f'sub-{sid}' / 'func' / f'sub-{sid}_task-abstractvalue_space-T1w_desc-{desc}_pe.nii.gz'
        return masker.transform(image.load_img(str(fn))).ravel()[ok]
    mu, amp = par('vonmises-prf', 'mu'), par('vonmises-prf', 'amplitude')
    gain = par('vonmises-prf.cv', 'cvr2') - par('aprf-null.cv', 'cvr2')
    pref = (np.rad2deg(mu) + np.where(amp < 0, 90., 0.)) % 180.
    print(f'  V1 {which}: {ok.sum()} usable voxels, mu {np.nanmin(mu):.2f}..{np.nanmax(mu):.2f} rad, '
          f'{(amp < 0).mean():.0%} negative amplitude, {(gain > 0).sum()} beat null')
    return y[:, ok], pref, gain


def coupling(res_v1, assign, res_ips_z, metric, n_bins, v1_mean):
    """Bin coupling for a stack of bin assignments (P x V; -1 = voxel left out)."""
    T = res_v1.shape[0]
    out = np.empty((len(assign), n_bins))
    for p, a in enumerate(assign):
        keep = a >= 0
        ind = np.zeros((keep.sum(), n_bins))
        ind[np.arange(keep.sum()), a[keep]] = 1
        ind /= np.maximum(ind.sum(0, keepdims=True), 1)
        c = res_v1[:, keep] @ ind                                # T x bins
        if metric == 'specific':
            c = c - v1_mean[:, None]
        r = zscore(c).T @ res_ips_z / T                          # bins x IPS voxels
        out[p] = (np.arctanh(np.clip(r, -.999, .999)).mean(1) if metric == 'raw'
                  else 100 * (r ** 2).mean(1))
    return out


def subsample_assignments(idx, n_bins, n_sub, n_draw, rng):
    members = [np.flatnonzero(idx == k) for k in range(n_bins)]
    rows = []
    for _ in range(n_draw):
        a = np.full(len(idx), -1)
        for k, m in enumerate(members):
            a[rng.choice(m, n_sub, replace=False)] = k
        rows.append(a)
    return np.array(rows)


def contrast(c, k):
    others = [j for j in range(c.shape[-1]) if j != k]
    return c[..., k] - c[..., others].mean(-1)


def main(subject, bids_folder=BIDS_FOLDER, n_perm=1000, n_sub=200, lags=1, seed=0,
         out_root=None):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    if len(sessions) != 2:
        raise SystemExit(f'sub-{subject}: needs 2 sessions, has {sessions}')
    par = get_paradigm(sub, sessions)
    betas = sub.get_single_trial_estimates(sessions, desc='gabor')
    if betas.shape[3] != len(par):
        raise SystemExit(f'{betas.shape[3]} betas vs {len(par)} trials')

    v1 = {w: load_v1(sub, betas, bids_folder, w) for w in ('ann', 'v1')}
    v1_sets = {'ann-all': ('ann', None), 'ann-tuned': ('ann', True), 'v1-tuned': ('v1', True)}
    ips = {seg: load_roi(sub, betas, roi, 'aprf.cv', bids_folder, False)
           for seg, roi in SEGMENTS.items()}

    rng = np.random.default_rng(seed + sum(map(ord, str(subject))))
    rows_bins, rows_con, rows_null, rows_cnt, rows_pref = [], [], [], [], []
    for v1_name, (roi, tuned) in v1_sets.items():
        y_v1, pref, gain = v1[roi]
        keep = np.isfinite(pref) & ((gain > 0) if tuned else True)
        rows_pref.append(pd.DataFrame({'v1_sel': v1_name, 'pref': pref[keep],
                                       'cvr2_gain': gain[keep]}))
        res_by_ses = {}
        for s in sessions:
            ses = (par['session'] == s).to_numpy()
            p_ses = par[ses].reset_index(drop=True)
            res_by_ses[s] = residualise(y_v1[ses][:, keep], p_ses, lags)

        for n_bins in BIN_SCHEMES:
            centres = np.arange(n_bins) * 180. / n_bins
            b90 = n_bins // 2
            idx = bin_index(pref[keep], n_bins)
            counts = np.bincount(idx, minlength=n_bins)
            rows_cnt += [{'n_bins': n_bins, 'v1_sel': v1_name, 'bin': k, 'centre': centres[k],
                          'n_vox': int(counts[k])} for k in range(n_bins)]
            n_min = int(counts.min())
            perms = np.array([rng.permutation(idx) for _ in range(n_perm)])
            subs = (subsample_assignments(idx, n_bins, n_min, n_sub, rng)
                    if n_min >= MIN_BIN else None)
            print(f'  {v1_name} K={n_bins}: {keep.sum()} voxels, bins {counts.tolist()}')

            for s in sessions:
                ses = (par['session'] == s).to_numpy()
                p_ses = par[ses].reset_index(drop=True)
                cond = p_ses['condition'].iloc[0]
                res_v1 = res_by_ses[s]
                v1_mean = res_v1.mean(1)
                for seg, (y_ips, sel_ips) in ips.items():
                    res_ips = residualise(y_ips[ses], p_ses, lags)
                    for ips_name, cols in (('all', np.ones(res_ips.shape[1], bool)),
                                           ('aprf', sel_ips)):
                        if cols.sum() < 10:
                            continue
                        rz = zscore(res_ips[:, cols])
                        for metric in ('raw', 'specific'):
                            args = (rz, metric, n_bins, v1_mean)
                            obs = coupling(res_v1, idx[None], *args)[0]
                            null = coupling(res_v1, perms, *args)
                            sb = (coupling(res_v1, subs, *args).mean(0) if subs is not None
                                  else np.full(n_bins, np.nan))
                            key = dict(n_bins=n_bins, v1_sel=v1_name, session=s, condition=cond,
                                       segment=seg, ips_sel=ips_name, metric=metric)
                            rows_bins += [{**key, 'bin': k, 'centre': centres[k],
                                           'n_vox': int(counts[k]), 'coupling': obs[k],
                                           'coupling_sub': sb[k],
                                           'coupling_perm': null[:, k].mean()}
                                          for k in range(n_bins)]
                            for cname, kb in (('delta90', b90), ('delta0', 0)):
                                d_obs, d_null, d_sub = (contrast(obs, kb), contrast(null, kb),
                                                        contrast(sb, kb))
                                rows_con.append({**key, 'contrast': cname, 'delta_obs': d_obs,
                                                 'delta_perm_mean': d_null.mean(),
                                                 'delta_perm_sd': d_null.std(),
                                                 'delta_obs_minus_perm': d_obs - d_null.mean(),
                                                 'p_perm': (1 + (d_null >= d_obs).sum()) / (1 + n_perm),
                                                 'delta_sub': d_sub,
                                                 'n_sub': n_min if subs is not None else 0,
                                                 'n_bin': int(counts[kb]),
                                                 'n_ips': int(cols.sum()), 'n_v1': int(keep.sum()),
                                                 'n_trials': int(ses.sum())})
                                rows_null.append(pd.DataFrame({**key, 'contrast': cname,
                                                               'perm': np.arange(n_perm),
                                                               'delta': d_null}))
        print(f'    {v1_name} done')

    out = (Path(out_root) if out_root else
           bids_folder / 'derivatives' / 'followup_2026-10' / 'a5_coupling_90') / f'sub-{subject}'
    out.mkdir(parents=True, exist_ok=True)
    pre = f'{out}/sub-{subject}'
    w = dict(sep='\t', index=False)
    pd.DataFrame(rows_bins).assign(subject=subject).to_csv(f'{pre}_desc-bins.tsv', **w)
    con = pd.DataFrame(rows_con).assign(subject=subject)
    con.to_csv(f'{pre}_desc-contrasts.tsv', **w)
    pd.concat(rows_null).assign(subject=subject).to_csv(f'{pre}_desc-permnull.tsv.gz', **w)
    pd.DataFrame(rows_cnt).assign(subject=subject).to_csv(f'{pre}_desc-counts.tsv', **w)
    pd.concat(rows_pref).assign(subject=subject).to_csv(f'{pre}_desc-v1pref.tsv', **w)
    print(con.query("n_bins == 6 and v1_sel == 'ann-all' and ips_sel == 'all' and contrast == 'delta90'")
          [['condition', 'segment', 'metric', 'delta_obs', 'delta_obs_minus_perm', 'p_perm',
            'delta_sub', 'n_sub']].to_string())
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
