#!/usr/bin/env python3
"""Decoded posteriors in BOTH stimulus spaces, per trial and mapping (run on the cluster).

For every decoded trial of one decode_lowrank_cv.py setting (tree x N x noise),
the posterior is summarised on the value axis (CHF) AND on the orientation axis
(deg), using that session's own orientation -> value mapping (23 presented
pairs, monotone increasing; repo 'cdf' = Misaligned, 'inverse_cdf' = Aligned):

  value decoding        grid in CHF -> orientation by interpolating the mapping
  orientation decoding  grid of the 23 orientations -> their values (exact)

Orientation-decoded posteriors are summarised circularly on the orientation
axis (axial, period 180 deg) and linearly in value; value-decoded posteriors
linearly on both (the mapping covers 0-172.5 deg monotonically).

Writes derivatives/<tree>/mapping_profiles_nvoxels-<N>_noise-<noise>.tsv.gz,
one row per trial x ROI.

    python analyses/followup_2026-10/lowrank_decoding/mapping_profiles_collect.py \
        --tree decoding_lowrankcv --n-voxels 250 --noise lowrankcv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from abstract_values.utils.data import Subject

ROIS = {'gabor': ['BensonV1', 'BensonV2', 'BensonV3'],
        'value': ['BensonV1', 'NPCr', 'IPS0', 'IPS12', 'IPS345']}


def mapping_tables(sub):
    out = {}
    for s in sub.get_sessions():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ev = sub.get_events(s, sub.get_runs(s))
        g = (ev[ev['event_type'] == 'gabor'][['orientation', 'value']]
             .astype(float).drop_duplicates().sort_values('orientation'))
        out[s] = (sub.get_mapping(s), g['orientation'].to_numpy(), g['value'].to_numpy())
    return out


def moments(P, x):
    m = P @ x
    return m, np.sqrt(np.clip(P @ x ** 2 - m ** 2, 0, None))


def summarise_subject(fn, kind, maps, roi, subject):
    post = pd.read_csv(fn, sep='\t', index_col=[0, 1, 2, 3])
    grid = post.columns.astype(float).to_numpy()
    rows = []
    for s, g in post.groupby(level='session'):
        mapping, oris, vals = maps[s]
        P = g.to_numpy(float)
        P /= P.sum(1, keepdims=True)
        true = g.index.get_level_values(3).to_numpy(float)
        if kind == 'value':
            v_true = true
            o_true = np.interp(v_true, vals, oris)
            v_grid, o_grid = grid, np.interp(grid, vals, oris)
            ti = np.abs(true[:, None] - grid[None, :]).argmin(1)
            v_mean, v_sd = moments(P, v_grid)
            o_mean, o_sd = moments(P, o_grid)
            o_err = o_mean - o_true
        else:
            o_grid = np.rad2deg(grid)
            o_true = np.rad2deg(true)
            v_grid = np.interp(o_grid, oris, vals)
            v_true = np.interp(o_true, oris, vals)
            ti = np.abs(true[:, None] - grid[None, :]).argmin(1)
            z = P @ np.exp(2j * grid)
            o_mean = np.rad2deg(np.angle(z) / 2) % 180
            o_sd = np.rad2deg(np.sqrt(-2 * np.log(np.clip(np.abs(z), 1e-12, 1))) / 2)
            o_err = (o_mean - o_true + 90) % 180 - 90
            v_mean, v_sd = moments(P, v_grid)
        pt = P[np.arange(len(P)), ti]
        hpd = (P * (P > pt[:, None])).sum(1) + 0.5 * (P * (P == pt[:, None])).sum(1)
        rows.append(pd.DataFrame(dict(
            subject=subject, roi=roi, decoder=kind, session=s, mapping=mapping,
            run=g.index.get_level_values(1), trial_nr=g.index.get_level_values(2),
            orientation=np.round(o_true, 2), value=v_true,
            v_mean=v_mean, v_sd=v_sd, v_err=v_mean - v_true,
            o_mean=o_mean, o_sd=o_sd, o_err=o_err,
            logp=np.log(pt + 1e-300) + np.log(len(grid)), hpd=hpd)))
    return pd.concat(rows)


def main(bids_folder, tree, n_voxels, noise):
    root = Path(bids_folder) / 'derivatives' / tree
    out = []
    subjects = sorted({p.name[4:] for k in ROIS for p in (root / k).glob('sub-*')})
    for subject in subjects:
        sub = Subject(subject, bids_folder=bids_folder)
        maps = mapping_tables(sub)
        for kind, rois in ROIS.items():
            for roi in rois:
                fn = (root / kind / f'sub-{subject}' / 'func' /
                      f'sub-{subject}_mask-{roi}_nvoxels-{n_voxels}_noise-{noise}_pars.tsv')
                if fn.exists():
                    out.append(summarise_subject(fn, kind, maps, roi, subject))
        print(f'sub-{subject} done', flush=True)
    out = pd.concat(out, ignore_index=True)
    fn = root / f'mapping_profiles_nvoxels-{n_voxels}_noise-{noise}.tsv.gz'
    out.to_csv(fn, sep='\t', index=False, float_format='%.5g')
    print(fn, len(out))
    print(out.groupby(['decoder', 'roi']).subject.nunique().to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-abstractvalue')
    p.add_argument('--tree', default='decoding_lowrankcv')
    p.add_argument('--n-voxels', default='250')
    p.add_argument('--noise', default='lowrankcv')
    a = p.parse_args()
    main(a.bids_folder, a.tree, a.n_voxels, a.noise)
