#!/usr/bin/env python3
"""Per-voxel "most-connected V1 orientation" for NPCr voxels, per session.

For every tuned NPCr voxel: its preferred value (joint log-Gaussian fit) and,
per session, the V1 orientation its residual fluctuations couple to most
strongly. V1 channels come from the inverted vonmises encoding model (default
24 basis functions, kappa 16: the narrowest setting in plot_specificity.py) on
a 5-deg grid; coupling is the channel-centred CF from test_coupling.py.

Two readouts of the CF over orientation:
  peak      argmax (5-deg resolution)
  centroid  axial circular mean of the positive part of the CF

Both readouts also after removing the session's NPCr-mean CF ("_specific"), so a
profile shared by the whole ROI cannot set every voxel's peak.

Also a coupling map per session: mean voxel-specific CF per (preferred-value
bin, V1 orientation). Averaging CFs before taking any peak keeps the noise
linear, which a per-voxel argmax does not.

Output: derivatives/connective_fields/peaks/sub-<S>/
  sub-<S>_desc-peaks.tsv.gz   one row per tuned NPCr voxel
  sub-<S>_desc-maps.tsv       value bin x V1 orientation x session mapping
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from abstract_values.connective_fields.gates import (
    connective_field, demean_runs, get_paradigm, grid_fit, load_roi, remove_gain, residualise)
from abstract_values.connective_fields.test_coupling import (
    IEM_GRID, fit_vonmises_weights, iem_channels)
from abstract_values.utils.data import Subject, BIDS_FOLDER


def axial_centroid(cf, grid):
    w = np.clip(cf, 0, None)
    z = (w * np.exp(2j * np.deg2rad(grid))[:, None]).sum(0)
    return np.rad2deg(np.angle(z)) / 2 % 180, np.abs(z) / np.maximum(w.sum(0), 1e-12)


VALUE_BINS = np.arange(2, 44, 2.)


def main(subject, bids_folder=BIDS_FOLDER, n_basis=24, kappa=16., lags=0, gain=False,
         npc_class='all'):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    par = get_paradigm(sub, sessions)
    betas = sub.get_single_trial_estimates(sessions, desc='gabor')
    if npc_class == 'all':
        y_npc, sel_npc = load_roi(sub, betas, 'NPCr', 'aprf.cv', bids_folder, False)
    else:
        # As in test_coupling.py: value aPRF vs its orientation-space twin.
        y_npc, sel_npc, delta = load_roi(sub, betas, 'NPCr', 'aprf.cv', bids_folder, False,
                                         compare='vonmises-prf.cv')
        sel_npc = sel_npc & ((delta > 0) if npc_class == 'value' else (delta < 0))
    y_v1, _ = load_roi(sub, betas, 'BensonV1ecc075-375', 'vonmises.cv', bids_folder, False)
    y_npc = y_npc[:, sel_npc]

    mode, fwhm, fit_r = grid_fit(demean_runs(y_npc, par), par['value'].to_numpy(), 'value')
    w = fit_vonmises_weights(y_v1, par, n_basis, kappa)
    maps = []
    out = pd.DataFrame({'voxel': np.arange(len(mode)), 'mode': mode, 'fwhm': fwhm,
                        'fit_r': fit_r})
    for s in sessions:
        ses = (par['session'] == s).to_numpy()
        p_ses = par[ses].reset_index(drop=True)
        cond = p_ses['condition'].iloc[0]
        res_v1 = residualise(y_v1[ses], p_ses, lags)
        res_npc = residualise(y_npc[ses], p_ses, lags)
        if gain:
            res_v1 = remove_gain(res_v1, y_v1[ses], p_ses)[0]
            res_npc = remove_gain(res_npc, y_npc[ses], p_ses)[0]
        d = iem_channels(res_v1, w, kappa=kappa)
        cf = connective_field(res_npc, d)
        out[f'peak_{cond}'] = IEM_GRID[np.argmax(cf, 0)]
        out[f'centroid_{cond}'], out[f'concentration_{cond}'] = axial_centroid(cf, IEM_GRID)
        out[f'cf_max_{cond}'] = cf.max(0)
        spec = cf - cf.mean(1, keepdims=True)
        out[f'peak_specific_{cond}'] = IEM_GRID[np.argmax(spec, 0)]
        out[f'centroid_specific_{cond}'], _ = axial_centroid(spec, IEM_GRID)
        vbin = np.clip(np.digitize(mode, VALUE_BINS[1:-1]), 0, len(VALUE_BINS) - 2)
        for b in np.unique(vbin):
            m = vbin == b
            for k, th in enumerate(IEM_GRID):
                maps.append({'condition': cond, 'value_bin': VALUE_BINS[b] + 1.,
                             'orientation': th, 'cf': spec[k, m].mean(),
                             'cf_raw': cf[k, m].mean(), 'n': int(m.sum())})

    dst = (bids_folder / 'derivatives' / 'connective_fields'
           / (('peaks' if lags == 0 else f'peaks_lags-{lags}') + ('_gain' if gain else '')
              + ('' if npc_class == 'all' else f'_npc-{npc_class}'))
           / f'sub-{subject}')
    dst.mkdir(parents=True, exist_ok=True)
    out.assign(subject=subject).to_csv(dst / f'sub-{subject}_desc-peaks.tsv.gz', sep='\t',
                                       index=False)
    pd.DataFrame(maps).assign(subject=subject).to_csv(dst / f'sub-{subject}_desc-maps.tsv',
                                                      sep='\t', index=False)
    print(f'{len(out)} voxels; saved to {dst}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject')
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--n-basis', type=int, default=24)
    p.add_argument('--kappa', type=float, default=16.)
    p.add_argument('--lags', type=int, default=0)
    p.add_argument('--gain', action='store_true', help='Remove per-trial gain in both regions')
    p.add_argument('--npc-class', choices=['all', 'value', 'orientation'], default='all',
                   help='Only NPCr voxels where aprf.cv beats / loses to vonmises-prf.cv')
    a = p.parse_args()
    main(a.subject, a.bids_folder, a.n_basis, a.kappa, a.lags, a.gain, a.npc_class)
