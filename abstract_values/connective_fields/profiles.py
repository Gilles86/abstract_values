#!/usr/bin/env python3
"""Coupling profiles behind the mapping score, summarised for plotting.

``test_coupling.py`` reduces each NPCr voxel's connective field (CF) to one
number. This script keeps the shape, so the effect can be seen rather than
inferred. Same data, residuals, labels and channels as the npc_from_v1
direction there (8 V1 orientation channels, joint tuning labels).

Two summaries per subject, for lags 0 and 1 (``residualise``):

aligned   Each voxel-session CF re-expressed as a function of orientation
          relative to theta*, the orientation worth the voxel's preferred value
            'current'  theta* under this session's mapping
            'other'    theta* under the other session's mapping
          Only voxels whose theta* differs by >= MIN_SHIFT deg between the
          mappings, so the two references pick out different V1 channels. If
          coupling follows the current mapping, the 'current' profile peaks at
          0 and the 'other' profile is displaced.
heatmap   Per preferred-value bin: the CF difference between the sessions
          (cdf minus inverse_cdf) per V1 channel, next to the difference the
          two mappings predict (voxel-wise z-scored over channels).

``--projection iem`` builds the V1 channels by inverting the vonmises encoding
model (all weights of every voxel) on a 5-deg grid instead of argmax bins.

Output: derivatives/connective_fields/profiles[_iem]/sub-<S>/sub-<S>_desc-{aligned,heatmap}.tsv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from abstract_values.connective_fields.gates import (
    connective_field, demean_runs, get_paradigm, grid_fit, load_mappings, load_roi,
    residualise, zscore)
from abstract_values.connective_fields.test_coupling import (
    IEM_GRID, KAPPA, N_BASIS, bin_channels, centre_ori, fit_vonmises_weights, iem_channels,
    orientation_edges, value_prediction, variant_suffix)
from abstract_values.utils.data import Subject, BIDS_FOLDER

N_CHANNELS = 8
MIN_SHIFT = 15.
VALUE_BINS = np.linspace(2, 42, 9)


def wrap(d):
    return (d + 90) % 180 - 90


def main(subject, bids_folder=BIDS_FOLDER, projection='bins', n_channels=N_CHANNELS,
         n_basis=N_BASIS, kappa=KAPPA):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    par = get_paradigm(sub, sessions)
    betas = sub.get_single_trial_estimates(sessions, desc='gabor')
    y_npc, sel_npc = load_roi(sub, betas, 'NPCr', 'aprf.cv', bids_folder, False)
    y_v1, _ = load_roi(sub, betas, 'BensonV1ecc075-375', 'vonmises.cv', bids_folder, False)
    y_npc = y_npc[:, sel_npc]

    ori, maps = load_mappings()
    mode, fwhm, _ = grid_fit(demean_runs(y_npc, par), par['value'].to_numpy(), 'value')
    v1_lab, _, _ = grid_fit(demean_runs(y_v1, par), par['orientation'].to_numpy(), 'orientation')
    if projection == 'iem':
        centres, step = IEM_GRID, IEM_GRID[1] - IEM_GRID[0]
        v1_w = fit_vonmises_weights(y_v1, par, n_basis, kappa)
    else:
        centres, step = np.arange(n_channels) * 180. / n_channels, 180. / n_channels

    theta = {c: np.interp(mode, maps[c], ori) for c in maps}          # voxels
    shifted = np.abs(wrap(theta['cdf'] - theta['inverse_cdf'])) >= MIN_SHIFT
    vbin = np.clip(np.digitize(mode, VALUE_BINS[1:-1]), 0, len(VALUE_BINS) - 2)
    pred = {c: zscore(value_prediction(mode, fwhm, np.interp(centres, ori, maps[c])), axis=0)
            for c in maps}                                              # channels x voxels

    aligned, heat = [], []
    for lags in (0, 1):
        cf = {}
        for s in sessions:
            ses = (par['session'] == s).to_numpy()
            p_ses = par[ses].reset_index(drop=True)
            cond = p_ses['condition'].iloc[0]
            res_v1 = residualise(y_v1[ses], p_ses, lags)
            d = (iem_channels(res_v1, v1_w, kappa=kappa) if projection == 'iem' else
                 bin_channels(res_v1, centre_ori(v1_lab, n_channels),
                              orientation_edges(n_channels))[0])
            cf[cond] = connective_field(residualise(y_npc[ses], p_ses, lags), d)

        for cond, c in cf.items():
            other = 'cdf' if cond == 'inverse_cdf' else 'inverse_cdf'
            for ref, th in (('current', theta[cond]), ('other', theta[other])):
                off = np.round(wrap(centres[:, None] - th[None, :]) / step) * step + 0.
                off[off == 90] = -90
                df = pd.DataFrame({'offset': off[:, shifted].ravel(),
                                   'cf': c[:, shifted].ravel()})
                for o, g in df.groupby('offset'):
                    aligned.append({'lags': lags, 'session_mapping': cond, 'reference': ref,
                                    'offset': o, 'cf': g.cf.mean(), 'n': len(g)})

        diff_obs = cf['cdf'] - cf['inverse_cdf']
        diff_pred = pred['cdf'] - pred['inverse_cdf']
        for b in range(len(VALUE_BINS) - 1):
            m = vbin == b
            if not m.any():
                continue
            for k, cc in enumerate(centres):
                heat.append({'lags': lags, 'value_bin': (VALUE_BINS[b] + VALUE_BINS[b + 1]) / 2,
                             'channel': cc, 'observed': diff_obs[k, m].mean(),
                             'predicted': diff_pred[k, m].mean(), 'n': int(m.sum())})

    out = (bids_folder / 'derivatives' / 'connective_fields'
           / ('profiles' + variant_suffix(projection, n_channels, n_basis, kappa))
           / f'sub-{subject}')
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(aligned).assign(subject=subject, n_shifted=int(shifted.sum())).to_csv(
        out / f'sub-{subject}_desc-aligned.tsv', sep='\t', index=False)
    pd.DataFrame(heat).assign(subject=subject).to_csv(
        out / f'sub-{subject}_desc-heatmap.tsv', sep='\t', index=False)
    print(f'{shifted.sum()} of {len(mode)} tuned NPCr voxels shift >= {MIN_SHIFT} deg; saved to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject')
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--projection', choices=['bins', 'iem'], default='bins')
    p.add_argument('--n-channels', type=int, default=N_CHANNELS)
    p.add_argument('--n-basis', type=int, default=N_BASIS)
    p.add_argument('--kappa', type=float, default=KAPPA)
    a = p.parse_args()
    main(a.subject, bids_folder=a.bids_folder, projection=a.projection,
         n_channels=a.n_channels, n_basis=a.n_basis, kappa=a.kappa)
