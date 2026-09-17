#!/usr/bin/env python3
"""Does residual NPC <-> V1 coupling follow the current orientation->value mapping?

Primary test of the connective-field analysis (step-0 gates: gates.py).

Per session, stimulus-locked variance is removed from every voxel (one mean per
orientation, one per run). What is left is trial-to-trial fluctuation. A
connective field (CF) is the correlation of a target voxel's residuals with
source "channels": source voxels binned by their tuning and averaged, expressed
as deviations from the source-wide mean.

Two directions, same logic:

  npc_from_v1  Target: tuned NPCr voxels, label = joint log-Gaussian value tuning
               f_i. Channels: V1 (0.75-3.75 deg) voxels binned by preferred
               orientation theta_k.
               Prediction under mapping c: f_i(m_c(theta_k)).
  v1_from_npc  Target: tuned V1 voxels, label = joint axial von Mises tuning g_j.
               Channels: tuned NPCr voxels in value-quantile bins v_k.
               Prediction: g_j(m_c^-1(v_k)).

Every label -- target and source -- comes from ONE fit on both sessions, so the
channels (membership and centres) are identical in the two sessions and nothing
about their definition differs with the mapping. Labels depend only on the
orientation means, the CF only on the residuals around them, so the joint fit
does not leak into the CF.

Score per target voxel = r(CF, prediction under this session's mapping)
                       - r(CF, prediction under the other mapping),
averaged over both sessions. Coupling that is the same in both conditions
contributes equally to both terms and cancels.

Control: the same score with tuning labels permuted across target voxels. It
keeps any ROI-wide change of the CF between sessions and removes only the
voxel-by-voxel correspondence between tuning and coupling, so
observed - shuffled is the part the hypothesis is actually about. Within one
subject the shuffled score need not be zero: the ROI-mean CF can differ between
sessions along the direction in which the two mappings' predictions differ.
Condition equals session within a subject, so only the group level, where the
session order is counterbalanced, says whether that part tracks the mapping.

Outputs, under derivatives/connective_fields/coupling/sub-<S>/:
  sub-<S>_desc-scores.tsv    observed and label-shuffled scores per direction
  sub-<S>_desc-bytuning.tsv  observed score binned by the target voxels' tuning

Usage:
  python -m abstract_values.connective_fields.test_coupling 03
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from abstract_values.connective_fields.gates import (
    colcorr, connective_field, demean_runs, get_paradigm, grid_fit, load_mappings,
    load_roi, lognormal_mode_fwhm, residualise, zscore)
from abstract_values.utils.data import Subject, BIDS_FOLDER


def bin_channels(res, labels, edges):
    """Average residuals within label bins; deviations from the channel mean, z-scored."""
    idx = np.clip(np.digitize(labels, edges[1:-1]), 0, len(edges) - 2)
    ch = np.stack([res[:, idx == k].mean(1) for k in range(len(edges) - 1)], 1)
    return zscore(ch - ch.mean(1, keepdims=True)), idx


def orientation_edges(n):
    """Bins centred on 0, 180/n, ... expressed on a label shifted by half a bin."""
    return np.arange(n + 1) * 180. / n


def centre_ori(labels, n):
    return (labels + 90. / n) % 180


def value_prediction(mode, fwhm, channel_values):
    p = lognormal_mode_fwhm(channel_values[:, None], mode[None, :], fwhm[None, :])
    return p - p.mean(0, keepdims=True)


def orientation_prediction(mu, kappa, channel_oris):
    p = np.exp(kappa[None, :] * np.cos(np.deg2rad(2 * (channel_oris[:, None] - mu[None, :]))))
    return p - p.mean(0, keepdims=True)


def run_direction(direction, par, target, source, n_channels, n_shuffle, rng):
    """target/source: dicts with residual-ready data (trials x voxels) and kind."""
    ori, maps = load_mappings()
    sessions = sorted(par['session'].unique())

    # Joint tuning label of the target voxels.
    loc, wid, _ = grid_fit(demean_runs(target['y'], par),
                           par[target['stim']].to_numpy(), target['kind'])

    src_lab, _, _ = grid_fit(demean_runs(source['y'], par),
                             par[source['stim']].to_numpy(), source['kind'])

    per_session = []
    for s in sessions:
        ses = (par['session'] == s).to_numpy()
        p_ses = par[ses].reset_index(drop=True)
        cond = p_ses['condition'].iloc[0]
        wrong = 'cdf' if cond == 'inverse_cdf' else 'inverse_cdf'

        res_src = residualise(source['y'][ses], p_ses)
        if source['kind'] == 'orientation':
            d, _ = bin_channels(res_src, centre_ori(src_lab, n_channels),
                                orientation_edges(n_channels))
            centres = np.arange(n_channels) * 180. / n_channels
            chan = lambda c: np.interp(centres, ori, maps[c])        # value per channel
        else:
            edges = np.quantile(src_lab, np.linspace(0, 1, n_channels + 1))
            d, idx = bin_channels(res_src, src_lab, edges)
            centres = np.array([np.median(src_lab[idx == k]) for k in range(n_channels)])
            chan = lambda c: np.interp(centres, maps[c], ori)        # orientation per channel

        cf = connective_field(residualise(target['y'][ses], p_ses), d)
        per_session.append(dict(cf=cf, right=chan(cond), wrong=chan(wrong)))

    def predict(channel_stim, lo, wi):
        return (value_prediction(lo, wi, channel_stim) if target['kind'] == 'value'
                else orientation_prediction(lo, wi, channel_stim))

    def score(lo, wi):
        per_voxel = [colcorr(x['cf'], predict(x['right'], lo, wi))
                     - colcorr(x['cf'], predict(x['wrong'], lo, wi)) for x in per_session]
        return np.nanmean(per_voxel, 0)

    observed = score(loc, wid)
    rows = [{'direction': direction, 'kind': 'observed', 'perm': -1,
             'score': float(np.nanmean(observed)), 'n_target': len(loc)}]
    for k in range(n_shuffle):
        perm = rng.permutation(len(loc))
        rows.append({'direction': direction, 'kind': 'label-shuffle', 'perm': k,
                     'score': float(np.nanmean(score(loc[perm], wid[perm]))),
                     'n_target': len(loc)})

    # How different the two mappings' predictions are for each voxel: the score
    # can only be large where they differ.
    x = per_session[0]
    dissim = 1 - colcorr(predict(x['right'], loc, wid), predict(x['wrong'], loc, wid))
    if target['kind'] == 'value':
        bins = np.linspace(2, 42, 9)
    else:
        bins = np.linspace(0, 180, 9)
    b = np.clip(np.digitize(loc, bins[1:-1]), 0, len(bins) - 2)
    by = pd.DataFrame({'bin': b, 'score': observed, 'dissimilarity': dissim}).groupby('bin').agg(
        score=('score', 'mean'), dissimilarity=('dissimilarity', 'mean'),
        n=('score', 'size')).reset_index()
    by['tuning'] = (bins[by['bin']] + bins[by['bin'] + 1]) / 2
    by['direction'] = direction
    return pd.DataFrame(rows), by


def main(subject, bids_folder=BIDS_FOLDER, n_channels=8, n_shuffle=200, seed=0,
         smoothed=False):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    if len(sessions) != 2:
        raise SystemExit(f'sub-{subject}: needs exactly 2 sessions, has {sessions}')
    par = get_paradigm(sub, sessions)
    betas = sub.get_single_trial_estimates(sessions, desc='gabor', smoothed=smoothed)
    if betas.shape[3] != len(par):
        raise SystemExit(f'{betas.shape[3]} betas vs {len(par)} trials')

    y_npc, sel_npc = load_roi(sub, betas, 'NPCr', 'aprf.cv', bids_folder, smoothed)
    y_v1, sel_v1 = load_roi(sub, betas, 'BensonV1ecc075-375', 'vonmises.cv', bids_folder,
                            smoothed)
    npc_tuned = dict(y=y_npc[:, sel_npc], kind='value', stim='value')
    v1_tuned = dict(y=y_v1[:, sel_v1], kind='orientation', stim='orientation')
    v1_all = dict(y=y_v1, kind='orientation', stim='orientation')

    rng = np.random.default_rng(seed)
    scores, by = zip(
        # Gates: all V1 voxels make stable channels; tuned-only leaves 1-voxel bins.
        run_direction('npc_from_v1', par, npc_tuned, v1_all, n_channels, n_shuffle, rng),
        run_direction('v1_from_npc', par, v1_tuned, npc_tuned, n_channels, n_shuffle, rng))
    scores = pd.concat(scores, ignore_index=True).assign(subject=subject)
    by = pd.concat(by, ignore_index=True).assign(subject=subject)

    out = bids_folder / 'derivatives' / 'connective_fields' / 'coupling' / f'sub-{subject}'
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / f'sub-{subject}_desc-scores.tsv', sep='\t', index=False)
    by.to_csv(out / f'sub-{subject}_desc-bytuning.tsv', sep='\t', index=False)
    summ = scores.groupby(['direction', 'kind'])['score'].agg(['mean', 'std'])
    print(summ)
    print(f'saved to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject')
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--n-channels', type=int, default=8)
    p.add_argument('--n-shuffle', type=int, default=200)
    p.add_argument('--smoothed', action='store_true')
    a = p.parse_args()
    main(a.subject, bids_folder=a.bids_folder, n_channels=a.n_channels,
         n_shuffle=a.n_shuffle, smoothed=a.smoothed)
