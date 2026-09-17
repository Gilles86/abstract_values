#!/usr/bin/env python3
"""Preferred value vs most-connected V1 orientation, per NPCr voxel.

  a, b  Distribution of each voxel's most-connected V1 orientation (axial
        centroid of its CF) given its preferred value, in CDF and inverse-CDF
        sessions. Columns are normalised (each value bin sums to 1), so the
        uneven distribution of preferred values does not dominate. Lines: the
        orientation worth that value under each mapping (theta*).
  c     Within voxel: centroid in the CDF session minus the inverse-CDF session,
        against preferred value, with the shift the mappings predict. Anything
        the two sessions share (stable anatomical coupling) cancels here.

Dots: subject means per value bin (circular in a-b), mean ± SEM across subjects.

  python -m abstract_values.connective_fields.plot_peaks
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.connective_fields.gates import load_mappings
from abstract_values.connective_fields.plot_coupling_explainer import MAP_COL, ROOT
from abstract_values.connective_fields.plot_gates import letter

REPO = Path(__file__).resolve().parents[2]
VBINS = np.arange(2, 44, 4.)
OBINS = np.arange(0, 181, 10.)


def wrap(d):
    return (d + 90) % 180 - 90


def axial_mean(deg):
    z = np.exp(2j * np.deg2rad(deg)).mean()
    return np.rad2deg(np.angle(z)) / 2 % 180


def main(out, readout='centroid'):
    files = sorted((ROOT / 'peaks').glob('sub-*/sub-*_desc-peaks.tsv.gz'))
    df = pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str}) for f in files],
                   ignore_index=True)
    ori, maps = load_mappings()
    theta = {c: np.interp(df['mode'], maps[c], ori) for c in maps}
    df['pred_shift'] = wrap(theta['cdf'] - theta['inverse_cdf'])
    df['obs_shift'] = wrap(df[f'{readout}_cdf'] - df[f'{readout}_inverse_cdf'])
    df['vbin'] = np.clip(np.digitize(df['mode'], VBINS[1:-1]), 0, len(VBINS) - 2)
    vcent = (VBINS[:-1] + VBINS[1:]) / 2
    print(f'{df.subject.nunique()} subjects, {len(df)} voxels')

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.45), constrained_layout=True)
    vv = np.linspace(2, 42, 400)
    for ax, cond, lab in ((axes[0], 'cdf', 'a'), (axes[1], 'inverse_cdf', 'b')):
        h, _, _ = np.histogram2d(df['mode'], df[f'{readout}_{cond}'], [VBINS, OBINS])
        h = h / h.sum(1, keepdims=True)
        ax.pcolormesh(VBINS, OBINS, h.T, cmap='Greys', vmin=0, vmax=np.quantile(h, .98),
                      rasterized=True)
        other = 'inverse_cdf' if cond == 'cdf' else 'cdf'
        ax.plot(vv, np.interp(vv, maps[other], ori), color=MAP_COL[other], lw=1, ls='--')
        ax.plot(vv, np.interp(vv, maps[cond], ori), color=MAP_COL[cond], lw=1.6)
        m = (df.groupby(['subject', 'vbin'])[f'{readout}_{cond}'].apply(axial_mean)
               .groupby('vbin').apply(axial_mean))
        ax.plot(vcent[m.index], m.values, 'o', ms=3.5, mfc='w', mec='k', mew=.9, zorder=4)
        ax.set_xticks([2, 12, 22, 32, 42])
        ax.set_yticks([0, 45, 90, 135, 180])
        ax.set_xlabel('NPCr preferred value (CHF)')
        ax.set_ylabel('Most-connected V1 orientation (°)')
        ax.text(.03, .97, 'CDF sessions' if cond == 'cdf' else 'Inverse-CDF sessions',
                transform=ax.transAxes, va='top', fontsize=7, color=MAP_COL[cond])
        letter(ax, lab)
    axes[1].set_ylabel('')
    axes[0].text(.97, .05, 'θ* (this mapping)', transform=axes[0].transAxes, ha='right',
                 fontsize=6.5, color=MAP_COL['cdf'])
    axes[0].text(.97, .14, 'θ* (other mapping)', transform=axes[0].transAxes, ha='right',
                 fontsize=6.5, color=MAP_COL['inverse_cdf'])

    ax = axes[2]
    h, _, _ = np.histogram2d(df['mode'], df['obs_shift'], [VBINS, np.arange(-90, 91, 10.)])
    h = h / h.sum(1, keepdims=True)
    ax.pcolormesh(VBINS, np.arange(-90, 91, 10.), h.T, cmap='Greys', vmin=0,
                  vmax=np.quantile(h, .98), rasterized=True)
    ax.plot(vv, wrap(np.interp(vv, maps['cdf'], ori) - np.interp(vv, maps['inverse_cdf'], ori)),
            color='#C44E52', lw=1.6)
    sm = df.groupby(['subject', 'vbin'])['obs_shift'].mean().unstack()
    ax.errorbar(vcent[sm.columns], sm.mean(), sm.sem(), fmt='o', ms=3.5, mfc='w', mec='k',
                mew=.9, color='k', lw=.8, zorder=4)
    ax.axhline(0, color='.6', lw=.6, ls='--')
    ax.set_xticks([2, 12, 22, 32, 42])
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_xlabel('NPCr preferred value (CHF)')
    ax.set_ylabel('CDF − inverse-CDF orientation (°)')
    ax.text(.03, .97, 'Predicted shift', transform=ax.transAxes, va='top', fontsize=6.5,
            color='#C44E52')
    letter(ax, 'c')

    r = df.groupby('subject').apply(lambda g: np.corrcoef(g.obs_shift, g.pred_shift)[0, 1],
                                    include_groups=False)
    t = stats.ttest_1samp(r, 0)
    slope = df.groupby('subject').apply(
        lambda g: np.polyfit(g.pred_shift, g.obs_shift, 1)[0], include_groups=False)
    ts = stats.ttest_1samp(slope, 0)
    print(f'per-subject r(observed, predicted shift) = {r.mean():.4f} ± {r.sem():.4f}, '
          f't{len(r)-1} = {t.statistic:.2f}, p = {t.pvalue:.2g}')
    print(f'per-subject slope = {slope.mean():.3f} ± {slope.sem():.3f}, '
          f't = {ts.statistic:.2f}, p = {ts.pvalue:.2g}')
    ax.text(.97, .03, f'r = {r.mean():.3f}, p = {t.pvalue:.1g}'.replace('0.0', '.0'),
            transform=ax.transAxes, ha='right', fontsize=6.5, color='.2')

    for a in axes:
        sns.despine(ax=a, offset=3, trim=True)
    fig.savefig(out, dpi=300)
    print(f'saved {out}')


if __name__ == '__main__':
    main(REPO / 'notes' / 'figures' / 'cf_coupling_peaks.pdf')
