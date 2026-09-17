#!/usr/bin/env python3
"""Preferred value vs most-connected V1 orientation, for NPCr voxels.

Top row -- coupling maps (group mean, n = 30). Each column is a 2-CHF bin of
NPCr preferred value; each row a V1 orientation from the inverted encoding
model (24 basis functions, kappa 16). Colour: mean coupling of those voxels'
residuals to that orientation, after removing the session's NPCr-wide profile.
Lines: theta*, the orientation worth that value under each mapping.
  a  CDF sessions        b  Inverse-CDF sessions
  c  a - b: stable coupling cancels; what the mapping change moved remains
Bottom row:
  d  The difference map the mappings predict from each voxel's value tuning
  e  Per-subject correlation between observed (c) and predicted (d) maps
  f  Per voxel: shift of the most-connected orientation (circular centroid)
     against the predicted shift of theta*

  python -m abstract_values.connective_fields.plot_peaks
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.connective_fields.gates import load_mappings, lognormal_mode_fwhm, zscore
from abstract_values.connective_fields.plot_coupling_explainer import CUR_COL, MAP_COL, ROOT
from abstract_values.connective_fields.plot_gates import letter
from abstract_values.connective_fields.test_coupling import IEM_GRID

REPO = Path(__file__).resolve().parents[2]


def wrap(d):
    return (d + 90) % 180 - 90


def theta_lines(ax, maps, ori, which=('cdf', 'inverse_cdf')):
    vv = np.linspace(2, 42, 400)
    for c in which:
        th = np.interp(vv, maps[c], ori)
        ax.plot(vv, th, color=MAP_COL[c], lw=1.1)


def heat(ax, m, vmax, cmap='vlag'):
    """m: DataFrame index orientation, columns value-bin centre."""
    v = m.columns.to_numpy()
    o = m.index.to_numpy()
    ve = np.r_[v - 1, v[-1] + 1]
    oe = np.r_[o - 2.5, o[-1] + 2.5]
    return ax.pcolormesh(ve, oe, m.to_numpy(), cmap=cmap, vmin=-vmax, vmax=vmax,
                         rasterized=True)


def predicted_map(peaks, maps, ori):
    """Mean over voxels per 2-CHF bin of z(f(m_cdf(theta))) - z(f(m_inv(theta)))."""
    rows = []
    for (s, b), g in peaks.groupby(['subject', 'value_bin']):
        d = 0
        for c, sign in (('cdf', 1), ('inverse_cdf', -1)):
            p = lognormal_mode_fwhm(np.interp(IEM_GRID, ori, maps[c])[:, None],
                                    g['mode'].to_numpy()[None, :], g['fwhm'].to_numpy()[None, :])
            d = d + sign * zscore(p - p.mean(0), axis=0)
        for k, th in enumerate(IEM_GRID):
            rows.append({'subject': s, 'value_bin': b, 'orientation': th,
                         'predicted': d[k].mean()})
    return pd.DataFrame(rows)


def main(out, variant='peaks'):
    ori, maps = load_mappings()
    files = sorted((ROOT / variant).glob('sub-*/sub-*_desc-maps.tsv'))
    mp = pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str}) for f in files],
                   ignore_index=True)
    peaks = pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str}) for f in
                       sorted((ROOT / variant).glob('sub-*/sub-*_desc-peaks.tsv.gz'))],
                      ignore_index=True)
    peaks['value_bin'] = np.clip(np.floor((peaks['mode'] - 2) / 2) * 2 + 3, 3, 41)
    print(f'{mp.subject.nunique()} subjects, {len(peaks)} voxels')

    wide = mp.pivot_table(index=['subject', 'condition', 'orientation'], columns='value_bin',
                          values='cf')
    grp = {c: wide.xs(c, level='condition').groupby('orientation').mean() for c in maps}
    diff_s = (wide.xs('cdf', level='condition') - wide.xs('inverse_cdf', level='condition'))
    diff = diff_s.groupby('orientation').mean()

    pred = predicted_map(peaks, maps, ori)
    pred_s = pred.pivot_table(index=['subject', 'orientation'], columns='value_bin',
                              values='predicted')
    pred_g = pred_s.groupby('orientation').mean()

    r = pd.Series({s: np.corrcoef(diff_s.xs(s).to_numpy().ravel(),
                                  pred_s.xs(s).reindex_like(diff_s.xs(s)).to_numpy().ravel())[0, 1]
                   for s in diff_s.index.get_level_values('subject').unique()})
    t = stats.ttest_1samp(r, 0)
    print(f'map r(observed diff, predicted diff) = {r.mean():.3f} ± {r.sem():.3f}, '
          f't{len(r)-1} = {t.statistic:.2f}, p = {t.pvalue:.2g}')

    peaks['pred_shift'] = wrap(np.interp(peaks['mode'], maps['cdf'], ori)
                               - np.interp(peaks['mode'], maps['inverse_cdf'], ori))
    peaks['obs_shift'] = wrap(peaks['centroid_specific_cdf'] - peaks['centroid_specific_inverse_cdf'])
    slope = peaks.groupby('subject').apply(
        lambda g: np.polyfit(g.pred_shift, g.obs_shift, 1)[0], include_groups=False)
    ts = stats.ttest_1samp(slope, 0)
    print(f'per-voxel centroid shift ~ predicted shift: slope {slope.mean():.3f} ± '
          f'{slope.sem():.3f}, t = {ts.statistic:.2f}, p = {ts.pvalue:.2g}')

    fig = plt.figure(figsize=(7.25, 4.9), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1], hspace=.04)
    ax_a, ax_b, ax_c = top.subplots(1, 3)
    ax_d, ax_e, ax_f = bottom.subplots(1, 3, gridspec_kw={'width_ratios': [1, .55, 1]})

    vmax = np.quantile(np.abs(np.r_[grp['cdf'].to_numpy().ravel(),
                                    grp['inverse_cdf'].to_numpy().ravel()]), .98)
    for ax, c, lab in ((ax_a, 'cdf', 'a'), (ax_b, 'inverse_cdf', 'b')):
        im = heat(ax, grp[c], vmax)
        theta_lines(ax, maps, ori)
        ax.set_title('CDF sessions' if c == 'cdf' else 'Inverse-CDF sessions', fontsize=8,
                     color=MAP_COL[c])
        letter(ax, lab)
    top.colorbar(im, ax=[ax_a, ax_b], shrink=.7, label='Coupling (r)', pad=.01)
    dmax = np.quantile(np.abs(diff.to_numpy()), .98)
    im = heat(ax_c, diff, dmax)
    theta_lines(ax_c, maps, ori)
    ax_c.set_title('CDF − inverse CDF', fontsize=8)
    top.colorbar(im, ax=ax_c, shrink=.7, label='Δ coupling (r)', pad=.01)
    letter(ax_c, 'c')

    pmax = np.quantile(np.abs(pred_g.to_numpy()), .98)
    im = heat(ax_d, pred_g, pmax)
    theta_lines(ax_d, maps, ori)
    ax_d.set_title('Predicted CDF − inverse CDF', fontsize=8)
    bottom.colorbar(im, ax=ax_d, shrink=.7, label='Δ prediction (z)', pad=.01)
    letter(ax_d, 'd')
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xticks([2, 12, 22, 32, 42])
        ax.set_yticks([0, 45, 90, 135, 180])
        ax.set_xlabel('NPCr preferred value (CHF)')
        ax.set_ylabel('V1 orientation (°)')
        ax.set_xlim(2, 42)
        ax.set_ylim(-2.5, 177.5)
    for ax in (ax_b, ax_c):
        ax.set_ylabel('')
    ax_a.text(40, 8, 'θ* CDF', color=MAP_COL['cdf'], ha='right', fontsize=6.5)
    ax_a.text(4, 165, 'θ* inverse CDF', color=MAP_COL['inverse_cdf'], fontsize=6.5)

    rng = np.random.default_rng(4)
    ax_e.scatter(rng.uniform(-.12, .12, len(r)), r, s=7, color=CUR_COL, alpha=.45, lw=0)
    ax_e.errorbar(0, r.mean(), r.sem(), color='.15', lw=.9, zorder=3)
    ax_e.plot(0, r.mean(), 'D', ms=5, mfc=CUR_COL, mec='.15', mew=1, zorder=4)
    ax_e.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax_e.set_xlim(-.5, .5)
    ax_e.set_xticks([])
    ax_e.set_ylabel('r(observed c, predicted d)')
    ax_e.text(0, ax_e.get_ylim()[1], f'p = {t.pvalue:.1g}'.replace('0.', '.'), ha='center',
              va='bottom', fontsize=6.5, color='.3')
    letter(ax_e, 'e')

    hb = ax_f.hexbin(peaks.pred_shift, peaks.obs_shift, gridsize=(24, 18), cmap='Greys',
                     mincnt=1, linewidths=0, rasterized=True)
    bins = np.linspace(-24, 24, 7)
    peaks['pbin'] = pd.cut(peaks.pred_shift, bins, labels=(bins[:-1] + bins[1:]) / 2)
    sm = peaks.groupby(['subject', 'pbin'], observed=True).obs_shift.mean().unstack()
    ax_f.errorbar(sm.columns.astype(float), sm.mean(), sm.sem(), fmt='o', ms=3.5, mfc='w',
                  mec='k', mew=.9, color='k', lw=.8, zorder=4)
    ax_f.plot([-24, 24], [-24, 24], color=CUR_COL, lw=1)
    ax_f.axhline(0, color='.6', lw=.6, ls='--')
    ax_f.set_xlabel('Predicted shift of θ* (°)')
    ax_f.set_ylabel('Shift of most-connected\norientation (°)')
    ax_f.set_yticks([-90, -45, 0, 45, 90])
    ax_f.text(.97, .03, f'Slope {slope.mean():.2f}, p = {ts.pvalue:.1g}'.replace('0.', '.'),
              transform=ax_f.transAxes, ha='right', fontsize=6.5, color='.2')
    ax_f.text(20, 40, 'Identity', color=CUR_COL, fontsize=6.5, ha='right')
    letter(ax_f, 'f')

    for ax in (ax_e, ax_f):
        sns.despine(ax=ax, offset=3, trim=True)
    fig.savefig(out, dpi=300)
    print(f'saved {out}')


if __name__ == '__main__':
    import sys
    variant = sys.argv[1] if len(sys.argv) > 1 else 'peaks'
    main(REPO / 'notes' / 'figures' / f'cf_coupling_{variant}.pdf', variant)
