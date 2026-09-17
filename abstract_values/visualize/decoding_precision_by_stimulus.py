#!/usr/bin/env python3
"""Decoding precision across the stimulus range, with central tendency taken out.

``decoding_scatter_error.pdf`` plots absolute decoding error per stimulus. For
value that curve is U-shaped, mostly because decoded values regress toward the
middle of the range: extreme values get large errors whatever the precision.
Absolute error mixes that bias with noise. This figure separates them.

Per subject x condition x presented stimulus (8 trials per session):

  bias       mean(decoded) - true                          systematic pull
  noise      SD of decoded across trials of the same stimulus  bias-free spread
  threshold  noise / slope, with slope the subject x condition regression
             slope of decoded on true -- the stimulus difference the decoder
             can resolve. Central tendency compresses decoded values, which
             shrinks their spread too, so raw noise alone flatters compressed
             ranges.
  posterior  mean posterior SD of the decoder itself

Orientation (V1) uses axial circular statistics and has no range to regress
into, so its threshold equals its noise.

Decoders: leave-one-run-out, joint encoding models over both sessions (no
session shift): von Mises basis for orientation, value PRF for value; 250
voxels, spherical noise, lambda 0.1. Per-trial summary: notes/data/decoding_trials_nv250.tsv.

Tests for value (NPCr):
  condition x density: per subject, r across values between the CDF - inverse
  difference in log precision (1/noise, 1/threshold, 1/posterior SD) and the
  difference in log value density. Efficient coding predicts r > 0 (so does an
  orientation code, see notes/report_2026-09-17.md).
Control: the same for value decoded from V1. V1 codes orientation, so whatever
density relation it shows is what the orientation-to-value warp produces alone;
NPCr has to exceed it before its relation says anything about value coding.
For orientation (V1): cos 2theta / cos 4theta harmonics of log noise and log
posterior SD (horizontal vs vertical, cardinal vs oblique).

    python -m abstract_values.visualize.decoding_precision_by_stimulus
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.connective_fields.gates import load_mappings
from abstract_values.connective_fields.plot_gates import letter  # sets rcParams
from abstract_values.utils.data import Subject

REPO = Path(__file__).resolve().parents[2]
COND_COL = {'cdf': '#3B5BA5', 'inverse_cdf': '#5D8C3F'}
COND_LAB = {'cdf': 'CDF', 'inverse_cdf': 'Inverse CDF'}
GRID = np.arange(6, 38.5, .5)


def circ_mean_sd(deg):
    z = np.exp(2j * np.deg2rad(deg)).mean()
    return np.rad2deg(np.angle(z) / 2) % 180, np.rad2deg(np.sqrt(-2 * np.log(np.clip(np.abs(z), 1e-12, 1))) / 2)


def load():
    d = pd.read_csv(REPO / 'notes' / 'data' / 'decoding_trials_nv250.tsv', sep='\t',
                    dtype={'subject': str})
    d['condition'] = [Subject(s).get_mapping(int(ses)) for s, ses in zip(d.subject, d.session)]
    return d


def per_stimulus(d, quantity, roi):
    g = d[(d.quantity == quantity) & (d.roi == roi)].copy()
    rows = []
    for (s, c), h in g.groupby(['subject', 'condition']):
        if quantity == 'value':
            slope = np.polyfit(h.true, h.post_mean, 1)[0]
        for t, k in h.groupby('true'):
            if quantity == 'value':
                bias, noise = k.post_mean.mean() - t, k.post_mean.std()
                thr = noise / slope if slope > .05 else np.nan
            else:
                m, noise = circ_mean_sd(k.post_mean)
                bias = (m - t + 90) % 180 - 90
                thr = noise
            rows.append({'subject': s, 'condition': c, 'true': t, 'bias': bias, 'noise': noise,
                         'threshold': thr, 'post_sd': k.post_sd.mean(),
                         'slope': slope if quantity == 'value' else np.nan})
    return pd.DataFrame(rows)


def density(values, bw=2.):
    k = np.exp(-.5 * ((GRID[:, None] - values[None, :]) / bw) ** 2).sum(1)
    return k / (k.sum() * (GRID[1] - GRID[0]))


def density_test(ps, col, ddiff):
    rs = {}
    for s, g in ps.groupby('subject'):
        prof = {}
        for c in COND_COL:
            h = g[(g.condition == c) & np.isfinite(g[col]) & (g[col] > 0)].sort_values('true')
            if len(h) < 8:
                break
            prof[c] = np.interp(GRID, h.true, -np.log(h[col]))
        if len(prof) == 2:
            rs[s] = np.corrcoef(prof['cdf'] - prof['inverse_cdf'], ddiff)[0, 1]
    return pd.Series(rs)


def harmonic_test(ps, col):
    out = {}
    for name, f in (('cos 2θ', lambda t: np.cos(2 * t)), ('cos 4θ', lambda t: np.cos(4 * t))):
        b = []
        for s, g in ps.groupby('subject'):
            g = g.groupby('true')[col].mean()
            t = np.deg2rad(g.index.values)
            X = np.column_stack([np.ones_like(t), np.cos(2 * t), np.sin(2 * t), np.cos(4 * t), np.sin(4 * t)])
            beta = np.linalg.lstsq(X, np.log(g.values), rcond=None)[0]
            b.append(beta[1] if name == 'cos 2θ' else beta[3])
        b = np.array(b)
        out[name] = (b.mean(), b.std(ddof=1) / np.sqrt(len(b)), stats.ttest_1samp(b, 0).pvalue)
    return out


def line(ax, ps, col, xticks, xlabel, ylabel):
    for c in COND_COL:
        sns.lineplot(data=ps[ps.condition == c], x='true', y=col, errorbar=('se', 1),
                     color=COND_COL[c], ax=ax, lw=1.3, err_kws={'lw': 0, 'alpha': .2})
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main():
    ori, maps = load_mappings()
    d = load()
    pv = {c: maps[c][(ori > 0) & (ori < 180)] for c in maps}
    ddiff = np.log(density(pv['cdf'])) - np.log(density(pv['inverse_cdf']))

    ori_ps = per_stimulus(d, 'gabor', 'BensonV1')
    val_ps = per_stimulus(d, 'value', 'NPCr')
    v1val_ps = per_stimulus(d, 'value', 'BensonV1')
    n = d.subject.nunique()

    print('V1 orientation harmonics of log noise / log posterior SD (negative c = more precise at 0°):')
    for col in ('noise', 'post_sd'):
        print(f'  {col}: ' + ', '.join(f'{k} {m:+.3f} ± {se:.3f} (p = {p:.2g})'
                                        for k, (m, se, p) in harmonic_test(ori_ps, col).items()))
    print('NPCr value: CDF − inverse log-precision difference vs log density difference:')
    tests = {}
    for roi, ps in (('NPCr', val_ps), ('BensonV1', v1val_ps)):
        for col in ('noise', 'threshold', 'post_sd'):
            r = density_test(ps, col, ddiff)
            t = stats.ttest_1samp(r, 0)
            tests[roi, col] = (r, t.pvalue)
            print(f'  {roi} {col}: r = {r.mean():+.3f} ± {r.sem():.3f}, t{len(r)-1} = {t.statistic:.2f}, p = {t.pvalue:.2g}')
    for col in ('noise', 'threshold', 'post_sd'):
        a, b = tests['NPCr', col][0], tests['BensonV1', col][0]
        i = a.index.intersection(b.index)
        print(f'  NPCr - V1 {col}: {(a[i] - b[i]).mean():+.3f}, paired p = {stats.ttest_rel(a[i], b[i]).pvalue:.2g}')
    sl = val_ps.drop_duplicates(['subject', 'condition']).pivot(index='subject', columns='condition', values='slope')
    print(f'  decoded-on-true slope: CDF {sl.cdf.mean():.2f}, inverse {sl.inverse_cdf.mean():.2f} '
          f'(paired p = {stats.ttest_rel(sl.cdf, sl.inverse_cdf).pvalue:.2g})')
    for col in ('noise', 'threshold', 'post_sd'):
        m = val_ps.groupby(['subject', 'condition'])[col].mean().unstack()
        print(f'  mean {col}: CDF {m.cdf.mean():.2f}, inverse {m.inverse_cdf.mean():.2f} '
              f'(paired p = {stats.ttest_rel(m.cdf, m.inverse_cdf, nan_policy="omit").pvalue:.2g})')

    fig, axes = plt.subplots(3, 4, figsize=(7.25, 5.7), constrained_layout=True)
    # row 1: orientation from V1
    ot = [0, 45, 90, 135, 180]
    line(axes[0, 0], ori_ps, 'bias', ot, 'True orientation (°)', 'Bias (°)')
    axes[0, 0].axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    line(axes[0, 1], ori_ps, 'noise', ot, 'True orientation (°)', 'Across-trial SD (°)')
    axes[0, 2].axis('off')
    axes[0, 2].text(.5, .5, 'Orientation has no range\nto regress into:\nthreshold = across-trial SD',
                    ha='center', va='center', fontsize=6.5, color='.45', transform=axes[0, 2].transAxes)
    line(axes[0, 3], ori_ps, 'post_sd', ot, 'True orientation (°)', 'Posterior SD (°)')
    axes[0, 0].text(.03, .97, f'V1 · orientation (n = {n})', transform=axes[0, 0].transAxes,
                    va='top', fontsize=7, fontweight='bold', color='.2')

    # row 2: value from NPCr
    vt = [2, 12, 22, 32, 42]
    line(axes[1, 0], val_ps, 'bias', vt, 'True value (CHF)', 'Bias (CHF)')
    axes[1, 0].axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    line(axes[1, 1], val_ps, 'noise', vt, 'True value (CHF)', 'Across-trial SD (CHF)')
    line(axes[1, 2], val_ps, 'threshold', vt, 'True value (CHF)', 'Threshold (SD / slope, CHF)')
    line(axes[1, 3], val_ps, 'post_sd', vt, 'True value (CHF)', 'Posterior SD (CHF)')
    axes[1, 0].text(.03, .03, 'NPCr · value', transform=axes[1, 0].transAxes, fontsize=7,
                    fontweight='bold', color='.2')
    axes[1, 0].text(.97, .97, 'CDF', color=COND_COL['cdf'], transform=axes[1, 0].transAxes,
                    ha='right', va='top')
    axes[1, 0].text(.97, .87, 'Inverse CDF', color=COND_COL['inverse_cdf'],
                    transform=axes[1, 0].transAxes, ha='right', va='top')
    # row 3: value from V1 (control: an orientation code read out in value space)
    line(axes[2, 0], v1val_ps, 'bias', vt, 'True value (CHF)', 'Bias (CHF)')
    axes[2, 0].axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    line(axes[2, 1], v1val_ps, 'noise', vt, 'True value (CHF)', 'Across-trial SD (CHF)')
    line(axes[2, 2], v1val_ps, 'threshold', vt, 'True value (CHF)', 'Threshold (SD / slope, CHF)')
    line(axes[2, 3], v1val_ps, 'post_sd', vt, 'True value (CHF)', 'Posterior SD (CHF)')
    axes[2, 0].text(.03, .03, 'V1 · value (control)', transform=axes[2, 0].transAxes, fontsize=7,
                    fontweight='bold', color='.2')
    for row, roi in ((1, 'NPCr'), (2, 'BensonV1')):
        for ax, col in ((axes[row, 1], 'noise'), (axes[row, 2], 'threshold'), (axes[row, 3], 'post_sd')):
            r, p = tests[roi, col]
            ax.text(.03, .03, f'Tracks density: r = {r.mean():+.2f}, p = {p:.1g}'.replace('0.', '.'),
                    transform=ax.transAxes, fontsize=5.5, color='.3')
    for ax, lab in zip([axes[0, 0], axes[0, 1], axes[0, 3], axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3],
                        axes[2, 0], axes[2, 1], axes[2, 2], axes[2, 3]],
                       'abcdefghijk'):
        letter(ax, lab)
        sns.despine(ax=ax, offset=3, trim=True)
    out = REPO / 'notes' / 'figures' / 'decoding_precision_by_stimulus.pdf'
    fig.savefig(out)
    val_ps.assign(quantity='value', roi='NPCr').to_csv(
        REPO / 'notes' / 'data' / 'decoding_precision_by_stimulus.tsv', sep='\t', index=False)
    ori_ps.assign(quantity='orientation', roi='BensonV1').to_csv(
        REPO / 'notes' / 'data' / 'decoding_precision_by_stimulus.tsv', sep='\t', index=False,
        mode='a', header=False)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
