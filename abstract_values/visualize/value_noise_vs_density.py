#!/usr/bin/env python3
"""Does value-decoding noise shrink where values are dense? Pooled, trial-level.

The per-stimulus curves in decoding_precision_by_stimulus.pdf rest on 8 trials
per cell and are noisy. This pools every trial into one mixed model per ROI and
voxel count:

    log(|residual| + 0.5) ~ log density + condition,  random intercept and
                            density slope per subject

residual = decoded posterior mean minus the mean for that subject x condition x
stimulus (bias removed; scaled by sqrt(n / (n-1))). Density: Gaussian kernel
(SD 2 CHF) over the 23 values presented in that condition.

Density is of VALUES, per condition: the 23 orientations are the same (uniform)
in both sessions, so only value density differs. log value density at a
presented value is, up to the kernel, minus the log slope of the mapping at that
orientation.

Two versions of the test:
  across values   slope from all variation in density, within and between
                  conditions. Within a condition, density is tied to position
                  on the value axis, so anything position-dependent in the
                  decoder (range edges, compression) leaks in.
  orientation FE  adds orientation fixed effects, so the slope comes only from
                  the two conditions assigning different densities to the SAME
                  orientation (log-density ratio up to +-1.3). This is the clean
                  version.

Sign: efficient coding puts more precision where values are dense, so a
NEGATIVE slope is the efficient-coding direction. An orientation code read out
in value units predicts the same sign for a trivial reason -- where the
mapping is flat a given orientation error spans fewer CHF -- which is what the
V1 value decoder (the control) measures.

Caveat: a noisier decoder has more stimulus-independent noise, which flattens
any slope. NPCr decodes value worse than V1 does (r ~ .24 vs ~ .48), so a
shallower NPCr slope is not by itself evidence of a different code.

    python -m abstract_values.visualize.value_noise_vs_density
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import warnings

from abstract_values.connective_fields.gates import load_mappings
from abstract_values.connective_fields.plot_gates import letter  # sets rcParams
from abstract_values.utils.data import Subject

warnings.filterwarnings('ignore')
REPO = Path(__file__).resolve().parents[2]
NVS = [('50', '50 voxels'), ('250', '250 voxels'), ('0', 'CV-selected')]
ROI_COL = {'NPCr': '#C44E52', 'BensonV1': '#3B5BA5'}
ROI_LAB = {'NPCr': 'NPCr', 'BensonV1': 'V1 (control)'}
G = np.linspace(2, 42, 801)


def density(values, bw=2.):
    k = np.exp(-.5 * ((G[:, None] - values[None, :]) / bw) ** 2).sum(1)
    return k / (k.sum() * (G[1] - G[0]))


def main():
    ori, maps = load_mappings()
    den = {c: density(maps[c][(ori > 0) & (ori < 180)]) for c in maps}
    cond, rows, qual, bins = {}, [], [], []
    for nv, lab in NVS:
        d = pd.read_csv(REPO / 'notes' / 'data' / f'decoding_trials_nv{nv}.tsv', sep='\t',
                        dtype={'subject': str})
        d = d[d.quantity == 'value'].copy()
        key = list(zip(d.subject, d.session))
        for k in set(key) - set(cond):
            cond[k] = Subject(k[0]).get_mapping(int(k[1]))
        d['condition'] = [cond[k] for k in key]
        d['orientation'] = [round(float(np.interp(t, maps[c], ori)), 1)
                            for t, c in zip(d.true, d.condition)]
        d['logdens'] = [np.log(np.interp(t, G, den[c])) for t, c in zip(d.true, d.condition)]
        grp = d.groupby(['roi', 'subject', 'condition', 'true']).post_mean
        n = grp.transform('size')
        d['abs_r'] = ((d.post_mean - grp.transform('mean')) * np.sqrt(n / (n - 1))).abs()
        d['y'] = np.log(d.abs_r + .5)
        d['ld'] = d.logdens - d.logdens.mean()
        d['dbin'] = pd.qcut(d.logdens, 3, labels=['Sparse', 'Medium', 'Dense'])
        for roi in ROI_COL:
            h = d[d.roi == roi]
            f = smf.mixedlm('y ~ ld + C(condition)', h, groups=h.subject, re_formula='~ld').fit()
            rows.append(dict(nv=lab, roi=roi, test='across values', slope=f.params['ld'],
                             se=f.bse['ld'], p=f.pvalues['ld']))
            f = smf.mixedlm('y ~ ld + C(condition) + C(orientation)', h, groups=h.subject,
                            re_formula='~ld').fit()
            rows.append(dict(nv=lab, roi=roi, test='orientation FE', slope=f.params['ld'],
                             se=f.bse['ld'], p=f.pvalues['ld']))
            r = h.groupby('subject').apply(lambda x: np.corrcoef(x.true, x.post_mean)[0, 1],
                                           include_groups=False)
            qual += [dict(nv=lab, roi=roi, subject=s, r=v) for s, v in r.items()]
            b = h.groupby(['subject', 'dbin'], observed=True).abs_r.mean().reset_index()
            bins.append(b.assign(nv=lab, roi=roi))
        f = smf.mixedlm('y ~ ld * C(roi) + C(condition) * C(roi)', d, groups=d.subject,
                        re_formula='~ld').fit()
        k = [x for x in f.params.index if x.startswith('ld:')][0]
        rows.append(dict(nv=lab, roi='NPCr - V1', test='across values', slope=f.params[k],
                         se=f.bse[k], p=f.pvalues[k]))
        f = smf.mixedlm('y ~ ld * C(roi) + C(condition) * C(roi) + C(orientation) * C(roi)', d,
                        groups=d.subject, re_formula='~ld').fit()
        k = [x for x in f.params.index if x.startswith('ld:')][0]
        rows.append(dict(nv=lab, roi='NPCr - V1', test='orientation FE', slope=f.params[k],
                         se=f.bse[k], p=f.pvalues[k]))
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda x: f'{x:.3g}'))
    res.to_csv(REPO / 'notes' / 'data' / 'value_noise_vs_density.tsv', sep='\t', index=False)
    qual, bins = pd.DataFrame(qual), pd.concat(bins)

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.3), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1, 1.2, 1]})
    ax = axes[0]
    sns.pointplot(data=qual, x='nv', y='r', hue='roi', palette=ROI_COL, errorbar=('se', 1),
                  dodge=.3, markers='D', linestyles='none', ax=ax, legend=False,
                  err_kws={'linewidth': .9}, markersize=4)
    ax.set_xlabel('')
    ax.set_ylabel('Value decoding r (decoded vs true)')
    ax.set_ylim(0, .6)
    for roi, y in (('BensonV1', .57), ('NPCr', .5)):
        ax.text(.03, y, ROI_LAB[roi], color=ROI_COL[roi], transform=ax.get_yaxis_transform())
    letter(ax, 'a')

    ax = axes[1]
    b = bins[bins.nv == '50 voxels']
    sns.pointplot(data=b, x='dbin', y='abs_r', hue='roi', palette=ROI_COL, errorbar=('se', 1),
                  dodge=.25, ax=ax, legend=False, err_kws={'linewidth': .9}, markersize=4,
                  linewidth=1.2)
    ax.set_xlabel('Local value density (terciles)')
    ax.set_ylabel('|Residual| after bias (CHF)')
    ax.text(.97, .97, '50 voxels', transform=ax.transAxes, ha='right', va='top', color='.35',
            fontsize=6.5)
    letter(ax, 'b')

    ax = axes[2]
    order = [lab for _, lab in NVS]
    for i, lab in enumerate(order):
        for j, roi in enumerate(ROI_COL):
            for t, (dx, mfc) in {'across values': (-.1, None), 'orientation FE': (.1, 'w')}.items():
                r = res[(res.nv == lab) & (res.roi == roi) & (res.test == t)].iloc[0]
                x = i + (j - .5) * .42 + dx
                ax.errorbar(x, r.slope, 1.96 * r.se, fmt='D', ms=4, color=ROI_COL[roi],
                            mfc=mfc or ROI_COL[roi], elinewidth=.9, capsize=0)
        pi = res[(res.nv == lab) & (res.roi == 'NPCr - V1') & (res.test == 'across values')].iloc[0].p
        ax.text(i, .045, f'Δ p = {pi:.2g}'.replace('0.', '.'), ha='center', fontsize=5.5, color='.35')
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel('Slope of log |residual|\non log density (95% CI)')
    ax.text(.03, .03, 'Negative: less noise where dense', transform=ax.transAxes, fontsize=6,
            color='.35')
    ax.plot(.62, .93, 'D', ms=3.5, color='.4', transform=ax.transAxes, clip_on=False)
    ax.text(.66, .93, 'Across values', transform=ax.transAxes, va='center', fontsize=6, color='.4')
    ax.plot(.62, .84, 'D', ms=3.5, color='.4', mfc='w', transform=ax.transAxes, clip_on=False)
    ax.text(.66, .84, 'Same orientation', transform=ax.transAxes, va='center', fontsize=6, color='.4')
    ax.set_ylim(-.17, .05)
    letter(ax, 'c')
    for a in axes:
        sns.despine(ax=a, offset=3, trim=True)
    out = REPO / 'notes' / 'figures' / 'value_noise_vs_density.pdf'
    fig.savefig(out)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
