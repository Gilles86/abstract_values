#!/usr/bin/env python3
"""Expected uncertainty against value density, on a dense decoding grid.

The expected-uncertainty (EU) result -- NPCr most precise where values are
SPARSE, the opposite of efficient coding -- is compared here across the choices
the simulation makes: the noise model, the decoding grid and the decoding prior.
Each variant simulates from the same session-shift fits (100 voxels, 1000
repeats per stimulus) and differs only in how the posterior is computed.
The density test:

  across values   per subject, r between the CDF - inverse difference in log
                  precision (1 / sd_E) and the difference in log value density
  same orientation  the two conditions assign a different value, and so a
                  different density, to the SAME orientation; paired by
                  orientation this removes anything tied to position on the
                  value axis

Efficient coding predicts positive r (more precision where values are dense);
a fixed orientation code predicts the same, for the trivial reason that a given
orientation error spans fewer CHF where the mapping is flat.

    python -m abstract_values.visualize.eu_density_flatgrid
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.connective_fields.gates import load_mappings
from abstract_values.connective_fields.plot_gates import letter  # sets rcParams
from abstract_values.utils.data import BIDS_FOLDER, Subject

REPO = Path(__file__).resolve().parents[2]
DERIV = BIDS_FOLDER / 'derivatives' / 'encoding_models' / 'aprf-session-shift'
COND_COL = {'cdf': '#3B5BA5', 'inverse_cdf': '#5D8C3F'}
GRID = np.arange(6, 38.5, .5)


def density(values, bw=2.):
    k = np.exp(-.5 * ((GRID[:, None] - values[None, :]) / bw) ** 2).sum(1)
    return k / (k.sum() * (GRID[1] - GRID[0]))


VARIANTS = [('Full noise,\npresented values', ''),
            ('Spherical,\npresented values', '_noise-spherical'),
            ('Spherical,\ndense grid', '_noise-spherical_prior-flat'),
            ('Spherical, dense grid,\nobjective prior', '_noise-spherical_prior-objective')]


def load(tag):
    rows = []
    for f in sorted(DERIV.glob(f'sub-*/ses-*/func/sub-*_mask-NPCr_nvoxels-100_nsims-1000'
                               f'{tag}_desc-expected_decoded_pe.tsv')):
        sub, ses = f.name.split('_')[0][4:], int(f.name.split('_')[1][4:])
        d = pd.read_csv(f, sep='\t')
        sd = d['sd_E'] if 'sd_E' in d else np.sqrt(d['var_E'])
        rows.append(pd.DataFrame({'subject': sub, 'session': ses,
                                  'condition': Subject(sub).get_mapping(ses),
                                  'value': d['value'], 'sd_E': sd}))
    if not rows:
        raise SystemExit(f'no EU files matching {tag} under {DERIV}')
    return pd.concat(rows, ignore_index=True)


def across_values(eu, ddiff):
    rs = {}
    for s, g in eu.groupby('subject'):
        prof = {}
        for c in COND_COL:
            h = g[g.condition == c].groupby('value').sd_E.mean()
            if len(h) < 8:
                break
            prof[c] = np.interp(GRID, h.index.values, -np.log(h.values))
        if len(prof) == 2:
            rs[s] = np.corrcoef(prof['cdf'] - prof['inverse_cdf'], ddiff)[0, 1]
    return pd.Series(rs)


def same_orientation(eu, ori, maps):
    """Per subject, r over the 23 orientations between the log-precision
    difference and the log-density difference the mappings impose there."""
    o = ori[(ori > 0) & (ori < 180)]
    v = {c: np.interp(o, ori, maps[c]) for c in COND_COL}
    den = {c: np.log(np.interp(v[c], GRID, density(v[c]))) for c in COND_COL}
    dd = den['cdf'] - den['inverse_cdf']
    rs = {}
    for s, g in eu.groupby('subject'):
        prec = {}
        for c in COND_COL:
            h = g[g.condition == c].groupby('value').sd_E.mean()
            if len(h) < 8:
                break
            prec[c] = -np.log(np.interp(v[c], h.index.values, h.values))
        if len(prec) == 2:
            rs[s] = np.corrcoef(prec['cdf'] - prec['inverse_cdf'], dd)[0, 1]
    return pd.Series(rs)


def main():
    ori, maps = load_mappings()
    pv = {c: maps[c][(ori > 0) & (ori < 180)] for c in maps}
    ddiff = np.log(density(pv['cdf'])) - np.log(density(pv['inverse_cdf']))

    res, data = [], {}
    for label, tag in VARIANTS:
        try:
            eu = load(tag)
        except SystemExit as e:
            print(e)
            continue
        data[label] = eu
        for name, r in (('across values', across_values(eu, ddiff)),
                        ('same orientation', same_orientation(eu, ori, maps))):
            t = stats.ttest_1samp(r, 0)
            res.append(dict(variant=label.replace('\n', ' '), test=name, n=len(r), r=r.mean(),
                            sem=r.sem(), t=t.statistic, p=t.pvalue))
            print(f'{label.replace(chr(10), " "):40s} {name:17s} n = {len(r):2d}  '
                  f'r = {r.mean():+.3f} ± {r.sem():.3f}, p = {t.pvalue:.2g}')
    res = pd.DataFrame(res)
    res.to_csv(REPO / 'notes' / 'data' / 'eu_density_flatgrid.tsv', sep='\t', index=False)

    n = len(data)
    fig, axes = plt.subplots(1, n + 1, figsize=(1.75 * (n + 1) + .6, 2.4),
                             constrained_layout=True)
    for ax, (label, eu) in zip(axes, data.items()):
        for c in COND_COL:
            sns.lineplot(data=eu[eu.condition == c], x='value', y='sd_E', errorbar=('se', 1),
                         color=COND_COL[c], ax=ax, lw=1.2, err_kws={'lw': 0, 'alpha': .2})
        ax.set_xticks([2, 22, 42])
        ax.set_xlabel('True value (CHF)')
        ax.set_ylabel('Expected decoded SD (CHF)')
        ax.text(.03, 1.0, label, transform=ax.transAxes, va='bottom', fontsize=6.5, color='.3')
    axes[0].text(.97, .97, 'CDF', color=COND_COL['cdf'], transform=axes[0].transAxes,
                 ha='right', va='top', fontsize=6.5)
    axes[0].text(.97, .86, 'Inverse CDF', color=COND_COL['inverse_cdf'], fontsize=6.5,
                 transform=axes[0].transAxes, ha='right', va='top')
    for ax in axes[1:n]:
        ax.set_ylabel('')

    ax = axes[-1]
    rng = np.random.default_rng(0)
    for i, (label, eu) in enumerate(data.items()):
        for j, (name, r) in enumerate((('across values', across_values(eu, ddiff)),
                                       ('same orientation', same_orientation(eu, ori, maps)))):
            x = i + (j - .5) * .34
            col = '.45' if j == 0 else '#C44E52'
            ax.scatter(x + rng.uniform(-.07, .07, len(r)), r, s=5, color=col, alpha=.35, lw=0)
            ax.errorbar(x, r.mean(), r.sem(), color='.15', lw=.9, zorder=3)
            ax.plot(x, r.mean(), 'D', ms=4.5, mfc=col, mec='.15', mew=1, zorder=4)
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    short = ['Full\nnoise', 'Spherical', '+ dense\ngrid', '+ objective\nprior'][:n]
    ax.set_xticks(range(n), short, fontsize=6)
    ax.set_ylabel('r(precision difference,\ndensity difference)')
    ax.text(.02, .03, 'Grey: across values   Red: same orientation', transform=ax.transAxes,
            fontsize=5.5, color='.35')
    ax.text(.02, .96, 'Efficient coding predicts > 0', transform=ax.transAxes, fontsize=5.5,
            color='.35', va='top')
    for a, lab in zip(axes, 'abcde'):
        letter(a, lab)
        sns.despine(ax=a, offset=3, trim=True)
    out = REPO / 'notes' / 'figures' / 'eu_density_flatgrid.pdf'
    fig.savefig(out)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
