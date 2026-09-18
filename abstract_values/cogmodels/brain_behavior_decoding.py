#!/usr/bin/env python3
"""Cognitive-model noise parameters against DECODING ACCURACY, stage by stage.

``brain_behavior.py`` uses model-derived precision (expected decoded SD) or the
extent of tuned cortex. This uses the decoders' actual out-of-sample accuracy on
real trials -- no simulation:

    kappa_r    (perceptual precision)  <->  orientation decoded from V1
    1/sigma_rep (value precision)      <->  value decoded from NPCr

and the crossed pairs as the control. Two accuracy measures per ROI:

    r     true vs decoded (Pearson for value; resultant length of the axial
          angular error for orientation)
    -MAE  minus the mean absolute error (CHF or degrees)

Decoders: leave-one-run-out, spherical noise, lambda 0.1; orientation from the
von Mises basis set, value from either the single log-Gaussian bump ('loggauss')
or its basis-set analogue ('weighted'). Voxel counts 50 / 250 / CV-selected come
from notes/data/decoding_trials_nv*.tsv, the matched basis-set pair from
notes/data/decoding_trials_weighted.tsv (100 voxels).

Correlations integrate over posterior draws and bootstrap subjects
(brain_behavior.draws_correlation), so the interval carries both the parameter's
posterior width and the n = 26 sampling error.

Writes notes/data/brain_behavior_decoding.tsv and
notes/figures/brain_behavior_decoding.pdf.

    python -m abstract_values.cogmodels.brain_behavior_decoding
"""
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.cogmodels.brain_behavior import draws_correlation
from abstract_values.connective_fields.plot_gates import letter  # sets rcParams

REPO = Path(__file__).resolve().parents[2]
TRACE = Path('/data/ds-abstractvalue/derivatives/cogmodels/efficient_coding_sequential_trace.nc')
MATCH_C, CROSS_C = '#3B5BA5', '#9C9C9C'


def accuracy(df, quantity):
    """Per subject: r with the truth and mean absolute error."""
    out = {}
    for s, g in df.groupby('subject'):
        if quantity == 'value':
            r = np.corrcoef(g.true, g.post_mean)[0, 1]
            mae = (g.post_mean - g.true).abs().mean()
        else:
            e = np.deg2rad(2 * (g.post_mean - g.true))
            r = np.abs(np.exp(1j * e).mean())
            mae = ((g.post_mean - g.true + 90) % 180 - 90).abs().mean()
        out[int(''.join(c for c in str(s) if c.isdigit()))] = (r, mae)
    return pd.DataFrame(out, index=['r', 'mae']).T


def load_sets():
    """{label: {'orientation': acc, 'value': acc}} for each decoder variant."""
    sets = {}
    for nv, lab in (('50', '50 voxels'), ('250', '250 voxels'), ('0', 'CV-selected')):
        d = pd.read_csv(REPO / 'notes' / 'data' / f'decoding_trials_nv{nv}.tsv', sep='\t',
                        dtype={'subject': str})
        sets[f'Single bump, {lab}'] = {
            'orientation': accuracy(d[(d.quantity == 'gabor') & (d.roi == 'BensonV1')], 'gabor'),
            'value': accuracy(d[(d.quantity == 'value') & (d.roi == 'NPCr')], 'value')}
    w = pd.read_csv(REPO / 'notes' / 'data' / 'decoding_trials_weighted.tsv', sep='\t',
                    dtype={'subject': str})
    d50 = pd.read_csv(REPO / 'notes' / 'data' / 'decoding_trials_nv50.tsv', sep='\t',
                      dtype={'subject': str})
    sets['Basis set, 100 voxels'] = {
        'orientation': accuracy(d50[(d50.quantity == 'gabor') & (d50.roi == 'BensonV1')], 'gabor'),
        'value': accuracy(w[(w.model == 'weighted') & (w.roi == 'NPCr')], 'value')}
    return sets


def main():
    post = az.from_netcdf(TRACE).posterior
    subs = [int(x) for x in post.kappa_r.coords['subject'].values]
    pars = {'κ_r (perceptual precision)': post.kappa_r.values.reshape(-1, len(subs)),
            '1/σ_rep (value precision)': 1 / post.sigma_rep.values.reshape(-1, len(subs))}
    stage = {'κ_r (perceptual precision)': 'orientation', '1/σ_rep (value precision)': 'value'}
    roi = {'orientation': 'V1 orientation', 'value': 'NPCr value'}

    rows = []
    for label, acc in load_sets().items():
        for par, draws in pars.items():
            for q in ('orientation', 'value'):
                a = acc[q].reindex(subs)
                for meas, y in (('r', a.r.values), ('−MAE', -a.mae.values)):
                    keep = np.isfinite(y)
                    rho, lo, hi = draws_correlation(draws[:, keep], y[keep])
                    rows.append(dict(decoder=label, parameter=par, neural=roi[q], measure=meas,
                                     matched=stage[par] == q, rho=rho, lo=lo, hi=hi,
                                     n=int(keep.sum())))
    res = pd.DataFrame(rows)
    res.to_csv(REPO / 'notes' / 'data' / 'brain_behavior_decoding.tsv', sep='\t', index=False)
    print(res.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    decoders = list(dict.fromkeys(res.decoder))
    fig, axes = plt.subplots(1, len(decoders), figsize=(7.25, 2.6), constrained_layout=True,
                             sharex=True)
    for ax, dec in zip(np.atleast_1d(axes), decoders):
        g = res[res.decoder == dec].reset_index(drop=True)
        labels = []
        for i, row in g.iterrows():
            col = MATCH_C if row.matched else CROSS_C
            ax.plot([row.lo, row.hi], [i, i], color=col, lw=1.2)
            ax.plot(row.rho, i, 'o', color=col, ms=3.5)
            labels.append(f'{row.parameter.split(" (")[0]} ~ {row.neural} [{row.measure}]')
        ax.axvline(0, color='.7', lw=.6, ls='--', zorder=0)
        ax.set_yticks(range(len(labels)), labels if ax is np.atleast_1d(axes)[0] else [],
                      fontsize=5.5)
        ax.set_ylim(len(labels) - .5, -.5)
        ax.set_xlabel('Spearman ρ (95% interval)')
        ax.set_title(dec, fontsize=7)
        sns.despine(ax=ax, offset=3, left=True)
    a0 = np.atleast_1d(axes)[0]
    a0.text(.02, .02, 'Matched', color=MATCH_C, transform=a0.transAxes, fontsize=6.5)
    a0.text(.02, .09, 'Crossed', color=CROSS_C, transform=a0.transAxes, fontsize=6.5)
    letter(a0, 'a')
    out = REPO / 'notes' / 'figures' / 'brain_behavior_decoding.pdf'
    fig.savefig(out)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
