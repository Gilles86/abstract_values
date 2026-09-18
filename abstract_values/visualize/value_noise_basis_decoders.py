#!/usr/bin/env python3
"""Value-decoding noise vs value density, with MATCHED model families.

The orientation decoder is a von Mises basis set (8 weights per voxel) while
the default value decoder is a single log-Gaussian bump, so the two sides of
the density analysis were not matched. ``decode_value.py --model weighted`` is
the basis-set analogue in value space; this compares both families in NPCr and
in V1 (the orientation-warp control), all at 100 voxels, spherical noise,
lambda 0.1, leave-one-run-out.

Per ROI x model family, a mixed model over all trials:

    log(|residual| + 0.5) ~ log density + condition [+ orientation]

residual = posterior mean minus the mean for that subject x condition x
stimulus. With orientation fixed effects the slope comes only from the two
conditions putting a different density on the SAME orientation.

Negative slope = less noise where values are dense (what efficient coding
predicts, and what an orientation code produces mechanically in value units).

    python -m abstract_values.visualize.value_noise_basis_decoders
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings

from abstract_values.connective_fields.gates import load_mappings
from abstract_values.utils.data import Subject

warnings.filterwarnings('ignore')
REPO = Path(__file__).resolve().parents[2]
G = np.linspace(2, 42, 801)


def density(values, bw=2.):
    k = np.exp(-.5 * ((G[:, None] - values[None, :]) / bw) ** 2).sum(1)
    return k / (k.sum() * (G[1] - G[0]))


def main():
    ori, maps = load_mappings()
    den = {c: density(maps[c][(ori > 0) & (ori < 180)]) for c in maps}
    d = pd.read_csv(REPO / 'notes' / 'data' / 'decoding_trials_weighted.tsv', sep='\t',
                    dtype={'subject': str})
    cond = {k: Subject(k[0]).get_mapping(int(k[1])) for k in set(zip(d.subject, d.session))}
    d['condition'] = [cond[k] for k in zip(d.subject, d.session)]
    d['orientation'] = [round(float(np.interp(t, maps[c], ori)), 1)
                        for t, c in zip(d.true, d.condition)]
    d['logdens'] = [np.log(np.interp(t, G, den[c])) for t, c in zip(d.true, d.condition)]
    grp = d.groupby(['model', 'roi', 'subject', 'condition', 'true']).post_mean
    n = grp.transform('size')
    d['y'] = np.log(((d.post_mean - grp.transform('mean')) * np.sqrt(n / (n - 1))).abs() + .5)
    d['ld'] = d.logdens - d.logdens.mean()

    rows = []
    for (mod, roi), h in d.groupby(['model', 'roi']):
        r = h.groupby('subject').apply(lambda x: np.corrcoef(x.true, x.post_mean)[0, 1],
                                       include_groups=False)
        slope = h.groupby(['subject', 'condition']).apply(
            lambda x: np.polyfit(x.true, x.post_mean, 1)[0], include_groups=False)
        for test, formula in (('across values', 'y ~ ld + C(condition)'),
                              ('same orientation', 'y ~ ld + C(condition) + C(orientation)')):
            f = smf.mixedlm(formula, h, groups=h.subject, re_formula='~ld').fit()
            rows.append(dict(model=mod, roi=roi, test=test, slope=f.params['ld'],
                             se=f.bse['ld'], p=f.pvalues['ld'], decoding_r=r.mean(),
                             compression=slope.mean()))
    for mod, h in d.groupby('model'):
        f = smf.mixedlm('y ~ ld * C(roi) + C(condition) * C(roi) + C(orientation) * C(roi)',
                        h, groups=h.subject, re_formula='~ld').fit()
        k = [x for x in f.params.index if x.startswith('ld:')][0]
        rows.append(dict(model=mod, roi='NPCr - V1', test='same orientation',
                         slope=f.params[k], se=f.bse[k], p=f.pvalues[k]))
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda x: f'{x:.3g}'))
    res.to_csv(REPO / 'notes' / 'data' / 'value_noise_basis_decoders.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
