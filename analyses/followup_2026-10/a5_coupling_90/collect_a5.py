#!/usr/bin/env python3
"""A5 group level (run on the cluster): collect per-subject outputs of a5_coupling_90.py.

Per (n_bins, v1_sel, condition, segment, ips_sel, metric, contrast), across subjects:
  delta_obs_minus_perm  observed delta minus the subject's permutation-null mean
                        (removes the bin-size advantage expected by sampling alone)
  delta_sub             delta with every bin subsampled to the smallest bin's count
  -> mean, bootstrap 95% CI (10,000 resamples of subjects), one-sample t, p, dz.
  group permutation p: mean-over-subjects observed delta vs the distribution of the
  mean-over-subjects permutation delta (permutation i of every subject). It inherits
  the anti-conservativeness of voxel-label permutation under spatial autocorrelation
  (memory: connective-field pitfalls), so the bootstrap / t on the subject-level
  differences is the inference of record.
Also the mapping difference (Aligned = inverse_cdf minus Misaligned = cdf) per subject.

Writes derivatives/followup_2026-10/a5_coupling_90/group/:
  subject_contrasts.tsv, subject_bins.tsv, subject_counts.tsv, subject_v1pref.tsv,
  group_contrasts.tsv, group_mapping_difference.tsv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

KEYS = ['n_bins', 'v1_sel', 'condition', 'segment', 'ips_sel', 'metric', 'contrast']


def summarise(x, rng, n_boot=10000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return dict(n=len(x), mean=np.nan, ci_lo=np.nan, ci_hi=np.nan, t=np.nan, p=np.nan, dz=np.nan)
    boot = rng.choice(x, (n_boot, len(x))).mean(1)
    t, p = stats.ttest_1samp(x, 0)
    return dict(n=len(x), mean=x.mean(), ci_lo=np.percentile(boot, 2.5),
                ci_hi=np.percentile(boot, 97.5), t=t, p=p, dz=x.mean() / x.std(ddof=1))


def main(root):
    root = Path(root)
    out = root / 'group'
    out.mkdir(exist_ok=True)
    rd = dict(sep='\t', dtype={'subject': str})
    subs = sorted(p for p in root.glob('sub-*') if (p / f'{p.name}_desc-contrasts.tsv').exists())
    print(f'{len(subs)} subjects')
    con = pd.concat([pd.read_csv(p / f'{p.name}_desc-contrasts.tsv', **rd) for p in subs])
    for name in ('bins', 'counts', 'v1pref'):
        pd.concat([pd.read_csv(p / f'{p.name}_desc-{name}.tsv', **rd) for p in subs]).to_csv(
            out / f'subject_{name}.tsv', sep='\t', index=False)
    con.to_csv(out / 'subject_contrasts.tsv', sep='\t', index=False)
    null = pd.concat([pd.read_csv(p / f'{p.name}_desc-permnull.tsv.gz', **rd) for p in subs])
    gnull = null.groupby(KEYS + ['perm'])['delta'].mean().rename('null_mean').reset_index()

    rng = np.random.default_rng(1)
    rows = []
    for key, g in con.groupby(KEYS):
        nl = gnull.merge(pd.DataFrame([dict(zip(KEYS, key))]), on=KEYS)['null_mean'].to_numpy()
        obs = g['delta_obs'].mean()
        for est in ('delta_obs_minus_perm', 'delta_sub'):
            rows.append({**dict(zip(KEYS, key)), 'estimator': est, **summarise(g[est], rng),
                         'p_groupperm': (1 + (nl >= obs).sum()) / (1 + len(nl))
                         if est == 'delta_obs_minus_perm' else np.nan,
                         'n_subjects_sub': int((g['n_sub'] > 0).sum())})
    grp = pd.DataFrame(rows)
    grp.to_csv(out / 'group_contrasts.tsv', sep='\t', index=False)

    k2 = [k for k in KEYS if k != 'condition']
    rows = []
    for key, g in con.groupby(k2):
        w = g.pivot_table(index='subject', columns='condition',
                          values=['delta_obs_minus_perm', 'delta_sub'])
        for est in ('delta_obs_minus_perm', 'delta_sub'):
            d = w[est]['inverse_cdf'] - w[est]['cdf']
            both = ((w[est]['inverse_cdf'] > 0) & (w[est]['cdf'] > 0)).mean()
            rows.append({**dict(zip(k2, key)), 'estimator': est,
                         'difference': 'aligned(inverse_cdf) - misaligned(cdf)',
                         **summarise(d, rng), 'frac_subjects_both_positive': both})
    pd.DataFrame(rows).to_csv(out / 'group_mapping_difference.tsv', sep='\t', index=False)

    show = grp.query("ips_sel == 'all' and contrast == 'delta90' and estimator == 'delta_obs_minus_perm'")
    print(show[['n_bins', 'v1_sel', 'condition', 'segment', 'metric', 'n', 'mean', 'ci_lo',
                'ci_hi', 'p', 'p_groupperm']].round(4).to_string())
    print(f'saved to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/'
                                     'followup_2026-10/a5_coupling_90')
    main(p.parse_args().root)
