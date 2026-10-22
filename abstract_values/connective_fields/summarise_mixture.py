"""Group summary of the two-component coupling fit (test_coupling --template mixture).

Per variant directory (target ROI × lags) and participant:
  b: value-wired weight (template moves with the mapping), observed − mean of
     the label shuffles
  a: orientation-wired weight (session-invariant quantile template), same
  per-voxel: Spearman ρ between b (and a) and the voxel's tuning preference
     cvR²(quantile) − cvR²(fixed value)
Group: one-sample t-tests across participants (ρ via Fisher z).

Input: derivatives/connective_fields/coupling*_tpl-mixture/sub-*/ (rsync them
to the same relative path under --root). Output: notes/data/cf_mixture_summary.tsv
and cf_mixture_subjects.tsv.

Usage:
    python -m abstract_values.connective_fields.summarise_mixture --root /path/to/connective_fields
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]


def per_subject(d):
    rows = []
    for sub in sorted(d.glob('sub-*')):
        s = sub.name.removeprefix('sub-')
        sc = sub / f'sub-{s}_desc-scores.tsv'
        vx = sub / f'sub-{s}_desc-voxels.tsv'
        if not (sc.exists() and vx.exists()):
            continue
        sc, vx = pd.read_csv(sc, sep='\t'), pd.read_csv(vx, sep='\t')
        obs, sh = sc[sc.kind == 'observed'].iloc[0], sc[sc.kind == 'label-shuffle']
        ok = vx[['b_value', 'a_orientation', 'delta_quantile_fixed']].notna().all(1)
        v = vx[ok]
        rb = stats.spearmanr(v.b_value, v.delta_quantile_fixed)[0] if len(v) > 10 else np.nan
        ra = (stats.spearmanr(v.a_orientation, v.delta_quantile_fixed)[0]
              if len(v) > 10 else np.nan)
        rows.append(dict(subject=s, n_voxels=len(vx),
                         b=obs.score - sh.score.mean(),
                         a=obs.score_orientation - sh.score_orientation.mean(),
                         b_obs=obs.score, a_obs=obs.score_orientation,
                         rho_b_pref=rb, rho_a_pref=ra))
    return pd.DataFrame(rows)


def group(ps):
    out = {}
    for c in ('b', 'a'):
        x = ps[c].dropna()
        t = stats.ttest_1samp(x, 0)
        out[c] = x.mean()
        out[f'{c}_sem'] = x.sem()
        out[f'{c}_t'], out[f'{c}_p'] = t.statistic, t.pvalue
    for c in ('rho_b_pref', 'rho_a_pref'):
        z = np.arctanh(ps[c].dropna().clip(-.999, .999))
        t = stats.ttest_1samp(z, 0)
        out[c] = float(np.tanh(z.mean()))
        out[f'{c}_t'], out[f'{c}_p'] = t.statistic, t.pvalue
    out['n'] = len(ps)
    return out


def main(root):
    rows, subs = [], []
    for d in sorted(Path(root).glob('coupling*_tpl-mixture')):
        ps = per_subject(d)
        if ps.empty:
            continue
        name = d.name
        roi = name.split('_target-')[1].split('_')[0] if '_target-' in name else 'NPCr'
        lags = int(name.split('lags-')[1].split('_')[0]) if 'lags-' in name else 0
        ctrl = ('gain' if '_gain' in name else '') + (
            name.split('_drive-')[1].split('_')[0] if '_drive-' in name else '')
        subs.append(ps.assign(roi=roi, lags=lags, control=ctrl or 'none'))
        rows.append(dict(roi=roi, lags=lags, control=ctrl or 'none', **group(ps)))
    res = pd.DataFrame(rows).sort_values(['control', 'lags', 'roi'])
    res.to_csv(REPO / 'notes/data/cf_mixture_summary.tsv', sep='\t', index=False)
    pd.concat(subs).to_csv(REPO / 'notes/data/cf_mixture_subjects.tsv', sep='\t',
                           index=False)
    with pd.option_context('display.width', 220):
        print(res.round(4).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    main(ap.parse_args().root)
