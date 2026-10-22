"""Model comparison on held-out cvR² in model-neutrally selected voxels.

Selection: ``encoding_models/permfdr_select_models.py`` (gated permutation FDR
on T = max over models of -log10 p, per ROI). Held-out cvR² per voxel and
model: ``collect_permnull.py`` table. Per participant × ROI, the median cvR²
of each model over the selected voxels (≥ MIN_VOX); contrasts are differences
of those medians, tested across participants (t-test, count of Δ > 0, Holm
over the contrasts within ROI × smoothing).

Also:
  * the same in voxels where no model beats the null on cvR² (ranking can flip
    in noise voxels);
  * the selection-bias check: Δ in each null copy's top 5% of voxels by T
    (weighted by ``n_top5_null``) vs Δ over all voxels — should be ≈ equal.

Writes ``notes/data/selected_models_per_subject.tsv`` (same layout as
``quantile_models_per_subject.tsv``, so ``figure_model_comparison.py --tsv``
can draw it) and ``selected_models_contrasts.tsv``.

Usage:
    python -m abstract_values.visualize.compare_selected_models
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
MODELS = ['gaussian', 'quantile', 'quantile_wrapped', 'gauss_session_shift',
          'linear']
CONTRASTS = [('quantile', 'gaussian', 'Quantile − fixed value'),
             ('quantile_wrapped', 'quantile', 'Wrapped − unwrapped'),
             ('gauss_session_shift', 'gaussian', 'Flexible − fixed value'),
             ('linear', 'gaussian', 'Linear − fixed value')]
MIN_VOX = 5


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o])
    out = np.empty_like(p)
    out[o] = np.minimum(adj, 1)
    return out


def per_subject(df, rois):
    rows = []
    for (s, sm), d in df.groupby(['subject', 'smoothed']):
        for roi in rois:
            if f'sel_{roi}' not in d:
                continue
            r = d[d[f'in_{roi}'].astype(bool)]
            best = r[[f'cvr2_{m}' for m in MODELS]].max(1)
            sets = {'Selected': r[f'sel_{roi}'].astype(bool),
                    'No model beats null': best <= r.cvr2_null,
                    'All voxels': pd.Series(True, index=r.index)}
            for name, m in sets.items():
                x = r[m]
                if len(x) < MIN_VOX:
                    continue
                row = dict(subject=s, smoothed=sm, roi=roi, selection=name,
                           n=len(x), frac=float(m.mean()))
                for c in MODELS + ['null']:
                    row[c] = 100 * float(x[f'cvr2_{c}'].median())
                for c in MODELS:                     # gain over null, for the ladder
                    row[c] -= row['null']
                rows.append(row)
            # bias check: null top-5% voxels (weighted) vs all voxels
            w = r.n_top5_null.to_numpy(float)
            if w.sum() > 0:
                row = dict(subject=s, smoothed=sm, roi=roi,
                           selection='Null top 5%', n=int((w > 0).sum()),
                           frac=float((w > 0).mean()))
                for c in MODELS + ['null']:
                    row[c] = 100 * float(np.average(r[f'cvr2_{c}'], weights=w))
                for c in MODELS:
                    row[c] -= row['null']
                rows.append(row)
    return pd.DataFrame(rows)


def contrasts(ps):
    rows = []
    for (sel, sm, roi), d in ps.groupby(['selection', 'smoothed', 'roi']):
        rr = []
        for a, b, lab in CONTRASTS:
            x = (d[a] - d[b]).dropna()
            if len(x) < 3:
                continue
            t = stats.ttest_1samp(x, 0)
            rr.append(dict(selection=sel, smoothed=sm, roi=roi, contrast=lab,
                           n=len(x), mean=x.mean(), sem=x.sem(),
                           n_pos=int((x > 0).sum()), t=t.statistic, p=t.pvalue))
        if rr:
            ph = holm([r['p'] for r in rr])
            for r, h in zip(rr, ph):
                r['p_holm'] = h
            rows += rr
    return pd.DataFrame(rows)


def main(sel_fn, cv_fn, tag):
    sel = pd.read_csv(sel_fn, sep='\t', dtype={'subject': str})
    cv = pd.read_csv(cv_fn, sep='\t', dtype={'subject': str})
    rois = [c.removeprefix('sel_') for c in sel if c.startswith('sel_')]
    keep = (['subject', 'smoothed', 'flat_index'] + [f'in_{r}' for r in rois]
            + [f'cvr2_{m}' for m in MODELS + ['null']])
    df = sel.merge(cv[keep], on=['subject', 'smoothed', 'flat_index'])
    ps = per_subject(df, rois)
    ps.to_csv(REPO / 'notes' / 'data' / f'selected_models_per_subject{tag}.tsv',
              sep='\t', index=False)
    ct = contrasts(ps)
    ct.to_csv(REPO / 'notes' / 'data' / f'selected_models_contrasts{tag}.tsv',
              sep='\t', index=False)
    with pd.option_context('display.width', 220, 'display.max_rows', 500):
        cnt = (ps[ps.selection == 'Selected'].groupby(['smoothed', 'roi'])
               .agg(n_subjects=('n', 'size'), median_vox=('n', 'median')))
        print(cnt.unstack(0).to_string())
        print(ct.round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selection', default=str(REPO / 'notes/data/permfdr_models_voxels.tsv.gz'))
    ap.add_argument('--cv', default=str(REPO / 'notes/data/permnull_voxels.tsv.gz'))
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    main(a.selection, a.cv, a.tag)
