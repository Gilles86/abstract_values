#!/usr/bin/env python3
"""Low-rank decoding with CV-selected voxel count vs fixed N = 50 / 250.

Inputs (rsync from the cluster, collect.py --tree ...):
  results/lowrank_trials.tsv.gz, results/lowrank_meta.tsv.gz            fixed N
  results/nvcv/lowrank_trials.tsv.gz, results/nvcv/lowrank_meta.tsv.gz  CV-selected N

Writes results/nvcv_contrasts.tsv (paired, 95% t-CI), results/nvcv_chosen_n.tsv
(chosen N per participant x ROI: median over folds, modal-N share, IQR) and
results/nvcv_cv_curve.tsv (mean inner-CV score per candidate N).

    python analyses/followup_2026-10/lowrank_decoding/compare_nvcv.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import metrics, tci  # same metric definitions as analyze.py

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'


def main():
    fx = pd.read_csv(RES / 'lowrank_trials.tsv.gz', sep='\t', dtype={'subject': str, 'n_voxels': str})
    cv = pd.read_csv(RES / 'nvcv' / 'lowrank_trials.tsv.gz', sep='\t',
                     dtype={'subject': str, 'n_voxels': str})
    meta = pd.read_csv(RES / 'nvcv' / 'lowrank_meta.tsv.gz', sep='\t', dtype={'subject': str})
    fx = fx[fx.noise == 'lowrankcv']
    tr = pd.concat([fx, cv], ignore_index=True)
    per = (tr.groupby(['quantity', 'roi', 'n_voxels', 'subject'])
           .apply(lambda g: metrics(g, g.name[0])).reset_index())
    per.to_csv(RES / 'nvcv_metrics_by_subject.tsv', sep='\t', index=False, float_format='%.5g')

    rows = []
    mcols = ['r', 'mae', 'logp', 'cov50', 'cov80', 'cov95', 'sd_over_rmse', 'slope']
    w = per.pivot_table(index=['quantity', 'roi', 'subject'], columns='n_voxels', values=mcols)
    for (q, r), g in w.groupby(level=[0, 1]):
        for metric in mcols:
            for ref in ('50', '250'):
                if (metric, ref) not in g or (metric, 'cv') not in g:
                    continue
                d = (g[(metric, 'cv')] - g[(metric, ref)]).values
                if np.isfinite(d).sum() < 3:
                    continue
                m, lo, hi, n, npos, p = tci(d)
                rows.append(dict(quantity=q, roi=r, metric=metric, contrast=f'cv − {ref}',
                                 mean=m, lo=lo, hi=hi, n=n, n_positive=npos, p=p,
                                 fixed=np.nanmean(g[(metric, ref)]),
                                 cvsel=np.nanmean(g[(metric, 'cv')])))
    con = pd.DataFrame(rows)
    con.to_csv(RES / 'nvcv_contrasts.tsv', sep='\t', index=False, float_format='%.4g')

    def modal_share(x):
        return x.value_counts(normalize=True).iloc[0]

    ch = (meta.groupby(['quantity', 'roi', 'subject'])
          .agg(n_roi=('n_roi', 'first'), N_median=('n_selected', 'median'),
               N_q25=('n_selected', lambda x: x.quantile(.25)),
               N_q75=('n_selected', lambda x: x.quantile(.75)),
               modal_share=('n_selected', modal_share), k_median=('k', 'median'),
               T_median=('temperature', 'median')).reset_index())
    ch['frac_of_roi'] = ch.N_median / ch.n_roi
    ch.to_csv(RES / 'nvcv_chosen_n_by_subject.tsv', sep='\t', index=False, float_format='%.4g')
    grp = []
    for (q, r), g in ch.groupby(['quantity', 'roi']):
        grp.append(dict(quantity=q, roi=r, n=len(g), roi_voxels_median=g.n_roi.median(),
                        N_median=g.N_median.median(),
                        N_iqr=f'{g.N_median.quantile(.25):g}–{g.N_median.quantile(.75):g}',
                        frac_of_roi_median=g.frac_of_roi.median(),
                        fold_modal_share=g.modal_share.mean(), k_median=g.k_median.median(),
                        T_median=g.T_median.median(),
                        frac_folds_all_voxels=(meta[(meta.quantity == q) & (meta.roi == r)]
                                               .eval('n_selected == n_roi')).mean()))
    grp = pd.DataFrame(grp)
    grp.to_csv(RES / 'nvcv_chosen_n.tsv', sep='\t', index=False, float_format='%.4g')

    ncols = [c for c in meta.columns if c.startswith('cv_N')]
    curve = (meta.melt(id_vars=['quantity', 'roi', 'subject', 'n_roi'], value_vars=ncols,
                       var_name='N', value_name='cv').dropna())
    curve['N'] = curve.N.str[4:].astype(int)
    # "all voxels" is its own column per participant; label it -1
    curve.loc[curve.N == curve.n_roi, 'N'] = -1
    cs = (curve.groupby(['quantity', 'roi', 'subject', 'N']).cv.mean().reset_index()
          .groupby(['quantity', 'roi', 'N']).cv.agg(['mean', 'sem', 'count']).reset_index())
    cs.to_csv(RES / 'nvcv_cv_curve.tsv', sep='\t', index=False, float_format='%.4g')

    pd.set_option('display.width', 220)
    print(grp.to_string(index=False))
    print(con[con.metric.isin(['r', 'mae', 'logp', 'cov80'])].to_string(
        index=False, float_format=lambda v: f'{v:.3f}'))
    print(cs.to_string(index=False, float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    main()
