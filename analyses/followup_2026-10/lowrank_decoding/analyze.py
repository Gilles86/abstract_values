#!/usr/bin/env python3
"""Spherical vs low-rank (nested-CV k, T) decoding noise: metrics, contrasts, figure.

Input: the two tables written by collect.py (rsync them to results/ first).
All statistics are paired across participants: mean difference, 95% t-CI, number
of participants where the difference is > 0.

    python analyses/followup_2026-10/lowrank_decoding/analyze.py
"""
import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.behavior.data import get_all_behavioral_data
from abstract_values.cogmodels.brain_behavior import draws_correlation
from abstract_values.connective_fields.plot_gates import letter  # sets rcParams

HERE = Path(__file__).resolve().parent
RES = Path(__import__('os').environ.get('LOWRANK_RES', HERE / 'results'))
TRACE = Path('/data/ds-abstractvalue/derivatives/cogmodels/efficient_coding_sequential_trace.nc')
ROIS = {'orientation': ['BensonV1', 'BensonV2', 'BensonV3'],
        'value': ['BensonV1', 'NPCr', 'IPS0', 'IPS12', 'IPS345']}
ROI_LABEL = {'BensonV1': 'V1', 'BensonV2': 'V2', 'BensonV3': 'V3', 'NPCr': 'NPCr',
             'IPS0': 'IPS0', 'IPS12': 'IPS1–2', 'IPS345': 'IPS3–5'}
NOISE = ['spherical', 'diagcv', 'lowrankcv']
NOISE_LABEL = {'spherical': 'Spherical (production)', 'diagcv': 'Diagonal + T',
               'lowrankcv': 'Low-rank + T'}
COL = {'spherical': '#9C9C9C', 'diagcv': '#8172B2', 'lowrankcv': '#3B5BA5'}


def subject_number(s):
    return int(''.join(c for c in str(s) if c.isdigit()))


METRICS = ['r', 'mae', 'logp', 'cov50', 'cov80', 'cov95', 'sd_over_rmse', 'sd_err_rho',
           'slope', 'pull_extremes', 'pit_sd']


def metrics(g, quantity):
    err = g.error.values
    out = dict.fromkeys(METRICS, np.nan)
    if quantity == 'orientation':
        out['r'] = np.abs(np.exp(1j * np.deg2rad(2 * err)).mean())   # resultant length
        rms = np.sqrt(np.mean(err ** 2))
    else:
        out['r'] = np.corrcoef(g.true, g.post_mean)[0, 1]
        rms = np.sqrt(np.mean(err ** 2))
        # central pull: slope of decoded on true (1 = none) and the signed error at
        # the extremes (bottom / top 20% of each participant's values)
        out['slope'] = np.polyfit(g.true, g.post_mean, 1)[0]
        lo, hi = np.quantile(g.true, [.2, .8])
        out['pull_extremes'] = 0.5 * (err[g.true <= lo].mean() - err[g.true >= hi].mean())
        out['pit_sd'] = g.pit.std()                                   # uniform: 0.289
    out['mae'] = np.abs(err).mean()
    out['logp'] = g.logp_true.mean()
    for c in (.5, .8, .95):
        out[f'cov{int(c * 100)}'] = (g.hpd < c).mean()
    out['sd_over_rmse'] = g.post_sd.mean() / rms
    out['sd_err_rho'] = stats.spearmanr(g.post_sd, np.abs(err)).statistic
    return pd.Series(out)[METRICS]


def tci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(.975, len(x) - 1) * se
    return m, m - h, m + h, len(x), int((x > 0).sum()), stats.ttest_1samp(x, 0).pvalue


def partial_r(x, y, z):
    Z = np.c_[np.ones(len(z)), z]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def behaviour():
    b = get_all_behavioral_data()
    b = b[b.event_type == 'feedback'].copy()
    b['bid'] = pd.to_numeric(b['response'], errors='coerce')
    return b.reset_index()[['subject', 'session', 'run', 'trial_nr', 'bid']]


def brain_behaviour(tr):
    """Decoded value vs bid beyond the true value (decoded_predicts_bid.py recipe)."""
    beh = behaviour()
    d = tr[(tr.quantity == 'value') & tr.roi.isin(['BensonV1', 'NPCr'])].copy()
    d['subject_n'] = d.subject.map(subject_number)
    m = d.merge(beh, left_on=['subject_n', 'session', 'run', 'trial_nr'],
                right_on=['subject', 'session', 'run', 'trial_nr'], suffixes=('', '_b'))
    m = m.sort_values(['subject', 'roi', 'n_voxels', 'noise', 'session', 'run', 'trial_nr'])
    m['bid_next'] = m.groupby(['subject', 'roi', 'n_voxels', 'noise', 'session', 'run']).bid.shift(-1)
    rows = []
    for (roi, nv, noise, s), h in m.groupby(['roi', 'n_voxels', 'noise', 'subject']):
        k, kn = h.dropna(subset=['bid']), h.dropna(subset=['bid_next'])
        rows.append(dict(roi=roi, n_voxels=nv, noise=noise, subject=s,
                         same_trial=partial_r(k.post_mean.values, k.bid.values, k.true.values),
                         next_trial=partial_r(kn.post_mean.values, kn.bid_next.values,
                                              kn.true.values)))
    return pd.DataFrame(rows)


def noise_parameters(per):
    """Decoding fidelity vs cogmodel noise (brain_behavior_decoding.py recipe)."""
    post = az.from_netcdf(TRACE).posterior
    subs = [int(x) for x in post.kappa_r.coords['subject'].values]
    pars = {'κ_r': post.kappa_r.values.reshape(-1, len(subs)),
            '1/σ_rep': 1 / post.sigma_rep.values.reshape(-1, len(subs))}
    cells = {'V1 orientation': ('orientation', 'BensonV1'), 'NPCr value': ('value', 'NPCr')}
    rows = []
    for (nv, noise), g in per.groupby(['n_voxels', 'noise']):
        for neural, (q, roi) in cells.items():
            a = g[(g.quantity == q) & (g.roi == roi)].copy()
            a.index = a.subject.map(subject_number)
            a = a.reindex(subs)
            for par, draws in pars.items():
                for meas, y in (('r', a.r.values), ('−MAE', -a.mae.values)):
                    keep = np.isfinite(y)
                    rho, lo, hi = draws_correlation(draws[:, keep], y[keep])
                    rows.append(dict(n_voxels=nv, noise=noise, neural=neural, parameter=par,
                                     measure=meas, matched=(par == 'κ_r') == (q == 'orientation'),
                                     rho=rho, lo=lo, hi=hi, n=int(keep.sum())))
    return pd.DataFrame(rows)


def figure(per, con, bb, tr, nv, out):
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.4), constrained_layout=True)
    cells = [(q, r) for q in ROIS for r in ROIS[q]]
    xt = [f'{ROI_LABEL[r]} {"ori" if q == "orientation" else "val"}' for q, r in cells]
    x = np.arange(len(cells))
    p = per[per.n_voxels == nv]
    off = {'spherical': -.22, 'diagcv': 0, 'lowrankcv': .22}

    def dots(ax, col, ref=None):
        for n in NOISE:
            for i, (q, r) in enumerate(cells):
                v = p[(p.quantity == q) & (p.roi == r) & (p.noise == n)][col].dropna()
                if not len(v):
                    continue
                m, lo, hi, *_ = tci(v)
                ax.plot([i + off[n]] * 2, [lo, hi], color=COL[n], lw=1)
                ax.plot(i + off[n], m, 'o', ms=3.4, color=COL[n])
        if ref is not None:
            ax.axhline(ref, color='.7', lw=.6, ls='--', zorder=0)
        ax.set_xticks(x, xt, fontsize=5.8, rotation=45, ha='right', rotation_mode='anchor')
        sns.despine(ax=ax, offset=3)

    ax = axes[0, 0]
    dots(ax, 'cov80', .8)
    ax.set_ylabel('80% interval coverage')
    ax.set_ylim(0, 1)
    ax.text(len(cells) - .5, .82, 'Nominal', color='.5', fontsize=6, ha='right', va='bottom')
    for i, n in enumerate(NOISE):
        ax.text(.02, .30 - i * .1, NOISE_LABEL[n], color=COL[n], fontsize=6,
                transform=ax.transAxes)
    letter(ax, 'a')

    ax = axes[0, 1]
    dots(ax, 'logp', 0)
    ax.set_ylabel('Log score vs flat (nats/trial)')
    lp = p.logp
    ax.set_ylim(max(lp.quantile(.02) - .2, -6), lp.max() + .1)
    letter(ax, 'b')

    ax = axes[0, 2]
    dots(ax, 'sd_over_rmse', 1)
    ax.set_ylabel('Posterior SD / RMS error')
    letter(ax, 'c')

    ax = axes[1, 0]
    c = con[(con.n_voxels == nv) & (con.metric == 'r') & (con.contrast == 'lowrankcv − spherical')]
    for i, (q, r) in enumerate(cells):
        row = c[(c.quantity == q) & (c.roi == r)]
        if len(row):
            row = row.iloc[0]
            ax.plot([i, i], [row.lo, row.hi], color=COL['lowrankcv'], lw=1)
            ax.plot(i, row['mean'], 'o', ms=3.4, color=COL['lowrankcv'])
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks(x, xt, fontsize=5.8, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_ylabel('Δ accuracy r (low-rank − spherical)')
    sns.despine(ax=ax, offset=3)
    letter(ax, 'd')

    ax = axes[1, 1]
    d = tr[(tr.quantity == 'value') & (tr.roi == 'NPCr') & (tr.n_voxels == nv)].copy()
    d['bin'] = pd.cut(d.true, np.linspace(d.true.min() - 1e-6, d.true.max() + 1e-6, 9))
    for n in ('spherical', 'lowrankcv'):
        gm = (d[d.noise == n].groupby(['subject', 'bin'], observed=True)
              .agg(true=('true', 'mean'), dec=('post_mean', 'mean')).reset_index()
              .groupby('bin', observed=True).agg(true=('true', 'mean'), m=('dec', 'mean'),
                                                  se=('dec', 'sem')))
        ax.fill_between(gm.true, gm.m - gm.se, gm.m + gm.se, color=COL[n], alpha=.25, lw=0)
        ax.plot(gm.true, gm.m, color=COL[n], lw=1.3)
    lim = [d.true.min(), d.true.max()]
    ax.plot(lim, lim, color='.7', lw=.6, ls='--', zorder=0)
    ax.text(lim[1], lim[1] + 1, 'Veridical', color='.5', fontsize=6, ha='right', va='bottom')
    ax.set_xlabel('True value (CHF)')
    ax.set_ylabel('Decoded value, NPCr (CHF)')
    sns.despine(ax=ax, offset=3, trim=True)
    letter(ax, 'e')

    ax = axes[1, 2]
    b = bb[bb.n_voxels == nv]
    for j, roi in enumerate(['BensonV1', 'NPCr']):
        for k, n in enumerate(('spherical', 'lowrankcv')):
            v = b[(b.roi == roi) & (b.noise == n)].same_trial
            m, lo, hi, *_ = tci(v)
            xx = j + (k - .5) * .3
            ax.scatter(xx + np.random.default_rng(j).uniform(-.05, .05, len(v)), v, s=4,
                       color=COL[n], alpha=.35, lw=0)
            ax.plot([xx, xx], [lo, hi], color='.15', lw=.9, zorder=3)
            ax.plot(xx, m, 'D', ms=4, mfc=COL[n], mec='.15', mew=.8, zorder=4)
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks([0, 1], ['V1 value', 'NPCr value'])
    ax.set_ylabel('Partial r (decoded vs bid,\ntrue value removed)')
    sns.despine(ax=ax, offset=3)
    letter(ax, 'f')
    for ax in (*axes[0], axes[1, 0]):
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    fig.savefig(out)
    print(f'saved {out}')


def production_check(tr, deriv=Path('/data/ds-abstractvalue/derivatives/decoding')):
    """Our 'spherical' arm against the existing production decodes (nv 50, lambda 0.1)."""
    rows = []
    for q, sub_dir, roi in (('orientation', 'gabor', 'BensonV1'), ('value', 'value', 'BensonV1'),
                            ('value', 'value', 'NPCr')):
        mine = tr[(tr.quantity == q) & (tr.roi == roi) & (tr.n_voxels == 50) &
                  (tr.noise == 'spherical')]
        for s, g in mine.groupby('subject'):
            fn = (deriv / sub_dir / f'sub-{s}' / 'func' /
                  f'sub-{s}_mask-{roi}_nvoxels-50_noise-spherical_lambda-0.1_pars.tsv')
            if not fn.exists():
                continue
            p = pd.read_csv(fn, sep='\t', index_col=[0, 1, 2, 3])
            grid = p.columns.astype(float).values
            P = p.values / p.values.sum(1, keepdims=True)
            if q == 'orientation':
                pm = np.rad2deg((np.angle(P @ np.exp(2j * grid)) / 2) % np.pi)
            else:
                pm = P @ grid
            ref = pd.Series(pm, index=p.index.droplevel(3))
            m = g.set_index(['session', 'run', 'trial_nr']).post_mean
            ref = ref.reindex(m.index)
            d = m - ref
            if q == 'orientation':
                d = (d + 90) % 180 - 90
            rows.append(dict(quantity=q, roi=roi, subject=s, n=int(ref.notna().sum()),
                             max_abs_diff=float(np.nanmax(np.abs(d))),
                             median_abs_diff=float(np.nanmedian(np.abs(d)))))
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(RES / 'lowrank_production_check.tsv', sep='\t', index=False, float_format='%.4g')
        print('production check (decoded mean, ours vs production file):')
        print(out.groupby(['quantity', 'roi'])[['max_abs_diff', 'median_abs_diff']]
              .agg(['median', 'max']).to_string())
    return out


def main(trials_fn, meta_fn, nv_fig):
    tr = pd.read_csv(trials_fn, sep='\t', dtype={'subject': str})
    meta = pd.read_csv(meta_fn, sep='\t', dtype={'subject': str})
    production_check(tr)

    per = (tr.groupby(['quantity', 'roi', 'n_voxels', 'noise', 'subject'])
           .apply(lambda g: metrics(g, g.name[0])).reset_index())
    km = (meta.groupby(['quantity', 'roi', 'n_voxels', 'subject'])
          .agg(k=('k', 'median'), T=('temperature', 'median'), T_diag=('temperature_diag', 'median'),
               dof=('dof_lowrank', 'median'), cv_best=('cv_best', 'mean'),
               cv_k0=('cv_k0_best', 'mean')).reset_index())
    per.to_csv(RES / 'lowrank_metrics_by_subject.tsv', sep='\t', index=False, float_format='%.5g')

    rows = []
    mcols = [c for c in per.columns if c not in ('quantity', 'roi', 'n_voxels', 'noise', 'subject')]
    w = per.pivot_table(index=['quantity', 'roi', 'n_voxels', 'subject'], columns='noise',
                        values=mcols)
    for (q, r, nv), g in w.groupby(level=[0, 1, 2]):
        for metric in mcols:
            for a, b_ in (('lowrankcv', 'spherical'), ('diagcv', 'spherical'), ('lowrankcv', 'diagcv')):
                if (metric, a) not in g or (metric, b_) not in g:
                    continue
                dd = (g[(metric, a)] - g[(metric, b_)]).values
                if np.isfinite(dd).sum() < 3:
                    continue
                m, lo, hi, n, npos, pv = tci(dd)
                rows.append(dict(quantity=q, roi=r, n_voxels=nv, metric=metric,
                                 contrast=f'{a} − {b_}', mean=m, lo=lo, hi=hi, n=n,
                                 n_positive=npos, p=pv,
                                 spherical=np.nanmean(g[(metric, 'spherical')]),
                                 other=np.nanmean(g[(metric, a)])))
    con = pd.DataFrame(rows)
    con.to_csv(RES / 'lowrank_contrasts.tsv', sep='\t', index=False, float_format='%.4g')

    # group table: mean [95% CI] per cell, the three noise models side by side
    grp = []
    for (q, r, nv, n), g in per.groupby(['quantity', 'roi', 'n_voxels', 'noise']):
        row = dict(quantity=q, roi=r, n_voxels=nv, noise=n, n=len(g))
        for mcol in mcols:
            if g[mcol].notna().sum() > 2:
                m, lo, hi, *_ = tci(g[mcol])
                row[mcol] = f'{m:.3f} [{lo:.3f}, {hi:.3f}]'
        grp.append(row)
    grp = pd.DataFrame(grp)
    grp.to_csv(RES / 'lowrank_group_metrics.tsv', sep='\t', index=False)

    kt = []
    for (q, r, nv), g in km.groupby(['quantity', 'roi', 'n_voxels']):
        kt.append(dict(quantity=q, roi=r, n_voxels=nv, n=len(g),
                       k_median=g.k.median(), k_iqr=f'{g.k.quantile(.25):g}–{g.k.quantile(.75):g}',
                       T_median=g['T'].median(),
                       T_iqr=f'{g["T"].quantile(.25):.2f}–{g["T"].quantile(.75):.2f}',
                       T_diag_median=g.T_diag.median(), dof_median=g.dof.median(),
                       cv_gain_lowrank_over_k0=(g.cv_best - g.cv_k0).mean(),
                       frac_folds_T_inf=(meta[(meta.quantity == q) & (meta.roi == r) &
                                              (meta.n_voxels == nv)].temperature == np.inf).mean()))
    kt = pd.DataFrame(kt)
    kt.to_csv(RES / 'lowrank_k_T.tsv', sep='\t', index=False, float_format='%.4g')
    print(kt.to_string(index=False))

    bb = brain_behaviour(tr)
    bb.to_csv(RES / 'lowrank_decoded_predicts_bid_by_subject.tsv', sep='\t', index=False,
              float_format='%.5g')
    bbs = []
    for (roi, nv, noise), g in bb.groupby(['roi', 'n_voxels', 'noise']):
        m, lo, hi, n, npos, pv = tci(g.same_trial)
        mn, lon, hin, *_, pvn = tci(g.next_trial)
        bbs.append(dict(roi=roi, n_voxels=nv, noise=noise, n=n, same=m, same_lo=lo, same_hi=hi,
                        same_p=pv, next=mn, next_lo=lon, next_hi=hin, next_p=pvn))
    w2 = bb.pivot_table(index=['roi', 'n_voxels', 'subject'], columns='noise', values='same_trial')
    for (roi, nv), g in w2.groupby(level=[0, 1]):
        m, lo, hi, n, npos, pv = tci(g.lowrankcv - g.spherical)
        bbs.append(dict(roi=roi, n_voxels=nv, noise='lowrankcv − spherical', n=n, same=m,
                        same_lo=lo, same_hi=hi, same_p=pv))
    bbs = pd.DataFrame(bbs)
    bbs.to_csv(RES / 'lowrank_decoded_predicts_bid.tsv', sep='\t', index=False, float_format='%.4g')
    print(bbs.to_string(index=False))

    if TRACE.exists():
        npar = noise_parameters(per)
        npar.to_csv(RES / 'lowrank_noise_parameter_correlations.tsv', sep='\t', index=False,
                    float_format='%.3f')
        print(npar[npar.matched].to_string(index=False))

    pd.set_option('display.width', 200)
    show = con[(con.contrast == 'lowrankcv − spherical') &
               con.metric.isin(['r', 'mae', 'logp', 'cov80', 'sd_over_rmse', 'slope'])]
    print(show.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    figure(per, con, bb, tr, nv_fig, RES / 'fig_lowrank_decoding.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--trials', default=str(RES / 'lowrank_trials.tsv.gz'))
    p.add_argument('--meta', default=str(RES / 'lowrank_meta.tsv.gz'))
    p.add_argument('--nv-figure', type=int, default=50)
    a = p.parse_args()
    main(a.trials, a.meta, a.nv_figure)
