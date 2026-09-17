#!/usr/bin/env python3
"""Horizontal-vs-vertical asymmetry: Fourier-prior fits against V1 precision.

The Fourier-prior efficient-coding fits describe each subject's orientation
prior as p(phi) ~ exp(sum_k a_k cos(k phi) + b_k sin(k phi)) on the doubled
angle phi = 2 theta:

    a1  cos 2theta  horizontal (0 deg) vs vertical (90 deg)
    b1  sin 2theta  45 deg vs 135 deg
    a2  cos 4theta  cardinal vs oblique              (fourier2 fits only)

Efficient coding makes precision track the prior (Fisher information ~ p^2),
so log(decoded SD) should carry -a_k in each harmonic. The neural analogue is
the same decomposition of V1's expected decoded orientation SD (von Mises
encoding model, simulated decoding; notes/data/expected_uncertainty_orientation_v1.tsv):

    log sd_E(theta) = c0 + c1 cos 2theta + s1 sin 2theta + c2 cos 4theta + s2 sin 4theta

Prediction, per subject: a1 ~ -c1, b1 ~ -s1, a2 ~ -c2 (matched pairs); the
crossed pairs (a1 ~ -s1, b1 ~ -c1) are the specificity control.

Correlations integrate over posterior draws and bootstrap subjects
(brain_behavior.draws_correlation). V1 test-retest of each harmonic comes from
the two sessions separately -- but compute_expected_decoded_orientation_vonmises
fits ONE set of basis weights over both sessions and only the noise model per
session, so that is not an independent test-retest.

Writes notes/figures/hv_prior_vs_v1.pdf and notes/data/hv_prior_vs_v1.tsv.
"""
from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.cogmodels.brain_behavior import draws_correlation
from abstract_values.connective_fields.plot_gates import letter  # sets rcParams
from abstract_values.utils.data import BIDS_FOLDER

REPO = Path(__file__).resolve().parents[2]
COG = BIDS_FOLDER / 'derivatives' / 'cogmodels'
FITS = {'Fourier k=1 (motor noise)': 'efficient_coding_categorical_fourier1_grid81_trunc_hn_motor_lapse0.003',
        'Fourier k=2 (no seam)': 'efficient_coding_categorical_fourier2_noseam'}
HARM = {'a1': 'c1', 'b1': 's1', 'a2': 'c2'}
LAB = {'a1': 'Horizontal vs vertical', 'b1': '45° vs 135°', 'a2': 'Cardinal vs oblique'}
MODEL_C, V1_C, CROSS_C = '#C44E52', '#3B5BA5', '.6'


def harmonics(theta_deg, y):
    t = np.deg2rad(theta_deg)
    X = np.column_stack([np.ones_like(t), np.cos(2 * t), np.sin(2 * t), np.cos(4 * t), np.sin(4 * t)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(zip(['c0', 'c1', 's1', 'c2', 's2'], beta))


def v1_harmonics():
    e = pd.read_csv(REPO / 'notes' / 'data' / 'expected_uncertainty_orientation_v1.tsv', sep='\t')
    e['log_sd'] = np.log(e.sd_E)
    rows = []
    for (s, ses), g in e.groupby(['subject', 'session']):
        g = g.groupby('orientation').log_sd.mean()
        rows.append({'subject': s, 'session': ses, **harmonics(g.index.values, g.values)})
    per_ses = pd.DataFrame(rows)
    both = e.groupby(['subject', 'orientation']).log_sd.mean().reset_index()
    joint = pd.DataFrame([{'subject': s, **harmonics(g.orientation.values, g.log_sd.values)}
                          for s, g in both.groupby('subject')]).set_index('subject')
    profile = both.groupby('orientation').log_sd.agg(['mean', 'sem'])
    return joint, per_ses, profile


def load_fit(tag):
    d = az.from_netcdf(COG / f'{tag}_trace.nc')
    post = d.posterior
    subs = [int(x) for x in post.prior_a1.coords['subject'].values]
    draws = {k: post[f'prior_{k}'].values.reshape(-1, len(subs)) for k in HARM
             if f'prior_{k}' in post.data_vars}
    rhat = az.rhat(d, var_names=[f'prior_{k}' for k in draws])
    diag = {k: (float(rhat[f'prior_{k}'].max()), int((rhat[f'prior_{k}'] > 1.01).sum()))
            for k in draws}
    return subs, draws, diag


def main():
    joint, per_ses, profile = v1_harmonics()
    retest = {}
    for c in ('c1', 's1', 'c2'):
        w = per_ses.pivot(index='subject', columns='session', values=c).dropna()
        retest[c] = stats.pearsonr(w[1], w[2])
    print('V1 harmonic test-retest (session 1 vs 2):',
          {k: f'r = {v.statistic:.2f}, p = {v.pvalue:.2g}' for k, v in retest.items()})
    print('V1 group harmonics (log SD):',
          {c: f'{joint[c].mean():+.3f} ± {joint[c].sem():.3f} (p = {stats.ttest_1samp(joint[c], 0).pvalue:.2g})'
           for c in ('c1', 's1', 'c2', 's2')})

    rows, fitinfo = [], {}
    for name, tag in FITS.items():
        subs, draws, diag = load_fit(tag)
        v1 = joint.reindex(subs)
        keep = v1.notna().all(1).to_numpy()
        fitinfo[name] = (subs, draws, keep, diag)
        print(f'{name}: n = {keep.sum()}; r_hat max / n(>1.01):', diag)
        for k in draws:
            gm = draws[k].mean(1)
            print(f'  group {k}: {gm.mean():+.3f} [{np.percentile(gm, 2.5):+.3f}, {np.percentile(gm, 97.5):+.3f}]'
                  f'  posterior sd per subject (median) {np.median(draws[k].std(0)):.3f}, '
                  f'between-subject sd of means {draws[k].mean(0).std():.3f}')
            for k2, c in HARM.items():
                if k2 not in draws and c not in v1:
                    continue
                if (k, c) not in [('a1', 'c1'), ('b1', 's1'), ('a2', 'c2'), ('a1', 's1'), ('b1', 'c1')]:
                    continue
                rho, lo, hi = draws_correlation(draws[k][:, keep], -v1[c].to_numpy()[keep])
                rows.append({'fit': name, 'model': k, 'v1': f'-{c}', 'matched': HARM[k] == c,
                             'rho': rho, 'lo': lo, 'hi': hi, 'n': int(keep.sum())})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    res.to_csv(REPO / 'notes' / 'data' / 'hv_prior_vs_v1.tsv', sep='\t', index=False)

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(7.25, 2.3), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.1, 1, 1, 1.3]})
    ax = axes[0]
    th = profile.index.values
    ax.fill_between(th, profile['mean'] - profile['sem'], profile['mean'] + profile['sem'],
                    color=V1_C, alpha=.2, lw=0)
    ax.plot(th, profile['mean'], color=V1_C, lw=1.2)
    g = harmonics(th, profile['mean'].values)
    t = np.deg2rad(th)
    ax.plot(th, g['c0'] + g['c1'] * np.cos(2 * t) + g['s1'] * np.sin(2 * t), color='.2', lw=1,
            ls='--')
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel('Orientation (°)')
    ax.set_ylabel('V1 log expected SD')
    ax.text(.03, .98, 'cos 2θ + sin 2θ part', transform=ax.transAxes, va='top', fontsize=6.5,
            color='.2')
    letter(ax, 'a')

    ax = axes[1]
    w = per_ses.pivot(index='subject', columns='session', values='c1').dropna()
    ax.scatter(-w[1], -w[2], s=9, color=V1_C, lw=0)
    ax.axhline(0, color='.8', lw=.6)
    ax.axvline(0, color='.8', lw=.6)
    ax.set_xlabel('V1 horizontal advantage, ses 1')
    ax.set_ylabel('V1 horizontal advantage, ses 2')
    ax.text(.03, .98, f'r = {retest["c1"].statistic:.2f}\n(one weight fit,\nper-session noise)',
            transform=ax.transAxes, va='top', fontsize=6, color='.2')
    letter(ax, 'b')

    ax = axes[2]
    subs, draws, keep, _ = fitinfo['Fourier k=1 (motor noise)']
    a1 = draws['a1'][:, keep]
    x = -joint.reindex(subs)['c1'].to_numpy()[keep]
    lo, hi = np.percentile(a1, [2.5, 97.5], axis=0)
    ax.errorbar(x, a1.mean(0), [a1.mean(0) - lo, hi - a1.mean(0)], fmt='o', ms=3, color=MODEL_C,
                elinewidth=.6, alpha=.8)
    ax.axhline(0, color='.8', lw=.6)
    ax.axvline(0, color='.8', lw=.6)
    ax.set_xlabel('V1 horizontal advantage (−c1)')
    ax.set_ylabel('Prior a1 (k=1 fit)')
    r = res[(res.fit == 'Fourier k=1 (motor noise)') & (res.model == 'a1') & (res.v1 == '-c1')].iloc[0]
    ax.text(.03, .98, f'ρ = {r.rho:.2f} [{r.lo:.2f}, {r.hi:.2f}]', transform=ax.transAxes,
            va='top', fontsize=6.5, color='.2')
    letter(ax, 'c')

    ax = axes[3]
    labels = []
    for i, row in res.reset_index(drop=True).iterrows():
        col = MODEL_C if row.matched else CROSS_C
        ax.plot([row.lo, row.hi], [i, i], color=col, lw=1.2)
        ax.plot(row.rho, i, 'o', color=col, ms=3.5)
        labels.append(f'{row.model} ~ V1 {row.v1}  [k={1 if "k=1" in row.fit else 2}]')
    ax.axvline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_yticks(range(len(labels)), labels, fontsize=5.5)
    ax.set_ylim(len(labels) - .5, -.5)
    ax.set_xlabel('Spearman ρ (95% interval)')
    letter(ax, 'd')
    for a in axes[:3]:
        sns.despine(ax=a, offset=3, trim=True)
    sns.despine(ax=axes[3], offset=3, left=True)
    out = REPO / 'notes' / 'figures' / 'hv_prior_vs_v1.pdf'
    fig.savefig(out)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
