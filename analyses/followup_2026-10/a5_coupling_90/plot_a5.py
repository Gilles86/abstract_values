#!/usr/bin/env python3
"""A5 slide figure (run locally on the group tables from collect_a5.py).

  a  How many V1 voxels prefer each orientation (share per subject): the bin sizes
     the permutation / subsampling nulls have to correct for.
  b  Coupling profile over V1 preferred orientation, IPS0-5, equal-count subsampled
     bins, per mapping (each subject's mean over bins removed).
  c  delta90 (90-deg bin minus the other bins, observed minus permutation-null
     mean) per IPS segment and mapping.
  d  The same contrast at IPS0-5 across analysis variants, and for the 0-deg bin.

Primary specification (fixed before looking at group results): whole Benson V1,
voxels with cvR2(vonmises-prf.cv) > null, 6 bins of 30 deg; IPS all voxels; raw
coupling; estimator observed minus permutation mean.

Usage:
  python plot_a5.py --group-dir /data/ds-abstractvalue/derivatives/followup_2026-10/a5_coupling_90/group \
      --out .../a5_coupling_90.pdf
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'Arial'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12, 'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5, 'axes.linewidth': 0.9, 'axes.spines.top': False,
    'axes.spines.right': False, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5, 'lines.linewidth': 1.6,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03})

MAP_C = {'cdf': '#3B5BA5', 'inverse_cdf': '#C44E52'}
MAP_L = {'cdf': 'Misaligned (90° value rare)', 'inverse_cdf': 'Aligned (90° value frequent)'}
PRIMARY = dict(n_bins=6, v1_sel='v1-tuned', ips_sel='all', metric='raw')
SEG_L = {'IPS0': 'IPS0', 'IPS12': 'IPS1–2', 'IPS345': 'IPS3–5', 'IPS0-5': 'IPS0–5'}


def q(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def boot_ci(x, rng, n=10000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    b = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def letter(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=13, fontweight='bold',
            va='bottom', ha='right')


def main(group_dir, out):
    g = Path(group_dir)
    rd = dict(sep='\t', dtype={'subject': str})
    cnt = pd.read_csv(g / 'subject_counts.tsv', **rd)
    bins = pd.read_csv(g / 'subject_bins.tsv', **rd)
    con = pd.read_csv(g / 'subject_contrasts.tsv', **rd)
    grp = pd.read_csv(g / 'group_contrasts.tsv', sep='\t')
    rng = np.random.default_rng(0)
    K = PRIMARY['n_bins']

    fig = plt.figure(figsize=(13.3, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.15, 1.35], hspace=0.55, wspace=0.42)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1])
    ax_d = fig.add_subplot(gs[:, 2])

    # a: share of V1 voxels per bin
    c = q(cnt, n_bins=K, v1_sel=PRIMARY['v1_sel']).copy()
    c['share'] = c['n_vox'] / c.groupby('subject')['n_vox'].transform('sum')
    sns.stripplot(data=c, x='centre', y='share', color='0.6', size=2.6, alpha=.5, jitter=.2,
                  ax=ax_a, zorder=1)
    m = c.groupby('centre')['share'].mean()
    ax_a.plot(range(K), m.values, 'o', color='k', ms=6, zorder=3)
    ax_a.axhline(1 / K, color='0.7', lw=.7, ls='--', zorder=0)
    ax_a.text(K - .5, 1 / K, 'Uniform', fontsize=9.5, color='0.45', va='bottom', ha='right')
    ax_a.set_xticks(range(K), [f'{x:g}' for x in m.index])
    ax_a.set_xlabel('V1 preferred orientation (deg)')
    ax_a.set_ylabel('Share of V1 voxels')
    med90 = q(c, centre=90.)['n_vox'].median()
    ax_a.text(.02, .98, f'90° bin: median {med90:.0f} voxels', transform=ax_a.transAxes,
              fontsize=9.5, va='top')
    letter(ax_a, 'a')

    # b: coupling profile, subsampled, IPS0-5
    b = q(bins, **PRIMARY, segment='IPS0-5').copy()
    b['y'] = b['coupling_sub'] - b.groupby(['subject', 'condition'])['coupling_sub'].transform('mean')
    for cond, off in (('cdf', -.08), ('inverse_cdf', .08)):
        rows = [boot_ci(q(b, condition=cond, centre=x)['y'], rng) for x in sorted(b['centre'].unique())]
        mu, lo, hi = np.array(rows).T
        xs = np.arange(K) + off
        ax_b.errorbar(xs, mu, yerr=[mu - lo, hi - mu], fmt='o-', color=MAP_C[cond], ms=4.5,
                      lw=1.3, elinewidth=1, capsize=0)
    ax_b.axhline(0, color='0.7', lw=.7, ls='--', zorder=0)
    ax_b.set_xticks(range(K), [f'{x:g}' for x in sorted(b['centre'].unique())])
    ax_b.set_xlabel('V1 preferred orientation (deg)')
    ax_b.set_ylabel('V1→IPS0–5 coupling\n(z, rel. mean; equal n)')
    ax_b.axvspan(K // 2 - .4, K // 2 + .4, color='0.93', zorder=-1, lw=0)
    letter(ax_b, 'b')

    # c: delta90 per segment x mapping, observed - perm
    segs = list(SEG_L)
    for i, seg in enumerate(segs):
        for cond, off in (('cdf', -.15), ('inverse_cdf', .15)):
            d = q(con, **PRIMARY, segment=seg, condition=cond, contrast='delta90')['delta_obs_minus_perm']
            mu, lo, hi = boot_ci(d, rng)
            ax_c.scatter(np.full(len(d), i + off) + rng.uniform(-.05, .05, len(d)), d,
                         s=7, color=MAP_C[cond], alpha=.3, lw=0, zorder=1)
            ax_c.errorbar(i + off, mu, yerr=[[mu - lo], [hi - mu]], fmt='D', color=MAP_C[cond],
                          mec='k', mew=1, ms=7, elinewidth=1.4, capsize=0, zorder=3)
    ax_c.axhline(0, color='0.6', lw=.8, ls='--', zorder=0)
    ax_c.set_xticks(range(len(segs)), [SEG_L[s] for s in segs])
    ax_c.set_ylabel('Δ coupling, 90° bin − other bins\n(observed − permutation null, z)')
    ylim = ax_c.get_ylim()
    for j, cond in enumerate(('inverse_cdf', 'cdf')):
        ax_c.text(.03, .98 - j * .055, MAP_L[cond], color=MAP_C[cond], transform=ax_c.transAxes,
                  fontsize=10.5, va='top')
    ax_c.set_ylim(ylim[0], ylim[1] + .25 * (ylim[1] - ylim[0]))
    letter(ax_c, 'c')

    # d: robustness forest at IPS0-5
    variants = [
        ('Primary: V1 tuned, 6 bins, raw', dict(PRIMARY), 'delta90', 'delta_obs_minus_perm'),
        ('Equal-count subsampling', dict(PRIMARY), 'delta90', 'delta_sub'),
        ('8 bins', {**PRIMARY, 'n_bins': 8}, 'delta90', 'delta_obs_minus_perm'),
        ('Stimulated annulus, all voxels', {**PRIMARY, 'v1_sel': 'ann-all'}, 'delta90', 'delta_obs_minus_perm'),
        ('Annulus, all voxels, subsampled', {**PRIMARY, 'v1_sel': 'ann-all'}, 'delta90', 'delta_sub'),
        ('Specific (V1 mean removed, r²)', {**PRIMARY, 'metric': 'specific'}, 'delta90', 'delta_obs_minus_perm'),
        ('IPS value-tuned voxels only', {**PRIMARY, 'ips_sel': 'aprf'}, 'delta90', 'delta_obs_minus_perm'),
        ('Horizontal (0°) bin instead', dict(PRIMARY), 'delta0', 'delta_obs_minus_perm'),
    ]
    for i, (lab, spec, contrast, est) in enumerate(variants):
        for cond, off in (('cdf', .15), ('inverse_cdf', -.15)):
            r = q(grp, **spec, segment='IPS0-5', condition=cond, contrast=contrast, estimator=est)
            if not len(r):
                continue
            r = r.iloc[0]
            # standardise by the across-subject SD so metrics are comparable: dz with CI
            d = q(con, **spec, segment='IPS0-5', condition=cond, contrast=contrast)[est].dropna()
            sd = d.std(ddof=1)
            ax_d.errorbar(r['mean'] / sd, i + off, xerr=[[(r['mean'] - r['ci_lo']) / sd],
                                                         [(r['ci_hi'] - r['mean']) / sd]],
                          fmt='o', color=MAP_C[cond], ms=5.5, elinewidth=1.3, capsize=0)
    ax_d.axvline(0, color='0.6', lw=.8, ls='--', zorder=0)
    ax_d.set_yticks(range(len(variants)), [v[0] for v in variants])
    ax_d.invert_yaxis()
    ax_d.set_xlabel('Effect size dz (IPS0–5), 95% CI')
    ax_d.tick_params(axis='y', length=0)
    letter(ax_d, 'd')

    for ax in (ax_a, ax_b, ax_c):
        sns.despine(ax=ax, offset=5, trim=False)
    sns.despine(ax=ax_d, offset=5, left=True)
    fig.suptitle('Exploratory: V1→IPS residual coupling is not specifically stronger '
                 'for 90°-preferring V1 voxels' if False else '', fontsize=13)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=200)
    print(f'saved {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--group-dir', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    main(a.group_dir, a.out)
