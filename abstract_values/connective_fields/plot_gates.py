#!/usr/bin/env python3
"""Cohort summary of the connective-field step-0 gates (see gates.py).

Reads the per-subject TSVs (rsync them from the cluster first) and writes
notes/figures/cf_gates.pdf plus a printed pass/fail table.

  rsync -av sciencecluster:/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/connective_fields/ \
      /data/ds-abstractvalue/derivatives/connective_fields/
  python -m abstract_values.connective_fields.plot_gates
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.utils.data import BIDS_FOLDER

REPO = Path(__file__).resolve().parents[2]

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'lines.linewidth': 1.2, 'lines.markersize': 4, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': .02,
})

V1_COL, NPC_COL = '#3B5BA5', '#C44E52'
SPEC_COL, INV_COL = '#C44E52', '#7F7F7F'
VARIANT_COL = {'selected': '#3B5BA5', 'all': '#9C9C9C'}


def read(gates_dir, desc):
    files = sorted(gates_dir.glob(f'sub-*/sub-*_desc-{desc}.tsv'))
    if not files:
        raise SystemExit(f'No {desc} files under {gates_dir}')
    return pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str}) for f in files],
                     ignore_index=True)


def summary_points(ax, df, x, y, order, color, offset=0., label=None):
    """Subject dots plus a mean ± SEM diamond per x category."""
    rng = np.random.default_rng(1)
    for i, cat in enumerate(order):
        v = df.loc[df[x] == cat, y].dropna().to_numpy()
        if not len(v):
            continue
        xs = i + offset + rng.uniform(-.07, .07, len(v))
        ax.scatter(xs, v, s=6, color=color, alpha=.35, lw=0, zorder=2)
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        ax.errorbar(i + offset, m, se, color='0.15', lw=.9, capsize=0, zorder=3)
        ax.plot(i + offset, m, 'D', ms=5, mfc=color, mec='0.15', mew=1., zorder=4)
    if label:
        ax.text(len(order) - 1 + offset + .15, df.loc[df[x] == order[-1], y].mean(),
                label, color=color, fontsize=7, va='center')


def variant_key(ax, x, y, variants, dy=.09, seg=.1):
    """Solid = tuned V1 voxels, dashed = all V1 voxels, drawn as glyphs."""
    for i, v in enumerate(variants):
        yy = y - i * dy
        ax.plot([x, x + seg], [yy, yy], transform=ax.transAxes, color='.3', lw=1.2,
                ls='-' if v == 'selected' else '--', clip_on=False)
        ax.text(x + seg + .03, yy, 'Tuned V1 voxels' if v == 'selected' else 'All V1 voxels',
                transform=ax.transAxes, color='.3', fontsize=6.5, va='center')


def letter(ax, s):
    ax.text(-.2, 1.06, s, transform=ax.transAxes, fontsize=8, fontweight='bold',
            va='bottom', ha='right')


def main(bids_folder, out):
    root = Path(bids_folder) / 'derivatives' / 'connective_fields'
    variants = {v: root / d for v, d in (('selected', 'gates'), ('all', 'gates_v1-all'))
                if (root / d).exists()}

    tun = read(variants['selected'], 'tuningreliability')
    tun = tun[tun.selection == 'selected'].copy()
    tun['split'] = tun['split'].str.replace(r'within-ses\d', 'within', regex=True)
    tun = tun.groupby(['subject', 'roi', 'split'], as_index=False)['reliability'].mean()

    rel = pd.concat([read(d, 'cfreliability') for d in variants.values()], ignore_index=True)
    sim = pd.concat([read(d, 'simulation') for d in variants.values()], ignore_index=True)

    # Counterbalancing: condition == session within subject; only parity balance
    # lets session effects cancel at the group level.
    first = rel[rel.session == 1].drop_duplicates('subject').set_index('subject')['condition']
    print(f'n = {len(first)} subjects; ses-1 mapping: {first.value_counts().to_dict()}')

    fig, axes = plt.subplots(1, 4, figsize=(7.25, 2.1), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.25, 1.25, 1, 1]})

    # a: tuning reliability
    ax = axes[0]
    order = ['within', 'across-ses', 'pooled-oddeven']
    summary_points(ax, tun[tun.roi == 'V1'], 'split', 'reliability', order, V1_COL, -.17)
    summary_points(ax, tun[tun.roi == 'NPC'], 'split', 'reliability', order, NPC_COL, .17)
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks(range(3), ['Within\nsession', 'Across\nsessions', 'Odd vs\neven runs'])
    ax.set_ylabel('Split-half reliability (r)')
    ax.text(.03, .98, 'V1 orientation', color=V1_COL, transform=ax.transAxes, va='top')
    ax.text(.03, .89, 'NPC value', color=NPC_COL, transform=ax.transAxes, va='top')
    letter(ax, 'a')

    # b: CF reliability (odd vs even runs within a session)
    ax = axes[1]
    long = (rel.groupby(['subject', 'v1_voxels'], as_index=False)
               [['voxel_rel', 'voxel_specific_rel', 'roi_profile_rel']].mean()
               .melt(['subject', 'v1_voxels'], var_name='metric', value_name='r'))
    order = ['roi_profile_rel', 'voxel_rel', 'voxel_specific_rel']
    for k, (v, off) in enumerate(zip(variants, (-.17, .17))):
        summary_points(ax, long[long.v1_voxels == v], 'metric', 'r', order,
                       VARIANT_COL[v], off if len(variants) > 1 else 0)
        ax.text(.03, .98 - .09 * k,
                'Tuned V1 voxels' if v == 'selected' else 'All V1 voxels',
                color=VARIANT_COL[v], transform=ax.transAxes, va='top')
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_ylim(top=1.2)
    ax.set_yticks([-.5, 0, .5, 1])
    ax.set_xticks(range(3), ['ROI\nmean', 'Per\nvoxel', 'Voxel-\nspecific'])
    ax.set_ylabel('CF reliability (r)')
    letter(ax, 'b')

    # c: power -- group t-test across subjects, per permutation
    ax = axes[2]
    for v, ls in zip(variants, ('-', '--')):
        for inj, col in (('specific', SPEC_COL), ('invariant', INV_COL)):
            d = sim[(sim.v1_voxels == v) & (sim.injection == inj)]
            pw = (d.groupby(['amplitude', 'perm'])['score']
                   .apply(lambda s: stats.ttest_1samp(s, 0, alternative='greater').pvalue < .05)
                   .groupby('amplitude').mean())
            ax.plot(pw.index, pw.values, ls=ls, color=col, marker='o', ms=3)
    ax.axhline(.05, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xlabel('Injected coupling (r)')
    ax.set_ylabel('Power (p < .05)')
    ax.set_ylim(-.02, 1.02)
    variant_key(ax, .4, .8, variants)
    ax.text(.97, .45, 'Condition-specific', color=SPEC_COL, transform=ax.transAxes, ha='right')
    ax.text(.97, .14, 'Condition-invariant', color=INV_COL, transform=ax.transAxes, ha='right')
    letter(ax, 'c')

    # d: where the observed CF magnitude sits on the injection scale
    ax = axes[3]
    for v, ls in zip(variants, ('-', '--')):
        d = (sim[(sim.v1_voxels == v) & (sim.injection == 'specific')]
             .groupby(['subject', 'amplitude'])['cf_specific_sd'].mean()
             .groupby('amplitude').mean())
        ax.plot(d.index, d.values, ls=ls, color='.3', marker='o', ms=3)
        obs = rel[rel.v1_voxels == v]['full_cf_specific_sd'].mean()
        ax.axhline(obs, color=VARIANT_COL[v], lw=1., ls=ls)
    ax.set_xlabel('Injected coupling (r)')
    ax.set_ylabel('Voxel-specific CF SD')
    ax.text(.03, .98, 'Observed', color=VARIANT_COL['selected'], transform=ax.transAxes,
            va='top')
    ax.text(.03, .89, 'Shuffled + injected', color='.3', transform=ax.transAxes, va='top')
    variant_key(ax, .03, .38, variants)
    letter(ax, 'd')

    for ax in axes:
        sns.despine(ax=ax, offset=5, trim=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f'saved {out}')

    # ── table ────────────────────────────────────────────────────────────
    def t(v):
        v = pd.Series(v).dropna()
        r = stats.ttest_1samp(v, 0)
        return f'{v.mean():+.3f} ± {v.sem():.3f}  (t{len(v)-1} = {r.statistic:.2f}, p = {r.pvalue:.2g})'
    print('\nTuning reliability (tuned voxels):')
    for (roi, split), g in tun.groupby(['roi', 'split']):
        print(f'  {roi:4s} {split:15s} {t(g.reliability)}')
    print('\nCF reliability (odd vs even runs):')
    cfr = rel.groupby(['subject', 'v1_voxels'], as_index=False).mean(numeric_only=True)
    for v, g in cfr.groupby('v1_voxels'):
        for m in ('roi_profile_rel', 'voxel_rel', 'voxel_specific_rel'):
            print(f'  V1 {v:8s} {m:20s} {t(g[m])}')
        print(f'  V1 {v:8s} min V1 voxels/channel: {int(rel[rel.v1_voxels == v].min_v1_per_channel.min())}')
    print('\nInjection (mean score across subjects; power):')
    for (v, inj, a), d in sim.groupby(['v1_voxels', 'injection', 'amplitude']):
        pw = (d.groupby('perm')['score']
               .apply(lambda s: stats.ttest_1samp(s, 0, alternative='greater').pvalue < .05).mean())
        print(f'  V1 {v:8s} {inj:9s} a={a:.2f}  score={d.score.mean():+.4f}  power={pw:.2f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--out', default=str(REPO / 'notes' / 'figures' / 'cf_gates.pdf'))
    a = p.parse_args()
    main(a.bids_folder, Path(a.out))
