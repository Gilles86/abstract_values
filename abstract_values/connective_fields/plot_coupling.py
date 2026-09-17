#!/usr/bin/env python3
"""Cohort result of the mapping-coupling test (see test_coupling.py).

  rsync -av sciencecluster:/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/connective_fields/ \
      /data/ds-abstractvalue/derivatives/connective_fields/
  python -m abstract_values.connective_fields.plot_coupling
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.connective_fields.plot_gates import letter, read  # also sets rcParams
from abstract_values.utils.data import BIDS_FOLDER

REPO = Path(__file__).resolve().parents[2]
OBS_COL, NULL_COL = '#C44E52', '#9C9C9C'
DIRECTIONS = {'npc_from_v1': 'NPC ← V1', 'v1_from_npc': 'V1 ← NPC'}


def group_stats(scores):
    """Per direction: t-tests across subjects and a group-level label permutation test."""
    rows = []
    for d, g in scores.groupby('direction'):
        obs = g[g.kind == 'observed'].set_index('subject')['score']
        sh = g[g.kind == 'label-shuffle']
        null_mean = sh.groupby('subject')['score'].mean().reindex(obs.index)
        diff = obs - null_mean
        # Group null: the cohort mean of one shuffle per subject, paired by perm index.
        group_null = sh.pivot(index='perm', columns='subject', values='score').mean(1)
        p_perm = (np.sum(group_null.to_numpy() - null_mean.mean() >= diff.mean()) + 1) / (len(group_null) + 1)
        for name, v in (('observed', obs), ('label-shuffle', null_mean), ('observed - shuffle', diff)):
            t = stats.ttest_1samp(v, 0)
            rows.append({'direction': d, 'quantity': name, 'n': len(v), 'mean': v.mean(),
                         'sem': v.sem(), 't': t.statistic, 'p': t.pvalue})
        rows[-1]['p_group_permutation'] = p_perm
    return pd.DataFrame(rows)


def main(bids_folder, out):
    root = Path(bids_folder) / 'derivatives' / 'connective_fields' / 'coupling'
    scores = read(root, 'scores')
    by = read(root, 'bytuning')

    st = group_stats(scores)
    pd.set_option('display.width', 160)
    print(st.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
    st.to_csv(REPO / 'notes' / 'data' / 'cf_coupling_group_stats.tsv', sep='\t', index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.2), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.1, 1.2, 1.2]})

    ax = axes[0]
    rng = np.random.default_rng(1)
    for i, d in enumerate(DIRECTIONS):
        g = scores[scores.direction == d]
        obs = g[g.kind == 'observed'].set_index('subject')['score']
        null = g[g.kind == 'label-shuffle'].groupby('subject')['score'].mean().reindex(obs.index)
        for off, v, col in ((-.16, null, NULL_COL), (.16, obs, OBS_COL)):
            xs = i + off + rng.uniform(-.05, .05, len(v))
            ax.scatter(xs, v, s=6, color=col, alpha=.4, lw=0, zorder=2)
            ax.errorbar(i + off, v.mean(), v.sem(), color='.15', lw=.9, zorder=3)
            ax.plot(i + off, v.mean(), 'D', ms=5, mfc=col, mec='.15', mew=1, zorder=4)
        for s in obs.index:
            ax.plot([i - .16, i + .16], [null[s], obs[s]], color='.8', lw=.4, zorder=1)
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks(range(2), list(DIRECTIONS.values()))
    ax.set_ylabel('Mapping score (Δr)')
    ax.text(.03, .98, 'Observed', color=OBS_COL, transform=ax.transAxes, va='top')
    ax.text(.03, .89, 'Labels shuffled', color=NULL_COL, transform=ax.transAxes, va='top')
    letter(ax, 'a')

    for ax, d, xlabel, ticks, lab in (
            (axes[1], 'npc_from_v1', 'NPC preferred value (CHF)', [2, 12, 22, 32, 42], 'b'),
            (axes[2], 'v1_from_npc', 'V1 preferred orientation (deg)', [0, 45, 90, 135, 180], 'c')):
        g = by[by.direction == d]
        sns.lineplot(data=g, x='tuning', y='score', errorbar=('se', 1), color=OBS_COL,
                     marker='o', ms=3, ax=ax, err_kws={'lw': 0})
        ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
        ax.set_xticks(ticks)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Mapping score (Δr)')
        ax.set_title(DIRECTIONS[d], fontsize=8)
        letter(ax, lab)

    for ax in axes:
        sns.despine(ax=ax, offset=5, trim=True)
    fig.savefig(out)
    print(f'saved {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--out', default=str(REPO / 'notes' / 'figures' / 'cf_coupling.pdf'))
    a = p.parse_args()
    main(a.bids_folder, Path(a.out))
