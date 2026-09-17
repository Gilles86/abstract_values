#!/usr/bin/env python3
"""How orientation-specific is the mapping-dependent NPC <- V1 coupling?

Aligned coupling profiles (as in plot_coupling_explainer.py, panel e) for V1
channels of increasing resolution, plus the mapping score for each.

  Binned V1 voxels         8 and 16 channels (22.5 / 11.25 deg)
  Inverted encoding model  8 basis kappa 2 (~49 deg wide), 16 basis kappa 8
                           (~24 deg), 24 basis kappa 16 (~17 deg)

  python -m abstract_values.connective_fields.plot_specificity
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from abstract_values.connective_fields.plot_coupling_explainer import (
    CUR_COL, OTH_COL, ROOT, profile_panel)
from abstract_values.connective_fields.plot_gates import letter, read
from abstract_values.connective_fields.profiles import MIN_SHIFT  # noqa: F401 (documents the selection)

REPO = Path(__file__).resolve().parents[2]
VARIANTS = [  # suffix, label, display step
    ('', 'Bins, 8 channels', None),
    ('_bins-16', 'Bins, 16 channels', None),
    ('_iem', 'Encoding model, κ = 2', 15),
    ('_iem-k16-kappa8', 'Encoding model, κ = 8', 10),
    ('_iem-k24-kappa16', 'Encoding model, κ = 16', 10),
]


def main(out):
    fig = plt.figure(figsize=(7.25, 4.4), constrained_layout=True)
    axes = fig.subplots(2, 3)
    prof_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2]]
    for ax, (suffix, label, step), lab in zip(prof_axes, VARIANTS, 'abdef'):
        profile_panel(ax, ROOT / f'profiles{suffix}', 0, label, step=step)
        letter(ax, lab)
    lo = min(ax.get_ylim()[0] for ax in prof_axes)
    hi = max(ax.get_ylim()[1] for ax in prof_axes)
    for ax in prof_axes:
        ax.set_ylim(lo, hi + .4 * (hi - lo))
    for ax in (axes[0, 1], axes[1, 1], axes[1, 2]):
        ax.set_ylabel('')
    ax = axes[0, 0]
    ax.text(.03, .88, 'θ* in this session', transform=ax.transAxes, va='top', color=CUR_COL,
            fontsize=6.5)
    ax.text(.03, .79, 'θ* in the other session', transform=ax.transAxes, va='top',
            color=OTH_COL, fontsize=6.5)

    # c: mapping score per variant
    ax = axes[0, 2]
    rng = np.random.default_rng(3)
    print('variant                     mean     sem      t      p')
    for i, (suffix, label, _) in enumerate(VARIANTS):
        s = read(ROOT / f'coupling{suffix}', 'scores')
        s = s[s.direction == 'npc_from_v1']
        obs = s[s.kind == 'observed'].set_index('subject').score
        d = obs - s[s.kind == 'label-shuffle'].groupby('subject').score.mean().reindex(obs.index)
        t = stats.ttest_1samp(d, 0)
        print(f'{label:26s} {d.mean():+.4f} {d.sem():.4f} {t.statistic:6.2f} {t.pvalue:.2g}')
        ax.scatter(i + rng.uniform(-.1, .1, len(d)), d, s=5, color=CUR_COL, alpha=.35, lw=0)
        ax.errorbar(i, d.mean(), d.sem(), color='.15', lw=.9, zorder=3)
        ax.plot(i, d.mean(), 'D', ms=4.5, mfc=CUR_COL, mec='.15', mew=1, zorder=4)
        ax.text(i, .066, f'{t.pvalue:.1g}'.replace('0.', '.'), ha='center', fontsize=5.5,
                color='.3')
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks(range(len(VARIANTS)), ['8', '16', 'κ 2', 'κ 8', 'κ 16'])
    ax.set_xlabel('Bins          Encoding model')
    ax.set_ylabel('Mapping score (Δr)')
    ax.set_ylim(-.03, .072)
    ax.text(-.55, .066, 'p =', ha='right', fontsize=5.5, color='.3')
    letter(ax, 'c')

    for a in axes.ravel():
        sns.despine(ax=a, offset=4, trim=True)
    fig.savefig(out)
    print(f'saved {out}')


if __name__ == '__main__':
    main(REPO / 'notes' / 'figures' / 'cf_coupling_specificity.pdf')
