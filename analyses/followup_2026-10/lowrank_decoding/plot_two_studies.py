#!/usr/bin/env python3
"""Decoding accuracy with spherical, van Bergen & Jehee and low-rank noise in
two studies: abstract_values (top: orientation in V1-V3, value in V1/IPS/NPCr;
plot_mean_r.py) and neural_priors (bottom: numerosity in NPCr, paper
configuration, truth = response; decoding_nvoxels_collect.py there),
split by prior range, in the paper's convention (narrow posterior truncated
at 25: the grid runs to 40 for both ranges, and calibrated low-rank posteriors
leak mass above 25 on narrow trials, which costs r without truncation).

neural_priors voxel sets: 'cvR² > 0' = production selection (voxels with
cvR² > 0 in the training runs, ~90-570, median ~200); 100 / 200 / 500 =
top-N by training R². Low-rank exists only for the production set.

    python plot_two_studies.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

import plot_mean_r as pm

NP = Path('/Users/gdehol/git/neural_priors/notes/data/decoding_nvoxels_by_subject.tsv')
NP_SETS = [('0', 'cvR² > 0'), ('100', '100'), ('200', '200'), ('500', '500')]
NP_NOISE = [('spherical', pm.SPH), ('jehee', pm.JEH), ('lowrank', pm.LR)]


def dots(ax, x, v, col, mk, filled, rng):
    v = v.dropna()
    ax.scatter(x + rng.uniform(-.025, .025, len(v)), v, s=2, color=col, alpha=.25, lw=0, zorder=1)
    h = stats.t.ppf(.975, len(v) - 1) * v.sem()
    ax.plot([x, x], [v.mean() - h, v.mean() + h], color=col, lw=1.1, zorder=2)
    ax.plot(x, v.mean(), mk, ms=4, color=col, mfc=col if filled else 'white', mew=1, zorder=3)


def np_panel(ax, d, metric, rng):
    off = np.linspace(-.22, .22, len(NP_NOISE))
    for i, (nv, _) in enumerate(NP_SETS):
        for o, (noise, col) in zip(off, NP_NOISE):
            v = d[(d.noise == noise) & (d.n_voxels == nv)][metric]
            if len(v):
                dots(ax, i + o, v, col, 'o', True, rng)
    ax.set_xticks(range(len(NP_SETS)), [lab for _, lab in NP_SETS])
    ax.set_xlim(-.5, len(NP_SETS) - .5)
    ax.set_xlabel('Voxels (NPCr)')
    ax.tick_params(axis='x', labelsize=6)


def main():
    df = pm.load()
    npd = pd.read_csv(NP, sep='\t', dtype={'subject': str, 'n_voxels': str})
    npd = npd[npd.config == 'responses']  # range: narrow / wide / both
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(7.25, 5.6), constrained_layout=True)
    top, bot = fig.subfigures(2, 1, height_ratios=[1, .95], hspace=.06)
    axes = top.subplots(1, 2, gridspec_kw={'width_ratios': [3, 5]})
    off = np.linspace(-.38, .38, len(pm.CELLS))
    for ax, (q, rois) in zip(axes, pm.ROIS.items()):
        for i, roi in enumerate(rois):
            for o, (noise, nv, lab, col, mk, filled) in zip(off, pm.CELLS):
                v = df[(df.quantity == q) & (df.roi == roi) & (df.noise == noise)
                       & (df.n_voxels == nv)].r
                if len(v.dropna()) >= pm.MIN_N:
                    dots(ax, i + o, v, col, mk, filled, rng)
        ax.set_xticks(range(len(rois)), [pm.ROI_LAB[r] for r in rois])
        ax.set_xlim(-.6, len(rois) - .4)
        ax.set_ylim(0, 1)
    axes[0].set_title('Orientation', loc='left')
    axes[0].set_ylabel('Resultant length of error')
    axes[1].set_title('Value (CHF)', loc='left')
    axes[1].set_ylabel('r(decoded, true value)')
    handles = [Line2D([], [], ls='', marker=mk, ms=4, color=col, mfc=col if f else 'white',
                      mew=1, label=lab) for _, _, lab, col, mk, f in pm.CELLS]
    axes[1].legend(handles=handles, loc='upper right', ncol=3, frameon=False, fontsize=6.5,
                   handletextpad=.3, columnspacing=1.0)
    top.suptitle('Abstract values · orientation-to-CHF value, 36 participants', x=.01,
                 ha='left', fontsize=8, fontweight='bold')

    bx = bot.subplots(1, 4, sharey=False)
    for j, (rg, lab) in enumerate((('narrow', 'Narrow range (10–25)'),
                                   ('wide', 'Wide range (10–40)'))):
        d = npd[npd.range == rg]
        np_panel(bx[j], d, 'r_paper', rng)
        bx[j].set_ylim(-.2, .7)
        bx[j].set_title(lab, loc='left')
        np_panel(bx[j + 2], d, 'mae_paper', rng)
        bx[j + 2].set_ylim(2, 12)
        bx[j + 2].set_title(lab, loc='left')
    bx[0].set_ylabel('r(decoded, response)')
    bx[2].set_ylabel('Mean absolute error (items)')
    for i, (noise, col) in enumerate(NP_NOISE):
        bx[0].text(.98, .96 - i * .075, {'spherical': 'Spherical', 'jehee': 'Jehee',
                                         'lowrank': 'Low-rank'}[noise],
                   color=col, transform=bx[0].transAxes, ha='right', va='top', fontsize=6.5)
    bot.suptitle('Neural priors · numerosity estimation, 39 participants', x=.01, ha='left',
                 fontsize=8, fontweight='bold')
    for ax in list(axes) + list(bx):
        ax.spines['left'].set_position(('outward', 4))
        ax.spines['bottom'].set_position(('outward', 4))
    out = pm.RES / 'fig_decoding_two_studies.pdf'
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    print(out)


if __name__ == '__main__':
    main()
