#!/usr/bin/env python3
"""Mean decoding accuracy per ROI: spherical vs van Bergen & Jehee vs low-rank noise,
at 50 / 250 voxels and at many voxels (whole ROI for spherical and Jehee, the
cross-validated N for low-rank, which picks ~ the whole ROI).

Orientation 'r' is the resultant length of the doubled-angle error (analyze.py);
value 'r' is Pearson r(decoded posterior mean, true value). Mean over
participants +/- 95% CI (t), participants as faint dots.

    python plot_mean_r.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 8,
    'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
RES = Path(__file__).parent / 'results'
SPH, JEH, LR = '#8C8C8C', '#C44E52', '#3B5BA5'
# (noise, n_voxels) -> label, colour, marker, filled; open = 50, filled = 250,
# diamond = whole ROI (low-rank: cross-validated N)
CELLS = [('spherical', '50', 'Spherical · 50', SPH, 'o', False),
         ('jehee', '50', 'Jehee · 50', JEH, 'o', False),
         ('lowrankcv', '50', 'Low-rank · 50', LR, 'o', False),
         ('spherical', '250', 'Spherical · 250', SPH, 'o', True),
         ('jehee', '250', 'Jehee · 250', JEH, 'o', True),
         ('lowrankcv', '250', 'Low-rank · 250', LR, 'o', True),
         ('spherical', 'all', 'Spherical · whole ROI', SPH, 'D', True),
         ('jehee', 'all', 'Jehee · whole ROI', JEH, 'D', True),
         ('lowrankCVN', 'cv', 'Low-rank · CV voxels', LR, 'D', True)]
ROIS = {'orientation': ['BensonV1', 'BensonV2', 'BensonV3'],
        'value': ['BensonV1', 'IPS0', 'IPS12', 'IPS345', 'NPCr']}
ROI_LAB = {'BensonV1': 'V1', 'BensonV2': 'V2', 'BensonV3': 'V3', 'IPS0': 'IPS0',
           'IPS12': 'IPS1–2', 'IPS345': 'IPS3–5', 'NPCr': 'NPCr'}
MIN_N = 30          # IPS1–2 at 250 voxels ran for 8 participants only: drop


def load():
    d = pd.read_csv(RES / 'lowrank_metrics_by_subject.tsv', sep='\t',
                    dtype={'subject': str, 'n_voxels': str})
    c = pd.read_csv(RES / 'nvcv_metrics_by_subject.tsv', sep='\t',
                    dtype={'subject': str, 'n_voxels': str})
    frames = [d, c[c.n_voxels == 'cv'].assign(noise='lowrankCVN')]
    j = RES / 'jehee_metrics_by_subject.tsv'      # jehee_metrics.py
    if j.exists():
        j = pd.read_csv(j, sep='\t', dtype={'subject': str, 'n_voxels': str})
        # spherical at 50 / 250 is already in the low-rank run: keep that one
        frames.append(j[(j.noise == 'jehee') | (j.n_voxels == 'all')])
    return pd.concat(frames)


def main():
    df = load()
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.9), constrained_layout=True,
                             gridspec_kw={'width_ratios': [3, 5]})
    off = np.linspace(-.38, .38, len(CELLS))
    rng = np.random.default_rng(0)
    for ax, (q, rois) in zip(axes, ROIS.items()):
        for i, roi in enumerate(rois):
            for o, (noise, nv, lab, col, mk, filled) in zip(off, CELLS):
                r = df[(df.quantity == q) & (df.roi == roi) & (df.noise == noise)
                       & (df.n_voxels == nv)].r.dropna()
                if len(r) < MIN_N:
                    continue
                x = i + o
                ax.scatter(x + rng.uniform(-.025, .025, len(r)), r, s=2, color=col,
                           alpha=.25, lw=0, zorder=1)
                h = stats.t.ppf(.975, len(r) - 1) * r.sem()
                ax.plot([x, x], [r.mean() - h, r.mean() + h], color=col, lw=1.1, zorder=2)
                ax.plot(x, r.mean(), mk, ms=4 if mk == 'o' else 3.6, color=col, zorder=3,
                        mfc=col if filled else 'white', mew=1)
        ax.set_xticks(range(len(rois)), [ROI_LAB[r] for r in rois])
        ax.set_xlim(-.6, len(rois) - .4)
        ax.set_ylim(0, 1)
        ax.spines['left'].set_position(('outward', 4))
        ax.spines['bottom'].set_position(('outward', 4))
    axes[0].set_title('Orientation decoding', loc='left')
    axes[0].set_ylabel('Resultant length of error')
    axes[1].set_title('Value decoding', loc='left')
    axes[1].set_ylabel('r(decoded, true value)')
    handles = [Line2D([], [], ls='', marker=mk, ms=4, color=col,
                      mfc=col if f else 'white', mew=1, label=lab)
               for _, _, lab, col, mk, f in CELLS]
    axes[1].legend(handles=handles, loc='upper right', ncol=3, frameon=False,
                   fontsize=6.5, handletextpad=.3, columnspacing=1.0)
    out = RES / 'fig_mean_decoding_r.pdf'
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    print(out)


if __name__ == '__main__':
    main()
