#!/usr/bin/env python3
"""One figure that explains and shows the mapping-coupling result.

Top row, the logic (computed from the task mappings, no data):
  a  The two orientation->value mappings; one value sits at two orientations.
  b  An NPCr voxel's value tuning.
  c  Read through each mapping, that tuning predicts coupling to different V1
     orientation populations in the two sessions.
  d  The predicted orientation shift as a function of preferred value.
Bottom row, the data (n = 30):
  e  Coupling profile aligned to the orientation worth the voxel's preferred
     value -- in this session vs in the other session (argmax V1 bins).
  f  Same with V1 channels from the inverted vonmises encoding model.
  g  Mapping score (observed - label-shuffled) per subject, for both V1
     projections, without and with the neighbouring trials removed.

  python -m abstract_values.connective_fields.plot_coupling_explainer
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.connective_fields.gates import load_mappings, lognormal_mode_fwhm
from abstract_values.connective_fields.plot_gates import letter, read  # sets rcParams
from abstract_values.utils.data import BIDS_FOLDER

REPO = Path(__file__).resolve().parents[2]
ROOT = BIDS_FOLDER / 'derivatives' / 'connective_fields'
MAP_COL = {'cdf': '#3B5BA5', 'inverse_cdf': '#5D8C3F'}
MAP_LAB = {'cdf': 'CDF mapping', 'inverse_cdf': 'Inverse-CDF mapping'}
CUR_COL, OTH_COL = '#C44E52', '#8C8C8C'
EX_MODE, EX_FWHM = 16., 8.


def wrapped_profile(root, lags, step=None):
    """Subject-mean coupling per offset and reference, closed at +-90 deg.

    ``step`` pools offsets into coarser bins (voxel-count weighted): on the 5-deg
    encoding-model grid, theta* falls between grid points and single offsets
    alternate in which voxels they collect.
    """
    a = read(root, 'aligned')
    a = a[a.lags == lags].copy()
    if step:
        a['offset'] = (np.round(a.offset / step) * step + 0.).where(lambda o: o != 90, -90.)
    a['w'] = a.cf * a.n
    a = a.groupby(['subject', 'reference', 'offset'], as_index=False)[['w', 'n']].sum()
    a['cf'] = a.w / a.n
    edge = a[a.offset == -90].assign(offset=90.)
    return pd.concat([a, edge], ignore_index=True)


def profile_panel(ax, root, lags, title, step=None):
    a = wrapped_profile(root, lags, step)
    for ref, col, z in (('other', OTH_COL, 2), ('current', CUR_COL, 3)):
        sns.lineplot(data=a[a.reference == ref], x='offset', y='cf', errorbar=('se', 1),
                     color=col, ax=ax, lw=1.4, zorder=z, err_kws={'lw': 0, 'alpha': .2})
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.axvline(0, color='.85', lw=.6, zorder=0)
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_xlabel('V1 orientation relative to θ* (°)')
    ax.set_ylabel('Coupling (r, channel-centred)')
    ax.text(.03, .98, title, transform=ax.transAxes, va='top', color='.3')


def main(out):
    ori, maps = load_mappings()
    fig = plt.figure(figsize=(7.25, 4.7), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.08], hspace=.06)
    ax_a, ax_b, ax_c, ax_d = top.subplots(1, 4, gridspec_kw={'width_ratios': [1, .8, 1, 1]})
    ax_e, ax_f, ax_g = bottom.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1.15]})

    # a: mappings
    theta_ex = {c: np.interp(EX_MODE, maps[c], ori) for c in maps}
    for c in maps:
        ax_a.plot(ori, maps[c], color=MAP_COL[c], lw=1.4)
        ax_a.plot([theta_ex[c]] * 2, [2, EX_MODE], color=MAP_COL[c], lw=.8, ls=':')
    ax_a.plot([0, max(theta_ex.values())], [EX_MODE] * 2, color='.4', lw=.8, ls=':')
    ax_a.text(np.mean(list(theta_ex.values())), 4, 'θ*', ha='center', color='.3', fontsize=7)
    ax_a.text(3, EX_MODE + 1.2, f'{EX_MODE:g} CHF', color='.3', fontsize=6.5)
    ax_a.text(100, 12, 'CDF', color=MAP_COL['cdf'], fontsize=7)
    ax_a.text(62, 30, 'Inverse CDF', color=MAP_COL['inverse_cdf'], fontsize=7, ha='right')
    ax_a.set_xticks([0, 45, 90, 135, 180])
    ax_a.set_yticks([2, 22, 42])
    ax_a.set_xlabel('Gabor orientation (°)')
    ax_a.set_ylabel('Value (CHF)')
    letter(ax_a, 'a')

    # b: an NPC voxel's value tuning
    v = np.linspace(2, 42, 300)
    ax_b.plot(v, lognormal_mode_fwhm(v, EX_MODE, EX_FWHM), color=CUR_COL, lw=1.4)
    ax_b.set_xticks([2, 16, 42])
    ax_b.set_yticks([])
    ax_b.set_xlabel('Value (CHF)')
    ax_b.set_ylabel('NPC voxel response')
    ax_b.text(.97, .97, 'Prefers\n16 CHF', transform=ax_b.transAxes, ha='right', va='top',
              color=CUR_COL, fontsize=6.5)
    letter(ax_b, 'b')

    # c: predicted coupling across V1 orientation, per mapping
    th = np.linspace(0, 180, 361)
    for c in maps:
        ax_c.plot(th, lognormal_mode_fwhm(np.interp(th, ori, maps[c]), EX_MODE, EX_FWHM),
                  color=MAP_COL[c], lw=1.4)
        ax_c.text(theta_ex[c] + (5 if c == 'cdf' else -5), 1.04,
                  ('CDF\nθ* = ' if c == 'cdf' else 'Inverse CDF\nθ* = ') + f'{theta_ex[c]:.0f}°',
                  color=MAP_COL[c], ha='left' if c == 'cdf' else 'right', fontsize=6.5)
    ax_c.set_ylim(0, 1.35)
    ax_c.set_xticks([0, 45, 90, 135, 180])
    ax_c.set_yticks([])
    ax_c.set_xlabel('V1 preferred orientation (°)')
    ax_c.set_ylabel('Predicted coupling')
    letter(ax_c, 'c')

    # d: predicted shift over preferred value
    vv = np.linspace(2, 42, 400)
    shift = np.interp(vv, maps['cdf'], ori) - np.interp(vv, maps['inverse_cdf'], ori)
    ax_d.fill_between(vv, 0, shift, color='.85', lw=0)
    ax_d.plot(vv, shift, color='.25', lw=1.2)
    ax_d.plot(EX_MODE, theta_ex['cdf'] - theta_ex['inverse_cdf'], 'o', color=CUR_COL, ms=3.5)
    ax_d.axhline(0, color='.6', lw=.6)
    ax_d.set_xticks([2, 12, 22, 32, 42])
    ax_d.set_yticks([-20, 0, 20])
    ax_d.set_xlabel('Preferred value (CHF)')
    ax_d.set_ylabel('Predicted shift of θ* (°)')
    letter(ax_d, 'd')

    # e, f: aligned coupling profiles
    profile_panel(ax_e, ROOT / 'profiles', 0, 'Binned V1 voxels')
    profile_panel(ax_f, ROOT / 'profiles_iem', 0, 'Inverted encoding model', step=15)
    lo = min(ax_e.get_ylim()[0], ax_f.get_ylim()[0])
    hi = max(ax_e.get_ylim()[1], ax_f.get_ylim()[1])
    for ax in (ax_e, ax_f):
        ax.set_ylim(lo, hi + .45 * (hi - lo))
    for ax in (ax_e, ax_f):
        ax.text(.03, .88, 'θ* in this session', transform=ax.transAxes, va='top',
                color=CUR_COL, fontsize=6.5)
        ax.text(.03, .79, 'θ* in the other session', transform=ax.transAxes, va='top',
                color=OTH_COL, fontsize=6.5)
    ax_f.set_ylabel('')
    letter(ax_e, 'e')
    letter(ax_f, 'f')

    # g: subject scores
    rng = np.random.default_rng(2)
    cells = [('coupling', 0, 0), ('coupling_lags-1', 0, 1),
             ('coupling_iem', 1, 0), ('coupling_lags-1_iem', 1, 1)]
    for variant, x, lag in cells:
        s = read(ROOT / variant, 'scores')
        s = s[s.direction == 'npc_from_v1']
        obs = s[s.kind == 'observed'].set_index('subject').score
        d = obs - s[s.kind == 'label-shuffle'].groupby('subject').score.mean().reindex(obs.index)
        xx = x + (lag - .5) * .38
        col = CUR_COL if lag == 0 else '#E3A2A4'
        ax_g.scatter(xx + rng.uniform(-.06, .06, len(d)), d, s=6, color=col, alpha=.45, lw=0)
        ax_g.errorbar(xx, d.mean(), d.sem(), color='.15', lw=.9, zorder=3)
        ax_g.plot(xx, d.mean(), 'D', ms=5, mfc=col, mec='.15', mew=1, zorder=4)
        p = stats.ttest_1samp(d, 0).pvalue
        ax_g.text(xx, -.04, 'Stimulus\nremoved' if lag == 0 else '+ Neigh-\nbours', ha='center',
                  va='center', fontsize=5.5, color=CUR_COL if lag == 0 else '#D98184')
        ax_g.text(xx, .062, f'p = {p:.1g}'.replace('0.', '.'), ha='center', fontsize=6,
                  color='.3')
    ax_g.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax_g.set_xticks([0, 1], ['Binned\nV1 voxels', 'Inverted\nencoding model'])
    ax_g.set_ylabel('Mapping score (Δr)')
    ax_g.set_yticks([-.02, 0, .02, .04, .06])
    ax_g.set_ylim(-.05, .07)
    ax_g.set_xlim(-.5, 1.5)

    letter(ax_g, 'g')

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g):
        sns.despine(ax=ax, offset=4, trim=True)
    for ax in (ax_b, ax_c):
        ax.spines['left'].set_visible(False)
    fig.savefig(out)
    print(f'saved {out}')


if __name__ == '__main__':
    main(REPO / 'notes' / 'figures' / 'cf_coupling_explainer.pdf')
