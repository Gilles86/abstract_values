#!/usr/bin/env python3
"""Figures for the status report (notes/report_2026-09-17.md).

  report_fig1_behaviour.pdf     Task mappings, value density, bid noise per value
  report_fig2_tuning.pdf        Tuning reliability; value vs orientation comparisons
  report_fig3_precision.pdf     Precision per condition, neural and behavioural,
                                against the density prediction
  report_fig5_brain_behaviour.pdf  Between-subject brain-behaviour relations

Figure 4 of the report is notes/figures/cf_coupling_maps.pdf (plot_coupling_maps.py).

All inputs are local summary TSVs plus the local behavioural logs.

  python -m abstract_values.visualize.report_figures
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.behavior.data import get_all_behavioral_data
from abstract_values.connective_fields.gates import load_mappings
from abstract_values.connective_fields.plot_gates import letter  # also sets rcParams
from abstract_values.utils.data import BIDS_FOLDER

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / 'notes' / 'figures'
DATA = REPO / 'notes' / 'data'
CF = BIDS_FOLDER / 'derivatives' / 'connective_fields'
COND_COL = {'cdf': '#3B5BA5', 'inverse_cdf': '#5D8C3F'}
COND_LAB = {'cdf': 'CDF', 'inverse_cdf': 'Inverse CDF'}
V1_COL, NPC_COL = '#3B5BA5', '#C44E52'
GRID = np.arange(6, 38.5, .5)


def despine(axes):
    for ax in np.ravel(axes):
        sns.despine(ax=ax, offset=4, trim=True)


def dots(ax, x, v, col, jitter=.08, rng=None, ms=5):
    rng = rng or np.random.default_rng(0)
    v = pd.Series(v).dropna()
    ax.scatter(x + rng.uniform(-jitter, jitter, len(v)), v, s=6, color=col, alpha=.4, lw=0)
    ax.errorbar(x, v.mean(), v.sem(), color='.15', lw=.9, zorder=3)
    ax.plot(x, v.mean(), 'D', ms=ms, mfc=col, mec='.15', mew=1, zorder=4)


def presented_values(maps, ori):
    keep = (ori > 0) & (ori < 180)
    return {c: maps[c][keep] for c in maps}


def density(values, grid=GRID, bw=2.):
    k = np.exp(-.5 * ((grid[:, None] - values[None, :]) / bw) ** 2).sum(1)
    return k / (k.sum() * (grid[1] - grid[0]))


def behaviour():
    df = get_all_behavioral_data()
    df = df[df['event_type'] == 'feedback'].copy()
    df['response'] = pd.to_numeric(df['response'], errors='coerce')
    df['error'] = df['response'] - df['value']
    df = df.reset_index().dropna(subset=['error'])
    return (df.groupby(['subject', 'mapping', 'value']).error.std()
              .rename('sd').reset_index())


def diff_profile_r(tab, xcol, ycol, dens_diff, keep=None):
    """Per subject r between log-precision difference (cdf - inverse) and density difference.

    ``keep``: boolean mask over GRID (e.g. to leave out the 22 CHF anchor).
    """
    keep = np.ones(len(GRID), bool) if keep is None else keep
    rs = {}
    for s, g in tab.groupby('subject'):
        prof = {}
        for c in ('cdf', 'inverse_cdf'):
            h = g[g.mapping == c].sort_values(xcol)
            if len(h) < 5:
                break
            prof[c] = np.interp(GRID, h[xcol], np.log(1 / h[ycol]))
        if len(prof) == 2:
            rs[s] = np.corrcoef((prof['cdf'] - prof['inverse_cdf'])[keep], dens_diff[keep])[0, 1]
    return pd.Series(rs)


# ── figure 1 ──────────────────────────────────────────────────────────────────

def fig1(maps, ori, beh):
    pv = presented_values(maps, ori)
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.2), constrained_layout=True)
    ax = axes[0]
    for c in maps:
        ax.plot(ori, maps[c], color=COND_COL[c], lw=1.4, marker='o', ms=2)
    ax.text(150, 12, 'CDF', color=COND_COL['cdf'])
    ax.text(40, 30, 'Inverse CDF', color=COND_COL['inverse_cdf'], ha='right')
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_yticks([2, 22, 42])
    ax.set_xlabel('Gabor orientation (°)')
    ax.set_ylabel('Value (CHF)')
    letter(ax, 'a')

    ax = axes[1]
    for c in maps:
        ax.plot(GRID, density(pv[c]), color=COND_COL[c], lw=1.4)
        ax.plot(pv[c], np.full(len(pv[c]), -.004 if c == 'cdf' else -.009), '|',
                color=COND_COL[c], ms=4)
    ax.set_xticks([6, 14, 22, 30, 38])
    ax.set_xlabel('Value (CHF)')
    ax.set_ylabel('Value density')
    letter(ax, 'b')

    ax = axes[2]
    for c in maps:
        sns.lineplot(data=beh[beh.mapping == c], x='value', y='sd', errorbar=('se', 1),
                     color=COND_COL[c], ax=ax, lw=1.4, err_kws={'lw': 0, 'alpha': .2})
    ax.set_xticks([2, 12, 22, 32, 42])
    ax.set_xlabel('Value (CHF)')
    ax.set_ylabel('Bid SD (CHF)')
    ax.text(.03, .97, f'n = {beh.subject.nunique()}', transform=ax.transAxes, va='top',
            color='.3')
    letter(ax, 'c')
    despine(axes)
    fig.savefig(FIG / 'report_fig1_behaviour.pdf')


# ── figure 2 ──────────────────────────────────────────────────────────────────

def fig2():
    rng = np.random.default_rng(1)
    tun = pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str})
                     for f in sorted((CF / 'gates').glob('sub-*/sub-*_desc-tuningreliability.tsv'))])
    tun = tun[tun.selection == 'selected'].copy()
    tun['split'] = tun['split'].str.replace(r'within-ses\d', 'within', regex=True)
    tun = tun.groupby(['subject', 'roi', 'split'], as_index=False).reliability.mean()
    twin = pd.read_csv(DATA / 'value_vs_orientation_twin_winner.tsv', sep='\t', dtype={'subject': str})
    gen = pd.read_csv(DATA / 'condition_generalisation.tsv', sep='\t', dtype={'subject': str})
    gen = gen.groupby(['roi', 'subject']).r_advantage.mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.3), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.5, 1, 1]})
    ax = axes[0]
    order = ['within', 'across-ses']
    for i, sp in enumerate(order):
        dots(ax, i - .17, tun[(tun.roi == 'V1') & (tun.split == sp)].reliability, V1_COL, rng=rng)
        dots(ax, i + .17, tun[(tun.roi == 'NPC') & (tun.split == sp)].reliability, NPC_COL, rng=rng)
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks([0, 1], ['Within session\n(odd vs even runs)', 'Across sessions\n(the two mappings)'])
    ax.set_ylabel('Split-half reliability (r)')
    ax.text(.03, .98, 'V1 preferred orientation', color=V1_COL, transform=ax.transAxes, va='top')
    ax.text(.03, .89, 'NPCr preferred value', color=NPC_COL, transform=ax.transAxes, va='top')
    letter(ax, 'a')

    ax = axes[1]
    w = twin.pivot(index='subject', columns='roi', values='frac_value')
    dots(ax, 0, w['V1'], V1_COL, rng=rng)
    dots(ax, 1, w['NPCr'], NPC_COL, rng=rng)
    ax.axhline(.5, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks([0, 1], ['V1', 'NPCr'])
    ax.set_xlim(-.6, 1.6)
    ax.set_ylabel('Voxels where value model wins')
    p = stats.ttest_rel(w['NPCr'], w['V1'], nan_policy='omit').pvalue
    ax.text(.5, .98, f'NPCr vs V1: p = {p:.2f}', transform=ax.transAxes, ha='center',
            va='top', color='.3', fontsize=6.5)
    letter(ax, 'b')

    ax = axes[2]
    g = gen.pivot(index='subject', columns='roi', values='r_advantage')
    dots(ax, 0, g['BensonV1'], V1_COL, rng=rng)
    dots(ax, 1, g['NPCr'], NPC_COL, rng=rng)
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks([0, 1], ['V1', 'NPCr'])
    ax.set_xlim(-.6, 1.6)
    ax.set_ylabel('Cross-condition prediction,\nvalue − orientation (Δr)')
    p = stats.ttest_rel(g['NPCr'], g['BensonV1'], nan_policy='omit').pvalue
    ax.text(.5, .98, f'NPCr vs V1: p = {p:.2f}', transform=ax.transAxes, ha='center',
            va='top', color='.3', fontsize=6.5)
    letter(ax, 'c')
    despine(axes)
    fig.savefig(FIG / 'report_fig2_tuning.pdf')
    return w, g


# ── figure 3 ──────────────────────────────────────────────────────────────────

def fig3(maps, ori, beh):
    pv = presented_values(maps, ori)
    dens = {c: density(pv[c]) for c in maps}
    ddiff = np.log(dens['cdf']) - np.log(dens['inverse_cdf'])

    # Spherical noise + 200-point dense decoding grid: the full empirical noise
    # covariance and the presented-value grid together produced a much stronger
    # anti-density effect (see eu_density_flatgrid.py).
    eu = pd.read_csv(DATA / 'expected_uncertainty_per_condition_spherical_dense.tsv',
                     sep='\t', dtype={'subject': str}).rename(columns={'condition': 'mapping'})
    eu = eu.groupby(['subject', 'mapping', 'value'], as_index=False).sd_E.mean()
    r_neu = diff_profile_r(eu, 'value', 'sd_E', ddiff)
    beh_pos = beh[beh.sd > 0]
    r_beh = diff_profile_r(beh_pos, 'value', 'sd', ddiff)
    off22 = np.abs(GRID - 22) >= 4
    for name, tab, y in (('neural', eu, 'sd_E'), ('behaviour', beh_pos, 'sd')):
        r = diff_profile_r(tab, 'value', y, ddiff, off22)
        t = stats.ttest_1samp(r, 0)
        print(f'{name} precision-diff ~ density-diff, 18-26 CHF excluded: r = {r.mean():.3f}, '
              f't = {t.statistic:.2f}, p = {t.pvalue:.2g}')

    fig, axes = plt.subplots(1, 4, figsize=(7.25, 2.2), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1, 1, 1, .7]})
    ax = axes[0]
    ax.plot(GRID, ddiff, color='.25', lw=1.4)
    ax.axhline(0, color='.7', lw=.6, ls='--')
    ax.set_xticks([6, 14, 22, 30, 38])
    ax.set_xlabel('Value (CHF)')
    ax.set_ylabel('log density, CDF − inverse')
    ax.text(.03, .98, 'Predicted precision\ndifference', transform=ax.transAxes, va='top',
            color='.25', fontsize=6.5)
    letter(ax, 'a')

    ax = axes[1]
    for c in maps:
        sns.lineplot(data=eu[eu.mapping == c], x='value', y='sd_E', errorbar=('se', 1),
                     color=COND_COL[c], ax=ax, lw=1.4, err_kws={'lw': 0, 'alpha': .2})
    ax.set_xticks([2, 12, 22, 32, 42])
    ax.set_xlabel('Value (CHF)')
    ax.set_ylabel('NPCr expected SD (CHF)')
    ax.text(.5, .98, 'CDF', color=COND_COL['cdf'], transform=ax.transAxes, va='top', ha='center')
    ax.text(.5, .89, 'Inverse CDF', color=COND_COL['inverse_cdf'], transform=ax.transAxes,
            va='top', ha='center')
    letter(ax, 'b')

    ax = axes[2]
    for c in maps:
        sns.lineplot(data=beh[beh.mapping == c], x='value', y='sd', errorbar=('se', 1),
                     color=COND_COL[c], ax=ax, lw=1.4, err_kws={'lw': 0, 'alpha': .2})
    ax.set_xticks([2, 12, 22, 32, 42])
    ax.set_xlabel('Value (CHF)')
    ax.set_ylabel('Bid SD (CHF)')
    letter(ax, 'c')

    ax = axes[3]
    rng = np.random.default_rng(3)
    dots(ax, 0, r_neu, NPC_COL, rng=rng)
    dots(ax, 1, r_beh, '.35', rng=rng)
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks([0, 1], ['NPCr', 'Bids'])
    ax.set_xlim(-.6, 1.6)
    ax.set_ylabel('r(precision diff., a)')
    for x, r in ((0, r_neu), (1, r_beh)):
        p = stats.ttest_1samp(r, 0).pvalue
        ax.text(x, 1.02, f'p = {p:.1g}'.replace('0.', '.'), ha='center', va='bottom',
                fontsize=6, color='.3')
    ax.set_ylim(-1, 1.1)
    letter(ax, 'd')
    despine(axes)
    fig.savefig(FIG / 'report_fig3_precision.pdf')
    return r_neu, r_beh


# ── figure 5 ──────────────────────────────────────────────────────────────────

def fig5(beh):
    summ = pd.read_csv(DATA / 'brain_behavior_subject_summary.tsv', sep='\t')
    corr = pd.concat([
        pd.read_csv(DATA / 'brain_behavior_correlations_seq_expected.tsv', sep='\t').assign(measure='Expected SD'),
        pd.read_csv(DATA / 'brain_behavior_correlations_seq_extent.tsv', sep='\t').assign(measure='Tuned extent')])

    s = pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str}) for f in
                   sorted((CF / 'coupling_iem-k24-kappa16').glob('sub-*/sub-*_desc-scores.tsv'))])
    s = s[s.direction == 'npc_from_v1']
    obs = s[s.kind == 'observed'].set_index('subject').score
    coup = obs - s[s.kind == 'label-shuffle'].groupby('subject').score.mean().reindex(obs.index)
    coup.index = [int(''.join(ch for ch in k if ch.isdigit())) for k in coup.index]
    bsd = beh.groupby('subject').sd.mean()
    both = pd.concat([coup.rename('coupling'), bsd.rename('bid_sd')], axis=1).dropna()

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.4), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1, 1.35, 1]})
    ax = axes[0]
    ax.scatter(summ.log_tuned_vonmises, summ.behav_sd, s=10, color='.3', lw=0)
    b1, b0 = np.polyfit(summ.log_tuned_vonmises, summ.behav_sd, 1)
    xx = np.array([summ.log_tuned_vonmises.min(), summ.log_tuned_vonmises.max()])
    ax.plot(xx, b0 + b1 * xx, color=NPC_COL, lw=1.2)
    r, p = stats.pearsonr(summ.log_tuned_vonmises, summ.behav_sd)
    ax.text(.97, .97, f'r = {r:.2f}, p = {p:.1g}'.replace('0.0', '.0'), transform=ax.transAxes,
            ha='right', va='top', color='.2', fontsize=6.5)
    ax.set_xlabel('Tuned vertices, whole cortex (log₁₀)')
    ax.set_ylabel('Bid error SD (CHF)')
    letter(ax, 'a')

    ax = axes[1]
    labels = []
    for i, (_, row) in enumerate(corr.iterrows()):
        col = V1_COL if row.matched else '.6'
        ax.plot([row.lo, row.hi], [i, i], color=col, lw=1.2)
        ax.plot(row.rho, i, 'o', color=col, ms=4)
        labels.append(f'{row.behaviour.split(" (")[0]} ~ {row.neural.replace(" (cross)", "")}'
                      f'  [{row.measure}]')
    ax.axvline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_yticks(range(len(labels)), labels, fontsize=5.5)
    ax.set_xlabel('Spearman ρ (95% interval)')
    ax.text(-.6, -.9, 'Matched', color=V1_COL, fontsize=6.5)
    ax.text(-.6, -.35, 'Crossed', color='.6', fontsize=6.5)
    ax.set_ylim(len(labels) - .5, -1.3)
    letter(ax, 'b')

    ax = axes[2]
    ax.scatter(both.coupling, both.bid_sd, s=10, color='.3', lw=0)
    rho, p = stats.spearmanr(both.coupling, both.bid_sd)
    ax.text(.97, .97, f'ρ = {rho:.2f}, p = {p:.2f}', transform=ax.transAxes, ha='right',
            va='top', color='.2', fontsize=6.5)
    ax.axvline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xlabel('Mapping score (Δr)')
    ax.set_ylabel('Mean bid SD (CHF)')
    letter(ax, 'c')
    despine([axes[0], axes[2]])
    sns.despine(ax=axes[1], offset=4, trim=False, left=True)
    fig.savefig(FIG / 'report_fig5_brain_behaviour.pdf')
    return summ, both, rho, p


def main():
    ori, maps = load_mappings()
    beh = behaviour()
    fig1(maps, ori, beh)
    w, g = fig2()
    r_neu, r_beh = fig3(maps, ori, beh)
    summ, both, rho, p = fig5(beh)
    print('twin winner frac value:', w.mean().round(3).to_dict())
    print('generalisation adv:', g.mean().round(4).to_dict())
    for name, r in (('neural', r_neu), ('behaviour', r_beh)):
        t = stats.ttest_1samp(r, 0)
        print(f'{name} precision-diff ~ density-diff: r = {r.mean():.3f} ± {r.sem():.3f}, '
              f't{len(r)-1} = {t.statistic:.2f}, p = {t.pvalue:.2g}')
    print(f'coupling ~ bid SD: rho = {rho:.3f}, p = {p:.2g}, n = {len(both)}')


if __name__ == '__main__':
    main()
