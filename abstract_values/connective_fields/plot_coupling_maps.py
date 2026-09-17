#!/usr/bin/env python3
"""Hypothesis vs data: NPCr preferred value x V1 orientation coupling maps.

Row 1, PREDICTION (no brain data beyond each voxel's value tuning): if an NPCr
voxel couples to the V1 orientations currently worth its preferred value, its
coupling over V1 orientation is its value tuning read through the session's
mapping, f(m_c(theta)). Averaged per preferred-value bin:
  a  CDF sessions   b  inverse-CDF sessions   c  a - b
Row 2, DATA (n = 30): the same maps from trial-to-trial residual coupling,
V1 channels from the inverted encoding model (24 basis functions, kappa 16).
In both rows the NPCr-wide profile of each session is removed per orientation.
Data maps are smoothed for display (Gaussian, 1 bin); statistics use raw maps.
  d, e, f  as a, b, c
Row 3:
  g  Difference map (f) read out along the two theta* curves. Prediction:
     positive along theta* CDF, negative along theta* inverse CDF, both zero
     where the curves cross.
  h  Mapping score (observed - label-shuffled) with the gain controls.

  python -m abstract_values.connective_fields.plot_coupling_maps
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter

from abstract_values.connective_fields.gates import load_mappings, lognormal_mode_fwhm, zscore
from abstract_values.connective_fields.plot_coupling_explainer import CUR_COL, MAP_COL, ROOT
from abstract_values.connective_fields.plot_gates import letter, read
from abstract_values.connective_fields.test_coupling import IEM_GRID

REPO = Path(__file__).resolve().parents[2]
CONDS = ('cdf', 'inverse_cdf')
TITLE = {'cdf': 'CDF sessions', 'inverse_cdf': 'Inverse-CDF sessions'}


def wrap(d):
    return (d + 90) % 180 - 90


def load(variant):
    mp = pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str})
                    for f in sorted((ROOT / variant).glob('sub-*/sub-*_desc-maps.tsv'))],
                   ignore_index=True)
    pk = pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str})
                    for f in sorted((ROOT / variant).glob('sub-*/sub-*_desc-peaks.tsv.gz'))],
                   ignore_index=True)
    return mp, pk


def observed_maps(mp):
    """{cond: subject x orientation x value-bin array}, plus axes."""
    w = mp.pivot_table(index=['condition', 'subject', 'orientation'], columns='value_bin',
                       values='cf')
    subs = sorted(mp.subject.unique())
    out = {c: np.stack([w.xs((c, s)).to_numpy() for s in subs]) for c in CONDS}
    return out, subs, w.columns.to_numpy()


def predicted_maps(pk, subs, vbins, maps, ori):
    """Same layout, from each voxel's value tuning, NPCr-wide profile removed."""
    pk = pk.assign(value_bin=np.clip(np.floor((pk['mode'] - 2) / 2) * 2 + 3, 3, 41))
    out = {c: np.full((len(subs), len(IEM_GRID), len(vbins)), np.nan) for c in CONDS}
    for i, s in enumerate(subs):
        g = pk[pk.subject == s]
        for c in CONDS:
            p = lognormal_mode_fwhm(np.interp(IEM_GRID, ori, maps[c])[:, None],
                                    g['mode'].to_numpy()[None, :], g['fwhm'].to_numpy()[None, :])
            p = zscore(p - p.mean(0), axis=0)
            p = p - p.mean(1, keepdims=True)
            for j, b in enumerate(vbins):
                m = (g.value_bin == b).to_numpy()
                if m.any():
                    out[c][i, :, j] = p[:, m].mean(1)
    return out


def show(ax, m, vbins, vmax, cmap, smooth=False):
    if smooth:
        m = gaussian_filter(np.nan_to_num(m), 1, mode=('wrap', 'nearest'))
    ve = np.r_[vbins - 1, vbins[-1] + 1]
    oe = np.r_[IEM_GRID - 2.5, IEM_GRID[-1] + 2.5]
    return ax.pcolormesh(ve, oe, m, cmap=cmap, vmin=-vmax, vmax=vmax, rasterized=True)


def lines(ax, maps, ori):
    vv = np.linspace(2, 42, 400)
    for c in CONDS:
        ax.plot(vv, np.interp(vv, maps[c], ori), color=MAP_COL[c], lw=1.1)


def style_map(ax, ylabel=True):
    ax.set_xticks([2, 12, 22, 32, 42])
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlim(2, 42)
    ax.set_ylim(-2.5, 177.5)
    ax.set_xlabel('NPCr preferred value (CHF)')
    ax.set_ylabel('V1 orientation (°)' if ylabel else '')


def ridge_readout(diff, vbins, maps, ori, halfwidth=5):
    """Per subject, the difference map at theta*_c(v) (+-halfwidth deg, wrapped)."""
    rows = []
    for c in CONDS:
        th = np.interp(vbins, maps[c], ori)
        for j, (v, t) in enumerate(zip(vbins, th)):
            near = np.abs(wrap(IEM_GRID - t)) <= halfwidth
            for i, val in enumerate(diff[:, near, j].mean(1)):
                rows.append({'subject': i, 'ridge': c, 'value': v, 'delta': val})
    return pd.DataFrame(rows)


GAIN_CONTROLS = (('All trials', 'coupling_iem-k24-kappa16'),
                 ('Gain\nremoved', 'coupling_iem-k24-kappa16_gain'),
                 ('Low-drive\ntrials', 'coupling_iem-k24-kappa16_drive-low'),
                 ('High-drive\ntrials', 'coupling_iem-k24-kappa16_drive-high'))
VOXEL_CLASSES = (('Value-\ntuned', 'coupling_iem-k24-kappa16_npc-value'),
                 ('Orientation-\ntuned', 'coupling_iem-k24-kappa16_npc-orientation'))


def control_scores(variants):
    rows = []
    for label, variant in variants:
        if not (ROOT / variant).exists():
            continue
        s = read(ROOT / variant, 'scores')
        s = s[s.direction == 'npc_from_v1']
        obs = s[s.kind == 'observed'].set_index('subject').score
        d = obs - s[s.kind == 'label-shuffle'].groupby('subject').score.mean().reindex(obs.index)
        rows += [{'condition': label, 'subject': k, 'score': v} for k, v in d.items()]
    return pd.DataFrame(rows)


def map_agreement(obs, pred):
    diff_o = obs['cdf'] - obs['inverse_cdf']
    diff_p = pred['cdf'] - pred['inverse_cdf']
    r = [np.corrcoef(o[np.isfinite(p)].ravel(), p[np.isfinite(p)].ravel())[0, 1]
         for o, p in zip(diff_o, diff_p)]
    return np.array(r)


def main(out, variant='peaks'):
    ori, maps = load_mappings()
    mp, pk = load(variant)
    obs, subs, vbins = observed_maps(mp)
    pred = predicted_maps(pk, subs, vbins, maps, ori)
    r = map_agreement(obs, pred)
    t = stats.ttest_1samp(r, 0)
    print(f'{variant}: map agreement r = {r.mean():.3f} ± {r.std(ddof=1)/np.sqrt(len(r)):.3f}, '
          f't{len(r)-1} = {t.statistic:.2f}, p = {t.pvalue:.2g}')
    if (ROOT / f'{variant}_gain').exists():
        og, sg, _ = observed_maps(load(f'{variant}_gain')[0])
        rg = map_agreement(og, pred)
        tg = stats.ttest_1samp(rg, 0)
        print(f'{variant}_gain: map agreement r = {rg.mean():.3f}, t = {tg.statistic:.2f}, '
              f'p = {tg.pvalue:.2g}')

    fig = plt.figure(figsize=(7.25, 7.2), constrained_layout=True)
    rows = fig.subfigures(3, 1, height_ratios=[1, 1, 1.02], hspace=.05)
    rows[0].suptitle('Prediction  —  each voxel\'s value tuning read through the session\'s '
                     'mapping (no coupling data)', x=.01, ha='left', fontsize=8,
                     fontweight='bold', color='.25')
    rows[1].suptitle('Data  —  trial-to-trial NPCr–V1 coupling (n = 30)', x=.01, ha='left',
                     fontsize=8, fontweight='bold', color='.1')
    pa = rows[0].subplots(1, 3)
    da = rows[1].subplots(1, 3)
    ga, ha, ia = rows[2].subplots(1, 3, gridspec_kw={'width_ratios': [1.35, 1, .6]})

    # row 1: prediction
    pm = {c: np.nanmean(pred[c], 0) for c in CONDS}
    pdiff = pm['cdf'] - pm['inverse_cdf']
    vmax_p = np.nanquantile(np.abs(np.r_[pm['cdf'].ravel(), pm['inverse_cdf'].ravel()]), .99)
    for ax, c, lab in zip(pa, CONDS, 'ab'):
        im = show(ax, pm[c], vbins, vmax_p, 'PuOr_r')
        lines(ax, maps, ori)
        ax.set_title(f'Predicted: {TITLE[c]}', fontsize=7.5, color=MAP_COL[c])
        letter(ax, lab)
    rows[0].colorbar(im, ax=pa[:2], shrink=.75, pad=.01, label='Predicted coupling (z)')
    im = show(pa[2], pdiff, vbins, np.nanquantile(np.abs(pdiff), .99), 'PuOr_r')
    lines(pa[2], maps, ori)
    pa[2].set_title('Predicted: CDF − inverse CDF', fontsize=7.5)
    rows[0].colorbar(im, ax=pa[2], shrink=.75, pad=.01, label='Δ (z)')
    letter(pa[2], 'c')
    pa[0].text(40, 6, 'θ* CDF', color=MAP_COL['cdf'], ha='right', fontsize=6.5)
    pa[0].text(4, 160, 'θ* inverse CDF', color=MAP_COL['inverse_cdf'], fontsize=6.5)

    # row 2: data
    om = {c: obs[c].mean(0) for c in CONDS}
    odiff = om['cdf'] - om['inverse_cdf']
    vmax_o = np.quantile(np.abs(np.r_[gaussian_filter(om['cdf'], 1, mode=('wrap', 'nearest')).ravel(),
                                      gaussian_filter(om['inverse_cdf'], 1, mode=('wrap', 'nearest')).ravel()]), .99)
    for ax, c, lab in zip(da, CONDS, 'de'):
        im = show(ax, om[c], vbins, vmax_o, 'vlag', smooth=True)
        lines(ax, maps, ori)
        ax.set_title(f'Observed: {TITLE[c]}', fontsize=7.5, color=MAP_COL[c])
        letter(ax, lab)
    rows[1].colorbar(im, ax=da[:2], shrink=.75, pad=.01, label='Coupling (r)')
    sd = gaussian_filter(odiff, 1, mode=('wrap', 'nearest'))
    im = show(da[2], odiff, vbins, np.quantile(np.abs(sd), .99), 'vlag', smooth=True)
    lines(da[2], maps, ori)
    da[2].set_title('Observed: CDF − inverse CDF', fontsize=7.5)
    rows[1].colorbar(im, ax=da[2], shrink=.75, pad=.01, label='Δ coupling (r)')
    da[2].text(.03, .03, f'Match to c: r = {r.mean():.3f}, p = {t.pvalue:.1g}'.replace('0.0', '.0'),
               transform=da[2].transAxes, fontsize=6, color='.1',
               bbox=dict(facecolor='w', edgecolor='none', alpha=.8, pad=1))
    letter(da[2], 'f')
    for axs in (pa, da):
        for k, ax in enumerate(axs):
            style_map(ax, ylabel=k == 0)

    # g: ridge readout of the observed difference map
    rr = ridge_readout(obs['cdf'] - obs['inverse_cdf'], vbins, maps, ori)
    vv = np.linspace(2, 42, 2001)
    sh = np.interp(vv, maps['cdf'], ori) - np.interp(vv, maps['inverse_cdf'], ori)
    for x in vv[1:-1][np.diff(np.sign(sh))[:-1] != 0]:
        ga.axvline(x, color='.85', lw=3, zorder=0)
    for c in CONDS:
        sns.lineplot(data=rr[rr.ridge == c], x='value', y='delta', errorbar=('se', 1),
                     color=MAP_COL[c], ax=ga, marker='o', ms=3, err_kws={'lw': 0, 'alpha': .2})
    ga.axhline(0, color='.6', lw=.6, ls='--')
    ga.set_xticks([2, 12, 22, 32, 42])
    ga.set_xlabel('NPCr preferred value (CHF)')
    ga.set_ylabel('Δ coupling, CDF − inverse CDF (r)')
    ga.text(.02, .97, 'Along θ* CDF: predicted > 0', transform=ga.transAxes, va='top',
            fontsize=6.5, color=MAP_COL['cdf'])
    ga.text(.02, .88, 'Along θ* inverse CDF: predicted < 0', transform=ga.transAxes, va='top',
            fontsize=6.5, color=MAP_COL['inverse_cdf'])
    ga.text(.98, .03, 'Grey lines: the θ* curves cross', transform=ga.transAxes, ha='right',
            fontsize=6, color='.45')
    sep = rr.pivot_table(index='subject', columns='ridge', values='delta')
    dd = sep['cdf'] - sep['inverse_cdf']
    td = stats.ttest_1samp(dd, 0)
    print(f'ridge contrast (CDF ridge - inverse ridge, all bins): {dd.mean():.4f} ± {dd.sem():.4f}, '
          f't = {td.statistic:.2f}, p = {td.pvalue:.2g}')
    letter(ga, 'g')

    # h: gain controls, i: NPCr voxel classes
    rng = np.random.default_rng(5)
    print('condition               mean     sem      t       p')
    for ax, variants, lab in ((ha, GAIN_CONTROLS, 'h'), (ia, VOXEL_CLASSES, 'i')):
        cs = control_scores(variants)
        order = list(dict.fromkeys(cs.condition))
        for i, cnd in enumerate(order):
            v = cs[cs.condition == cnd].score
            tt = stats.ttest_1samp(v, 0)
            print(f'{cnd.replace(chr(10), " "):22s} {v.mean():+.4f} {v.sem():.4f} '
                  f'{tt.statistic:6.2f} {tt.pvalue:.2g}  (n = {len(v)})')
            col = CUR_COL if cnd in ('All trials', 'Value-\ntuned') else '.45'
            ax.scatter(i + rng.uniform(-.1, .1, len(v)), v, s=5, color=col, alpha=.4, lw=0)
            ax.errorbar(i, v.mean(), v.sem(), color='.15', lw=.9, zorder=3)
            ax.plot(i, v.mean(), 'D', ms=4.5, mfc=col, mec='.15', mew=1, zorder=4)
            ax.text(i, .085, f'p = {tt.pvalue:.1g}'.replace('0.', '.'), ha='center',
                    fontsize=5.5, color='.3')
        if lab == 'i' and len(order) == 2:
            w = cs.pivot_table(index='subject', columns='condition', values='score').dropna()
            tp = stats.ttest_rel(w[order[0]], w[order[1]])
            print(f'value - orientation voxels: {(w[order[0]] - w[order[1]]).mean():+.4f}, '
                  f't = {tp.statistic:.2f}, p = {tp.pvalue:.2g} (n = {len(w)})')
            ax.plot([0, 1], [.1, .1], color='.3', lw=.7)
            ax.text(.5, .102, f'p = {tp.pvalue:.1g}'.replace('0.', '.'), ha='center',
                    va='bottom', fontsize=5.5, color='.3')
        ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
        ax.set_xticks(range(len(order)), order, fontsize=6.5)
        ax.set_xlim(-.5, len(order) - .5)
        ax.set_ylim(-.04, .11)
        ax.set_ylabel('Mapping score (Δr)' if lab == 'h' else '')
        letter(ax, lab)
    ia.set_title('NPCr voxels', fontsize=7)

    for ax in (ga, ha, ia):
        sns.despine(ax=ax, offset=3, trim=True)

    fig.savefig(out, dpi=300)
    print(f'saved {out}')


if __name__ == '__main__':
    main(REPO / 'notes' / 'figures' / 'cf_coupling_maps.pdf')
