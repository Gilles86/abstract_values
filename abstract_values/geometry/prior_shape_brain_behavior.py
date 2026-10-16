#!/usr/bin/env python3
"""Individual differences in orientation-prior SHAPE: behaviour vs brain.

Three shape indices, each estimated the same way in behaviour and in every ROI:

    H-V        cos 2θ   horizontal vs vertical     (Fourier k=1 on the 180° circle)
    Card-obl   cos 4θ   cardinal vs oblique         (Fourier k=2)
    90° peak   narrow   the two 7.5° steps touching 90° vs the rest, on top of
                        the smooth harmonics

All three are signed so that POSITIVE = more prior mass / more precision there
(H-V > 0: horizontal favoured; Card-obl > 0: cardinals favoured; 90° > 0:
vertical special).

Neural (model-free). Crossnobis RDM over the 23 orientations (7.5°..172.5°)
per (subject, ROI, session); each session's distances divided by their mean.
Neighbour distance nd(m) for the 21 in-range 7.5° pairs (midpoint m; the 15°
seam pair is dropped), regressed on
    [1, cos 2m, sin 2m, cos 4m, sin 4m, touches-90°].
Efficient coding puts discriminability ∝ prior density, so the cos 2m, cos 4m
and touches-90° weights are the neural prior-shape indices.

Behavioural (model-free). Per (session, orientation) cell (~7.5 valid trials)
the SD of the bid; per subject
    log SD = session + γ log|dv/dθ| + cos 2θ + sin 2θ + cos 4θ + sin 4θ + is-90°
where dv/dθ is that session's mapping slope (central difference). The slope
term absorbs the mapping (value-stage noise passes through unchanged,
orientation noise is scaled by the slope). Lower orientation-locked SD = more
precision = more prior mass, so the indices are MINUS those weights.
Caveat: 7.5° and 172.5° map onto the ends of the bid range, where the slider
compresses bids -- that loads on the cos 2θ term for everyone (a constant, so
it does not create individual differences, but individual differences in how
people use the range end up there too).

Model-based. Per-subject a1 (cos 2θ) and a2 (cos 4θ) from the hierarchical
categorical model with a Fourier k=2 prior (efficient_coding_categorical_
fourier2_noseam; 26 subjects, numbered 3-28). Group-level r̂ ≤ 1.02 but every
iteration hit max tree depth and subjects with any prior coefficient r̂ > 1.05
are dropped. The F4/F8 fits did not converge (r̂ 2.9-4.4) and are not used.
Correlations integrate over posterior draws + subject bootstrap
(cogmodels.brain_behavior.draws_correlation).

Test-retest: each index from session 1 and session 2 separately (the two
sessions also differ in mapping), Spearman ρ across subjects. A correlation
between two measures cannot exceed roughly sqrt(rel_x * rel_y).

Spearman ρ, 95% CI by subject bootstrap; Benjamini-Hochberg FDR within each
panel. Exploratory: no prior prediction was registered.

    ~/mambaforge/envs/abstract_values/bin/python -m abstract_values.geometry.prior_shape_brain_behavior
    -> notes/figures/fig_prior_shape_brain_behavior.{pdf,png}
       notes/data/prior_shape_indices.tsv          (per subject; not committed)
       notes/data/prior_shape_correlations.tsv
"""
from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import abstract_values.connective_fields.plot_gates  # noqa: F401  (sets house rcParams)
from abstract_values.geometry.analyze_crossnobis import load
from abstract_values.utils.data import BIDS_FOLDER

REPO = Path(__file__).resolve().parents[2]
TRACE = (BIDS_FOLDER / 'derivatives' / 'cogmodels'
         / 'efficient_coding_categorical_fourier2_noseam_trace.nc')
ROIS = ['V1', 'V2', 'V3', 'V3a/b', 'IPS0', 'IPS1–2', 'IPS3–5']
IDX = ['hv', 'card', 'p90']
IDX_LAB = {'hv': 'Horizontal vs vertical (cos 2θ)', 'card': 'Cardinal vs oblique (cos 4θ)',
           'p90': '90° peak'}
IDX_COL = {'hv': '#3B5BA5', 'card': '#1B8A80', 'p90': '#C44E52'}
RHAT_MAX = 1.05


def _design(theta_deg, is90):
    t = np.deg2rad(theta_deg)
    return np.column_stack([np.cos(2 * t), np.sin(2 * t), np.cos(4 * t), np.sin(4 * t),
                            is90.astype(float)])


def _subj(s):
    s = str(s)
    return int(s[3:]) if s.startswith('pil') else int(s)


# ---------------------------------------------------------------- neural
def neural_indices():
    d = load()
    d = d[d.roi.isin(ROIS)].copy()
    d['dn'] = d.d / d.groupby(['subject', 'roi', 'session']).d.transform('mean')
    step = (d.ori_j - d.ori_i).round(1)
    nb = d[step == 7.5].copy()
    nb['mid'] = (nb.ori_i + nb.ori_j) / 2
    nb['t90'] = (nb.ori_i == 90) | (nb.ori_j == 90)
    rows = []
    for sess_lab, g0 in [('both', nb), ('1', nb[nb.session == 1]), ('2', nb[nb.session == 2])]:
        for (s, r), g in g0.groupby(['subject', 'roi']):
            g = g.groupby('mid').agg(dn=('dn', 'mean'), t90=('t90', 'first')).reset_index()
            X = np.column_stack([np.ones(len(g)), _design(g.mid.to_numpy(), g.t90.to_numpy())])
            b = np.linalg.lstsq(X, g.dn.to_numpy(), rcond=None)[0]
            rows.append(dict(subject=_subj(s), source=r, session=sess_lab,
                             hv=b[1], card=b[3], p90=b[5]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- behaviour
def behaviour_indices():
    from abstract_values.behavior.data import get_all_behavioral_data
    df = get_all_behavioral_data()
    df = df[df.event_type == 'feedback'].copy()
    df['response'] = pd.to_numeric(df.response, errors='coerce')
    df = df.reset_index().dropna(subset=['response'])
    cell = (df.groupby(['subject', 'session', 'mapping', 'orientation'])
              .agg(sd=('response', 'std'), n=('response', 'size'), value=('value', 'first'))
              .reset_index())
    cell = cell[cell.n >= 3]

    def slope(g):
        g = g.sort_values('orientation')
        g['slope'] = np.abs(np.gradient(g.value.to_numpy(), g.orientation.to_numpy()))
        return g
    cell = cell.groupby(['subject', 'session'], group_keys=False).apply(slope)
    cell = cell[(cell.sd > 0) & (cell.slope > 0)]
    rows = []
    for sess_lab in ['both', 1, 2]:
        c0 = cell if sess_lab == 'both' else cell[cell.session == sess_lab]
        for s, g in c0.groupby('subject'):
            cols = [np.ones(len(g)), np.log(g.slope)]
            if sess_lab == 'both':
                cols.append((g.session == 2).astype(float))
            X = np.column_stack(cols + [_design(g.orientation.to_numpy(),
                                                 (g.orientation == 90).to_numpy())])
            b = np.linalg.lstsq(X, np.log(g.sd.to_numpy()), rcond=None)[0][len(cols):]
            rows.append(dict(subject=int(s), source='Behaviour', session=str(sess_lab),
                             hv=-b[0], card=-b[2], p90=-b[4]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- model
def model_draws():
    idata = az.from_netcdf(TRACE)
    post = idata.posterior
    rh = az.rhat(post[['prior_a1', 'prior_b1', 'prior_a2', 'prior_b2']])
    bad = np.zeros(post.sizes['subject'], bool)
    for v in rh.data_vars:
        bad |= rh[v].values > RHAT_MAX
    subs = post.subject.values[~bad].astype(int)
    out = {}
    for k, v in [('hv', 'prior_a1'), ('card', 'prior_a2')]:
        x = post[v].stack(s=('chain', 'draw')).transpose('s', 'subject').values
        out[k] = x[:, ~bad]
    dropped = post.subject.values[bad].astype(int).tolist()
    return subs, out, dropped


# ---------------------------------------------------------------- stats
def boot_rho(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r, p = stats.spearmanr(x, y)
    bs = [stats.spearmanr(x[i], y[i]).statistic
          for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return r, lo, hi, p, len(x)


def draws_rho(xd, y, n=2000, seed=0):
    """Spearman ρ per (posterior draw, subject bootstrap); p from the
    two-sided fraction of the distribution beyond 0."""
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n):
        i = rng.integers(xd.shape[0])
        b = rng.integers(0, len(y), len(y))
        rs.append(stats.spearmanr(xd[i][b], y[b]).statistic)
    rs = np.array(rs)
    p = 2 * min((rs <= 0).mean(), (rs >= 0).mean())
    return np.median(rs), *np.percentile(rs, [2.5, 97.5]), max(p, 1 / n), len(y)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def main():
    ind = pd.concat([neural_indices(), behaviour_indices()], ignore_index=True)
    ind.to_csv(REPO / 'notes/data/prior_shape_indices.tsv', sep='\t', index=False)
    both = ind[ind.session == 'both']
    beh = both[both.source == 'Behaviour'].set_index('subject')
    sources = ['Behaviour'] + ROIS
    rows = []

    # group effect (standardised mean) and test-retest
    for src in sources:
        for k in IDX:
            v = both.loc[both.source == src, k].to_numpy()
            t = stats.ttest_1samp(v, 0)
            s1 = ind[(ind.source == src) & (ind.session == '1')].set_index('subject')[k]
            s2 = ind[(ind.source == src) & (ind.session == '2')].set_index('subject')[k]
            j = s1.index.intersection(s2.index)
            r, lo, hi, p, n = boot_rho(s1[j].to_numpy(), s2[j].to_numpy())
            rows.append(dict(panel='group', source=src, index=k, est=v.mean() / v.std(ddof=1),
                             lo=np.nan, hi=np.nan, p=t.pvalue, n=len(v)))
            rows.append(dict(panel='retest', source=src, index=k, est=r, lo=lo, hi=hi, p=p, n=n))

    # behaviour (model-free) vs brain
    for src in ROIS:
        nr = both[both.source == src].set_index('subject')
        j = nr.index.intersection(beh.index)
        for k in IDX:
            r, lo, hi, p, n = boot_rho(beh.loc[j, k].to_numpy(), nr.loc[j, k].to_numpy())
            rows.append(dict(panel='beh_brain', source=src, index=k, est=r, lo=lo, hi=hi, p=p, n=n))

    # model (F2) vs behaviour and vs brain
    subs, md, dropped = model_draws()
    print(f'Model: {len(subs)} subjects kept, dropped (r̂ > {RHAT_MAX}): {dropped}')
    for src in sources:
        tab = beh if src == 'Behaviour' else both[both.source == src].set_index('subject')
        keep = np.isin(subs, tab.index)
        for k in ['hv', 'card']:
            y = tab.loc[subs[keep], k].to_numpy()
            r, lo, hi, p, n = draws_rho(md[k][:, keep], y)
            rows.append(dict(panel='model', source=src, index=k, est=r, lo=lo, hi=hi, p=p, n=n))

    res = pd.DataFrame(rows)
    res['q'] = np.nan
    for pan in ['group', 'retest', 'beh_brain', 'model']:
        m = res.panel == pan
        res.loc[m, 'q'] = bh(res.loc[m, 'p'])
    res.to_csv(REPO / 'notes/data/prior_shape_correlations.tsv', sep='\t', index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(res.round(3).to_string())
    plot(res, len(subs))


def plot(res, n_model):
    sources = ['Behaviour'] + ROIS
    xlab = ['Beh.'] + ROIS
    fig, axes = plt.subplots(1, 4, figsize=(7.25, 2.35), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1, .9, 1]))
    off = {'hv': -.22, 'card': 0, 'p90': .22}

    def dots(ax, pan, srcs, idx, start=0):
        for k in idx:
            r = res[(res.panel == pan) & (res['index'] == k)].set_index('source').loc[srcs]
            x = np.arange(len(srcs)) + start + off[k] * (1.0 if len(idx) == 3 else 1.4)
            if r.lo.notna().any():
                ax.vlines(x, r.lo, r.hi, color=IDX_COL[k], lw=.8, alpha=.7)
            sig = (r.q < .05).to_numpy()
            ax.plot(x[sig], r.est[sig], 'o', ms=3.6, color=IDX_COL[k], mew=0)
            ax.plot(x[~sig], r.est[~sig], 'o', ms=3.6, mfc='white', mec=IDX_COL[k], mew=.9)
        ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)

    # a: group effect
    ax = axes[0]
    dots(ax, 'group', sources, IDX)
    ax.set_xticks(range(len(sources)), xlab)
    ax.set_ylabel('Group effect (mean / SD)')

    # b: test-retest
    ax = axes[1]
    dots(ax, 'retest', sources, IDX)
    ax.set_xticks(range(len(sources)), xlab)
    ax.set_ylabel('Session 1 vs 2 (ρ)')
    ax.text(.5, .97, 'No index is reliable', transform=ax.transAxes, ha='center', va='top', fontsize=6.5, color='.3')
    ax.set_ylim(-.75, 1)
    # c: behaviour vs brain
    ax = axes[2]
    dots(ax, 'beh_brain', ROIS, IDX)
    ax.set_xticks(range(len(ROIS)), ROIS)
    ax.set_ylabel('Behaviour vs brain (ρ)')
    ax.set_ylim(-.75, .75)
    # d: model vs behaviour / brain
    ax = axes[3]
    dots(ax, 'model', sources, ['hv', 'card'])
    ax.set_xticks(range(len(sources)), xlab)
    ax.set_ylabel(f'Fitted prior vs measure (ρ, n = {n_model})')
    ax.set_ylim(-.75, .75)


    # key: glyphs for the three indices + filled/open
    ax = axes[0]
    for i, k in enumerate(IDX):
        y = .97 - i * .085
        ax.plot(.30, y, 'o', ms=3.6, color=IDX_COL[k], mew=0, transform=ax.transAxes, clip_on=False)
        ax.text(.34, y, IDX_LAB[k], color=IDX_COL[k], fontsize=6, va='center', transform=ax.transAxes)
    ax = axes[2]
    ax.plot(.06, .97, 'o', ms=3.6, color='.3', mew=0, transform=ax.transAxes)
    ax.text(.11, .97, 'FDR q < .05', fontsize=6, va='center', transform=ax.transAxes, color='.3')
    ax.plot(.06, .885, 'o', ms=3.6, mfc='white', mec='.3', mew=.9, transform=ax.transAxes)
    ax.text(.11, .885, 'n.s.', fontsize=6, va='center', transform=ax.transAxes, color='.3')

    for a, l in zip(axes, 'abcd'):
        a.text(-.28, 1.04, l, transform=a.transAxes, fontsize=8, fontweight='bold',
               va='bottom', ha='right')
    for a in axes:
        for s in ('left', 'bottom'):
            a.spines[s].set_position(('outward', 4))
    for a in axes:
        plt.setp(a.get_xticklabels(), rotation=60, ha='right', rotation_mode='anchor')
    axes[0].set_ylim(-.3, 2.3)
    out = REPO / 'notes/figures/fig_prior_shape_brain_behavior'
    fig.savefig(f'{out}.pdf')
    fig.savefig(f'{out}.png', dpi=300)
    print(f'Wrote {out}.pdf / .png')


if __name__ == '__main__':
    main()
