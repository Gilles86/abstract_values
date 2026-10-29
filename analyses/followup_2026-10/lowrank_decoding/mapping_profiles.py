#!/usr/bin/env python3
"""Where on the value line is decoding precise? Per mapping, value axis vs orientation axis.

Input: results/mapping_profiles_<setting>.tsv.gz (mapping_profiles_collect.py, cluster),
results/simdecode_all.tsv.gz (simulate_decoding_mix outputs, optional), behaviour.

Per participant x decoder/ROI x mapping x stimulus (the 23 presented pairs):
posterior SD, |error|, signed bias, log score, 80% HPD coverage -- on the value
axis (CHF) and on the orientation axis (deg). Group: mean over participants
with 95% t-CI.

Formal tests (per participant, then t-CI across participants):
  beta_v   OLS slope of log posterior SD (CHF) on log |dv/dtheta| at each
           (mapping, stimulus), mapping intercepts. An orientation code -- and,
           because quantile = orientation rank in this design, ALSO a quantile /
           efficient value code -- predicts 1 (precise where the mapping packs
           values densely). A fixed-value code predicts 0.
  beta_o   the same on the orientation axis (log posterior SD in deg):
           orientation/quantile code 0, fixed-value code -1.
  r_v, r_o mapping-profile similarity: Pearson r between the two mappings'
           log-SD profiles on a common value grid (r_v) / over the 23
           orientations (r_o). Orientation/quantile code: r_o high, r_v low;
           fixed-value code: the reverse.
The same betas for behaviour (bid SD per stimulus and mapping).

    python analyses/followup_2026-10/lowrank_decoding/mapping_profiles.py \
        --profiles results/mapping_profiles_nvoxels-250_noise-lowrankcv.tsv.gz --label "Low-rank, 250 voxels"
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.behavior.data import get_all_behavioral_data
from abstract_values.connective_fields.plot_gates import letter  # sets rcParams

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
MAP_COL = {'cdf': '#C44E52', 'inverse_cdf': '#3B5BA5'}
MAP_LABEL = {'cdf': 'Misaligned', 'inverse_cdf': 'Aligned'}
CELLS = [('gabor', 'BensonV1', 'V1 · orientation decoder'),
         ('gabor', 'BensonV2', 'V2 · orientation decoder'),
         ('gabor', 'BensonV3', 'V3 · orientation decoder'),
         ('value', 'BensonV1', 'V1 · value decoder'),
         ('value', 'IPS0', 'IPS0 · value decoder'),
         ('value', 'IPS12', 'IPS1–2 · value decoder'),
         ('value', 'IPS345', 'IPS3–5 · value decoder'),
         ('value', 'NPCr', 'NPCr · value decoder')]
SIM_ROI = {'BensonV1': 'BensonV1', 'BensonV2': 'BensonV2', 'BensonV3': 'BensonV3',
           'IPS0': 'IPS0', 'IPS12': 'IPS12', 'IPS345': 'IPS345'}
COMMON_V = np.linspace(4, 40, 10)


def subject_number(s):
    return int(''.join(c for c in str(s) if c.isdigit()))


def tci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, len(x), np.nan
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(.975, len(x) - 1) * se
    return m, m - h, m + h, len(x), stats.ttest_1samp(x, 0).pvalue


def slopes(d):
    """|dv/dtheta| (CHF/deg) per (subject, mapping, orientation) from the 23 pairs."""
    rows = []
    for (s, mp), g in d.groupby(['subject', 'mapping']):
        t = g.groupby('orientation').value.mean().sort_index()
        sl = np.abs(np.gradient(t.values, t.index.values))
        rows.append(pd.DataFrame(dict(subject=s, mapping=mp, orientation=t.index,
                                      slope=sl)))
    return pd.concat(rows)


def cell_means(d):
    d = d.assign(abs_v=d.v_err.abs(), abs_o=d.o_err.abs(), cov80=(d.hpd < .8).astype(float),
                 cov50=(d.hpd < .5).astype(float), cov95=(d.hpd < .95).astype(float))
    keys = ['decoder', 'roi', 'subject', 'mapping', 'orientation']
    return d.groupby(keys).agg(
        value=('value', 'mean'), v_sd=('v_sd', 'mean'), o_sd=('o_sd', 'mean'),
        abs_v=('abs_v', 'mean'), abs_o=('abs_o', 'mean'), v_bias=('v_err', 'mean'),
        o_bias=('o_err', 'mean'), logp=('logp', 'mean'), cov50=('cov50', 'mean'),
        cov80=('cov80', 'mean'), cov95=('cov95', 'mean'), n=('v_sd', 'size')).reset_index()


def betas(c, sd_v='v_sd', sd_o='o_sd'):
    """Per participant: beta_v, beta_o, r_v, r_o (see module docstring)."""
    out = []
    for key, g in c.groupby([k for k in ('decoder', 'roi') if k in c] + ['subject']):
        g = g[np.isfinite(np.log(g[sd_v])) & np.isfinite(np.log(g[sd_o])) & (g.slope > 0)]
        if g.mapping.nunique() < 2:
            continue
        X = np.c_[np.ones(len(g)), (g.mapping == 'cdf').astype(float), np.log(g.slope)]
        bv = np.linalg.lstsq(X, np.log(g[sd_v]), rcond=None)[0][2]
        bo = np.linalg.lstsq(X, np.log(g[sd_o]), rcond=None)[0][2]
        a = g[g.mapping == 'cdf'].sort_values('value')
        b = g[g.mapping == 'inverse_cdf'].sort_values('value')
        rv = np.corrcoef(np.interp(COMMON_V, a.value, np.log(a[sd_v])),
                         np.interp(COMMON_V, b.value, np.log(b[sd_v])))[0, 1]
        ab = a.set_index('orientation')[sd_o].pipe(np.log).align(
            b.set_index('orientation')[sd_o].pipe(np.log), join='inner')
        ro = np.corrcoef(ab[0], ab[1])[0, 1]
        out.append(dict(zip([k for k in ('decoder', 'roi') if k in c] + ['subject'],
                            key if isinstance(key, tuple) else (key,)),
                        beta_v=bv, beta_o=bo, r_v=rv, r_o=ro))
    return pd.DataFrame(out)


def interaction(c, col, space, n_perm=5000, seed=0):
    """Mapping x position interaction of log(col), per decoder/ROI.

    Per participant: D(p) = log col[cdf](p) - log col[inverse_cdf](p) at matched
    positions p (value axis: COMMON_V by interpolation; orientation axis: the 23
    orientations), centred over p (removes the mapping main effect). Statistic
    = mean over p of (group-mean centred D)^2, i.e. squared RMS of the group
    interaction profile. Null: sign flips of whole participant profiles.
    95% CI of the RMS by participant bootstrap.
    """
    rng = np.random.default_rng(seed)
    out = []
    keys = [k for k in ('decoder', 'roi') if k in c]
    for key, g in c.groupby(keys):
        D = []
        for s, h in g.groupby('subject'):
            h = h.copy()
            # floor zero SDs (e.g. identical bids) at 5% of the participant's median
            h[col] = np.maximum(h[col], 0.05 * np.nanmedian(h[col]))
            a = h[h.mapping == 'cdf']
            b = h[h.mapping == 'inverse_cdf']
            if len(a) < 5 or len(b) < 5:
                continue
            if space == 'value':
                a, b = a.sort_values('value'), b.sort_values('value')
                da = np.interp(COMMON_V, a.value, np.log(a[col]))
                db = np.interp(COMMON_V, b.value, np.log(b[col]))
            else:
                ab = a.set_index('orientation')[col].align(b.set_index('orientation')[col],
                                                           join='inner')
                da, db = np.log(ab[0].values), np.log(ab[1].values)
            d = da - db
            if np.all(np.isfinite(d)):
                D.append(d - d.mean())
        L = min(len(x) for x in D)
        D = np.array([x[:L] for x in D])
        stat = lambda M: np.mean(M.mean(0) ** 2)
        obs = stat(D)
        null = np.array([stat(D * rng.choice([-1, 1], size=(len(D), 1))) for _ in range(n_perm)])
        boot = np.array([np.sqrt(stat(D[rng.integers(0, len(D), len(D))])) for _ in range(2000)])
        out.append(dict(zip(keys, key if isinstance(key, tuple) else (key,)), space=space,
                        measure=col, rms=np.sqrt(obs), lo=np.percentile(boot, 2.5),
                        hi=np.percentile(boot, 97.5),
                        null_rms=np.sqrt(np.median(null)),
                        p=(1 + (null >= obs).sum()) / (1 + n_perm), n=len(D)))
    return pd.DataFrame(out)


def behaviour_cells():
    b = get_all_behavioral_data()
    b = b[b.event_type == 'feedback'].copy()
    b['bid'] = pd.to_numeric(b['response'], errors='coerce')
    b = b.reset_index().dropna(subset=['bid'])
    b['orientation'] = b.orientation.astype(float).round(2)
    rows = []
    for (s, mp), g in b.groupby(['subject', 'mapping']):
        t = g.groupby('orientation').value.mean().sort_index()
        bid_ori = np.interp(g.bid, t.values, t.index.values)       # bid -> orientation
        g = g.assign(bid_ori=bid_ori)
        a = g.groupby('orientation').agg(value=('value', 'mean'), v_sd=('bid', 'std'),
                                         o_sd=('bid_ori', 'std'), n=('bid', 'size'))
        a['slope'] = np.abs(np.gradient(t.values, t.index.values))
        rows.append(a.reset_index().assign(subject=str(s), mapping=mp))
    return pd.concat(rows, ignore_index=True)


def group(c, cols):
    rows = []
    for key, g in c.groupby([k for k in ('decoder', 'roi') if k in c] + ['mapping', 'orientation']):
        r = dict(zip([k for k in ('decoder', 'roi') if k in c] + ['mapping', 'orientation'], key))
        r['value'] = g.value.mean()
        r['n_subjects'] = g.subject.nunique()
        for col in cols:
            m, lo, hi, *_ = tci(g[col])
            r[col], r[f'{col}_lo'], r[f'{col}_hi'] = m, lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def figure(gp, gb, sim, label, out):
    nrow = 1 + len(CELLS)
    fig, axes = plt.subplots(nrow, 2, figsize=(5.6, 1.25 * nrow), constrained_layout=True,
                             sharex='col')
    rows = [('beh', None, 'Behaviour · bid SD')] + CELLS
    for i, (dec, roi, title) in enumerate(rows):
        g = gb if dec == 'beh' else gp[(gp.decoder == dec) & (gp.roi == roi)]
        for j, (xcol, ycol) in enumerate((('value', 'v_sd'), ('orientation', 'o_sd'))):
            ax = axes[i, j]
            for mp, h in g.groupby('mapping'):
                h = h.sort_values(xcol)
                ax.fill_between(h[xcol], h[f'{ycol}_lo'], h[f'{ycol}_hi'], color=MAP_COL[mp],
                                alpha=.2, lw=0)
                ax.plot(h[xcol], h[ycol], color=MAP_COL[mp], lw=1.1)
                if dec != 'beh' and 'cov80' in h:
                    bad = h[(h.cov80 < .7) | (h.cov80 > .9)]
                    ax.plot(bad[xcol], bad[ycol], 'o', ms=2.4, mfc='white', mec=MAP_COL[mp],
                            mew=.7, zorder=4)
            if sim is not None and dec == 'value' and roi in SIM_ROI and j == 0:
                s = sim[sim.roi == SIM_ROI[roi]]
                for lam, ls in ((0.0, '--'), (1.0, ':')):
                    for mp, h in s[s.lam == lam].groupby('mapping'):
                        h = h.sort_values(xcol)
                        ax.plot(h[xcol], h[ycol], color=MAP_COL[mp], lw=.8, ls=ls, alpha=.9)
            ax.set_title(title if j == 0 else '', fontsize=6.5, loc='left')
            ax.set_ylabel('SD (CHF)' if j == 0 else 'SD (deg)', fontsize=6.5)
            sns.despine(ax=ax, offset=2)
    axes[-1, 0].set_xlabel('True value (CHF)')
    axes[-1, 1].set_xlabel('Orientation (deg)')
    axes[-1, 0].set_xticks([2, 12, 22, 32, 42])
    axes[-1, 1].set_xticks([0, 45, 90, 135, 180])
    a0 = axes[0, 0]
    for k, mp in enumerate(MAP_COL):
        a0.text(.02, .92 - .17 * k, MAP_LABEL[mp], color=MAP_COL[mp], fontsize=6,
                transform=a0.transAxes, va='top')
    ax = axes[4, 0]          # V1 value decoder: empty band between data and model
    for k, (lab, ls) in enumerate((('Model λ = 0 (fixed value)', '--'),
                                   ('Model λ = 1 (quantile)', ':'))):
        y = .62 - .13 * k
        ax.plot([.04, .12], [y, y], transform=ax.transAxes, color='.3', lw=.8, ls=ls)
        ax.text(.14, y, lab, transform=ax.transAxes, fontsize=5.5, va='center', color='.3')
    ax.plot(.08, .36, 'o', ms=2.4, mfc='white', mec='.3', mew=.7, transform=ax.transAxes)
    ax.text(.14, .36, '80% coverage off by > .1', transform=ax.transAxes, fontsize=5.5,
            va='center', color='.3')
    fig.suptitle(label, fontsize=7)
    letter(axes[0, 0], 'a')
    letter(axes[0, 1], 'b')
    fig.savefig(out)
    print(f'saved {out}')


def main(profiles, label, sim_fn, tag):
    d = pd.read_csv(profiles, sep='\t', dtype={'subject': str})
    c = cell_means(d).merge(slopes(d), on=['subject', 'mapping', 'orientation'])
    c.to_csv(RES / f'mapping_profiles_cells_{tag}.tsv', sep='\t', index=False, float_format='%.5g')
    cols = ['v_sd', 'o_sd', 'abs_v', 'abs_o', 'v_bias', 'o_bias', 'logp', 'cov50', 'cov80', 'cov95']
    gp = group(c, cols)
    gp.to_csv(RES / f'mapping_profiles_group_{tag}.tsv', sep='\t', index=False, float_format='%.4g')

    bc = behaviour_cells()
    gb = group(bc, ['v_sd', 'o_sd'])
    gb.to_csv(RES / 'mapping_profiles_behaviour_group.tsv', sep='\t', index=False,
              float_format='%.4g')

    bt = betas(c)
    bt.to_csv(RES / f'mapping_profiles_betas_by_subject_{tag}.tsv', sep='\t', index=False,
              float_format='%.4g')
    bb = betas(bc)
    rows = []
    for (dec, roi), g in bt.groupby(['decoder', 'roi']):
        for m in ('beta_v', 'beta_o', 'r_v', 'r_o'):
            mm, lo, hi, n, p = tci(g[m])
            rows.append(dict(decoder=dec, roi=roi, measure=m, mean=mm, lo=lo, hi=hi, n=n, p=p))
    for m in ('beta_v', 'beta_o', 'r_v', 'r_o'):
        mm, lo, hi, n, p = tci(bb[m])
        rows.append(dict(decoder='behaviour', roi='bid', measure=m, mean=mm, lo=lo, hi=hi,
                         n=n, p=p))
    # IPS / NPCr value decoders vs V1 value decoder vs V1 orientation decoder, paired
    w = bt.assign(cell=bt.decoder + ':' + bt.roi).pivot_table(index='subject', columns='cell',
                                                                values='beta_v')
    for a in ('value:IPS0', 'value:IPS12', 'value:IPS345', 'value:NPCr'):
        for b in ('value:BensonV1', 'gabor:BensonV1'):
            if a in w and b in w:
                mm, lo, hi, n, p = tci(w[a] - w[b])
                rows.append(dict(decoder='contrast', roi=f'{a} − {b}', measure='beta_v',
                                 mean=mm, lo=lo, hi=hi, n=n, p=p))
    tests = pd.DataFrame(rows)
    tests.to_csv(RES / f'mapping_profiles_tests_{tag}.tsv', sep='\t', index=False,
                 float_format='%.4g')
    print(tests.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    inter = pd.concat([interaction(c, 'v_sd', 'value'), interaction(c, 'o_sd', 'orientation'),
                       interaction(c, 'abs_v', 'value'), interaction(c, 'abs_o', 'orientation'),
                       interaction(bc.assign(decoder='behaviour', roi='bid'), 'v_sd', 'value'),
                       interaction(bc.assign(decoder='behaviour', roi='bid'), 'o_sd',
                                   'orientation')])
    inter.to_csv(RES / f'mapping_profiles_interaction_{tag}.tsv', sep='\t', index=False,
                 float_format='%.4g')
    print(inter.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    cov = gp.groupby(['decoder', 'roi', 'mapping'])[['cov50', 'cov80', 'cov95']].agg(['min', 'max'])
    print(cov.round(2).to_string())

    sim = None
    if sim_fn and Path(sim_fn).exists():
        s = pd.read_csv(sim_fn, sep='\t', dtype={'subject': str})
        s = s[s.lam.isin([0.0, 1.0]) & ~s.smoothed.astype(bool)]
        # slope per (mapping, value) from the decoding table, for the orientation axis
        sl = c.groupby(['mapping', 'orientation']).agg(value=('value', 'mean'),
                                                       slope=('slope', 'mean')).reset_index()
        rows = []
        for (roi, lam, mp), g in s.groupby(['roi', 'lam', 'mapping']):
            gm = g.groupby('value').post_sd.mean().reset_index()
            t = sl[sl.mapping == mp].sort_values('value')
            gm['orientation'] = np.interp(gm.value, t.value, t.orientation)
            gm['slope'] = np.interp(gm.value, t.value, t.slope)
            rows.append(gm.assign(roi=roi, lam=lam, mapping=mp, v_sd=gm.post_sd,
                                  o_sd=gm.post_sd / gm.slope))
        sim = pd.concat(rows)
        sim.to_csv(RES / 'mapping_profiles_simulated_group.tsv', sep='\t', index=False,
                   float_format='%.4g')
    figure(gp, gb, sim, label, RES / f'fig_mapping_profiles_{tag}.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--profiles', required=True)
    p.add_argument('--label', default='')
    p.add_argument('--sim', default=str(RES / 'simdecode_all.tsv.gz'))
    p.add_argument('--tag', default='best')
    a = p.parse_args()
    main(a.profiles, a.label, a.sim, a.tag)
