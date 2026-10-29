#!/usr/bin/env python3
"""Decoding precision as a function of value, per mapping: data vs simulate-then-decode.

Error-based measures per participant x ROI x mapping x presented stimulus,
computed identically for every source (empirical decoders and simulations):

  sd     SD of the decoding errors within a bin (5-CHF value bins / 22.5-deg
         orientation bins; for a single stimulus = SD of the decoded posterior
         means) = precision, on each axis
  bias   mean signed error;  abs  mean |error|
  post   mean posterior SD -- calibrated only for V1-V3 (see README)

Sources:
  emp_lowrank     CV-N low-rank + T  (results/nvcv/mapping_profiles_nvoxels-cv_noise-lowrankcv.tsv.gz)
  emp_spherical   spherical, N = 50  (results/mapping_profiles_nvoxels-50_noise-spherical.tsv.gz)
  ideal_<code>    ideal observer (true tuning + noise), noise scale 1; per
                  replicate, then averaged over replicates (matched trial counts)
  pipe_<code>     the real CV-N low-rank pipeline run on simulated data
codes: vonmises (orientation decoders' own fitted code), lam0 (fixed value),
lam1 (quantile = orientation rank).

Tests (per source x ROI): mapping x position interaction of log sd on the value
axis (10 matched values) and on the orientation axis (23 orientations): RMS of
the participant-centred group mapping-difference profile, sign-flip p (5000),
bootstrap CI. Closeness: per participant, RMS distance between the centred
empirical log-sd profile (mapping x bin) and each simulated one (lam0 vs lam1); delta =
d(lam0) - d(lam1) > 0 means closer to lam1.

Diagnosis (ideal sims): the same interaction as a function of noise scale
(1, .5, .25, .1), on posterior SD (calibrated by construction).
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapping_profiles import MAP_COL, MAP_LABEL, tci  # noqa: E402
from abstract_values.connective_fields.plot_gates import letter  # noqa: E402  (rcParams)

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
ROWS = [('gabor', 'BensonV1', 'V1 · orientation decoder', 'vonmises'),
        ('gabor', 'BensonV2', 'V2 · orientation decoder', 'vonmises'),
        ('gabor', 'BensonV3', 'V3 · orientation decoder', 'vonmises'),
        ('value', 'BensonV1', 'V1 · value decoder', 'lam'),
        ('value', 'IPS0', 'IPS0 · value decoder', 'lam'),
        ('value', 'IPS12', 'IPS1–2 · value decoder', 'lam'),
        ('value', 'IPS345', 'IPS3–5 · value decoder', 'lam'),
        ('value', 'NPCr', 'NPCr · value decoder', 'lam')]
SIM_LS = {'vonmises': '--', 'lam0': '--', 'lam1': ':'}
SIM_LABEL = {'vonmises': 'Sim: fitted orientation code', 'lam0': 'Sim: λ = 0 (fixed value)',
             'lam1': 'Sim: λ = 1 (quantile)'}


BINS = {'value': np.arange(2, 42.1, 5.0), 'orientation': np.arange(0, 180.1, 22.5)}
MIN_N = 4


def cells(d, by=()):
    """Per (source cols..., subject, roi, mapping, space, bin) error-based metrics.

    Bins: 5-CHF value bins and 22.5-deg orientation bins. Precision = SD of the
    decoding errors pooled over the presented stimuli in the bin (for one
    stimulus this is the SD of the decoded posterior means).
    """
    out = []
    keys = list(by) + ['decoder', 'roi', 'subject', 'mapping']
    for space, err, post, x in (('value', 'v_err', 'v_sd', 'value'),
                                ('orientation', 'o_err', 'o_sd', 'orientation')):
        e = BINS[space]
        # pooled WITHIN-stimulus SD: the spread of bias across the stimuli in a
        # bin (central pull) must not count as imprecision
        dd = d[list(dict.fromkeys(keys + ['orientation', x, err, post]))].assign(
            _abs=d[err].abs(), _cov=(d['hpd'] < .8).astype(float))
        st = (dd.groupby(keys + ['orientation'], sort=False)
              .agg(x=(x, 'mean'), var=(err, 'var'), m=(err, 'mean'), a=('_abs', 'mean'),
                   post=(post, 'mean'), cov80=('_cov', 'mean'), n=(err, 'size')).reset_index())
        st['bin'] = pd.cut(st.x, e, labels=(e[:-1] + e[1:]) / 2, include_lowest=True).astype(float)
        st['ss'] = st['var'].fillna(0) * (st.n - 1)
        for col in ('m', 'a', 'post', 'cov80'):
            st[col + '_w'] = st[col] * st.n
        g = st.groupby(keys + ['bin'], sort=False)
        tot = g[['ss', 'n', 'm_w', 'a_w', 'post_w', 'cov80_w']].sum()
        cnt = g.n.count()
        c = pd.DataFrame({'sd': np.sqrt(tot.ss / (tot.n - cnt)),
                          'bias': tot.m_w / tot.n, 'abs': tot.a_w / tot.n,
                          'post': tot.post_w / tot.n, 'cov80': tot.cov80_w / tot.n,
                          'n': tot.n}).reset_index()
        out.append(c[c.n >= MIN_N].assign(space=space))
    return pd.concat(out, ignore_index=True)


def interaction_binned(c, col='sd', n_perm=5000, seed=0):
    """Mapping x position interaction of log(col) per decoder/ROI and space (see docstring)."""
    rng = np.random.default_rng(seed)
    rows = []
    for (dec, roi, space), g in c.groupby(['decoder', 'roi', 'space']):
        w = g.pivot_table(index=['subject', 'bin'], columns='mapping', values=col).dropna()
        if not {'cdf', 'inverse_cdf'} <= set(w.columns):
            continue
        D = np.log(np.maximum(w['cdf'], 1e-3)) - np.log(np.maximum(w['inverse_cdf'], 1e-3))
        D = D.unstack('bin')
        keep = D.columns[D.notna().mean() >= .9]
        D = D[keep].dropna()
        if len(D) < 5 or len(keep) < 3:
            continue
        M = D.values - D.values.mean(1, keepdims=True)
        stat = lambda X: np.mean(X.mean(0) ** 2)
        obs = stat(M)
        null = np.array([stat(M * rng.choice([-1, 1], size=(len(M), 1))) for _ in range(n_perm)])
        boot = np.array([np.sqrt(stat(M[rng.integers(0, len(M), len(M))])) for _ in range(1000)])
        rows.append(dict(decoder=dec, roi=roi, space=space, measure=col, rms=np.sqrt(obs),
                         lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
                         null_rms=np.sqrt(np.median(null)),
                         p=(1 + (null >= obs).sum()) / (1 + n_perm), n=len(M),
                         bins=','.join(f'{b:g}' for b in keep)))
    return pd.DataFrame(rows)


def avg_reps(c):
    num = ['sd', 'bias', 'abs', 'post', 'cov80', 'n']
    return c.groupby(['decoder', 'roi', 'subject', 'mapping', 'space', 'bin'])[num].mean().reset_index()


def load_sources(ideal_scale=1.0):
    src = {}
    for name, fn in (('emp_lowrank', RES / 'nvcv' / 'mapping_profiles_nvoxels-cv_noise-lowrankcv.tsv.gz'),
                     ('emp_spherical', RES / 'mapping_profiles_nvoxels-50_noise-spherical.tsv.gz')):
        if fn.exists():
            src[name] = cells(pd.read_csv(fn, sep='\t', dtype={'subject': str}))
    sd = RES / 'simideal'
    ideal = None
    if (sd / 'ideal_all.tsv.gz').exists():
        ideal = pd.read_csv(sd / 'ideal_all.tsv.gz', sep='\t', dtype={'subject': str})
        for code, g in ideal[ideal.noise_scale == ideal_scale].groupby('code'):
            c = cells(g, by=['rep'])
            src[f'ideal_{code}'] = avg_reps(c)
    for code in ('vonmises', 'lam0', 'lam1'):
        fn = sd / f'pipeline_{code}.tsv.gz'
        if fn.exists():
            src[f'pipe_{code}'] = cells(pd.read_csv(fn, sep='\t', dtype={'subject': str}))
    return src, ideal


def group(c, cols=('sd', 'bias', 'abs', 'post', 'cov80')):
    rows = []
    for key, g in c.groupby(['decoder', 'roi', 'space', 'mapping', 'bin']):
        r = dict(zip(['decoder', 'roi', 'space', 'mapping', 'bin'], key))
        r['n_subjects'] = g.subject.nunique()
        for col in cols:
            m, lo, hi, *_ = tci(g[col])
            r[col], r[f'{col}_lo'], r[f'{col}_hi'] = m, lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def profile(c, space='value', col='sd'):
    """{subject: centred log-precision profile over (mapping x bin)}."""
    out = {}
    c = c[c.space == space]
    for subj, h in c.groupby('subject'):
        v = h.set_index(['mapping', 'bin'])[col]
        out[subj] = np.log(np.maximum(v, 1e-3))
    return out


def closeness(src, sim_kind, space='value'):
    rows = []
    emp = src['emp_lowrank']
    for dec, roi, _, family in ROWS:
        if family != 'lam':
            continue
        e = profile(emp[(emp.decoder == dec) & (emp.roi == roi)], space)
        sims = {}
        for code in ('lam0', 'lam1'):
            k = f'{sim_kind}_{code}'
            if k in src:
                s = src[k]
                sims[code] = profile(s[(s.decoder == dec) & (s.roi == roi)], space)
        if len(sims) < 2:
            continue
        for subj, ev in e.items():
            if subj in sims['lam0'] and subj in sims['lam1']:
                idx = ev.index.intersection(sims['lam0'][subj].index).intersection(
                    sims['lam1'][subj].index)
                if len(idx) < 6:
                    continue
                cen = lambda x: x.loc[idx] - x.loc[idx].mean()
                E, S0, S1 = cen(ev), cen(sims['lam0'][subj]), cen(sims['lam1'][subj])
                d0 = np.sqrt(np.mean((E - S0) ** 2))
                d1 = np.sqrt(np.mean((E - S1) ** 2))
                d01 = np.sqrt(np.mean((S0 - S1) ** 2))
                rows.append(dict(sim=sim_kind, space=space, roi=roi, subject=subj, d_lam0=d0, d_lam1=d1,
                                 delta=d0 - d1, d_lam0_lam1=d01))
    return pd.DataFrame(rows)


def diagnosis(ideal):
    """Mapping x position interaction of the ideal observer vs noise scale.

    On posterior SD ('post', calibrated by construction for the ideal observer)
    and on the SD of decoding errors ('sd').
    """
    rows = []
    for (code, scale), g in ideal.groupby(['code', 'noise_scale']):
        c = avg_reps(cells(g, by=['rep']))
        for col in ('post', 'sd'):
            r = interaction_binned(c, col, n_perm=1000)
            r['code'], r['noise_scale'] = code, scale
            rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    mae = (ideal.assign(a=ideal.v_err.abs()).groupby(['code', 'roi', 'noise_scale']).a.mean()
           .rename('mae_v').reset_index())
    return out.merge(mae, on=['code', 'roi', 'noise_scale'], how='left')


def figure(gsrc, sim_kind, out, title):
    fig, axes = plt.subplots(len(ROWS), 2, figsize=(5.6, 1.3 * len(ROWS)),
                             constrained_layout=True, sharex='col')
    emp = gsrc['emp_lowrank']
    for i, (dec, roi, label, family) in enumerate(ROWS):
        codes = ['vonmises'] if family == 'vonmises' else ['lam0', 'lam1']
        for j, space in enumerate(('value', 'orientation')):
            ax = axes[i, j]
            g = emp[(emp.decoder == dec) & (emp.roi == roi) & (emp.space == space)]
            for mp, h in g.groupby('mapping'):
                h = h.sort_values('bin')
                ax.fill_between(h.bin, h.sd_lo, h.sd_hi, color=MAP_COL[mp], alpha=.2, lw=0)
                ax.plot(h.bin, h.sd, 'o-', color=MAP_COL[mp], lw=1.1, ms=2.2)
            for code in codes:
                k = f'{sim_kind}_{code}'
                if k not in gsrc:
                    continue
                s = gsrc[k]
                s = s[(s.decoder == dec) & (s.roi == roi) & (s.space == space)]
                for mp, h in s.groupby('mapping'):
                    h = h.sort_values('bin')
                    ax.plot(h.bin, h.sd, color=MAP_COL[mp], lw=.8, ls=SIM_LS[code])
            ax.set_title(label if j == 0 else '', fontsize=6.5, loc='left')
            ax.set_ylabel('SD of decoded\nvalue (CHF)' if j == 0 else 'SD of orientation\nerror (deg)',
                          fontsize=6)
            sns.despine(ax=ax, offset=2)
    axes[-1, 0].set_xlabel('True value (CHF)')
    axes[-1, 1].set_xlabel('Orientation (deg)')
    axes[-1, 0].set_xticks([2, 12, 22, 32, 42])
    axes[-1, 1].set_xticks([0, 45, 90, 135, 180])
    a = axes[0, 0]
    for kk, mp in enumerate(MAP_COL):
        a.text(.02, .97 - .15 * kk, MAP_LABEL[mp], color=MAP_COL[mp], fontsize=6,
               transform=a.transAxes, va='top')
    for ax, codes in ((axes[0, 1], ['vonmises']), (axes[3, 1], ['lam0', 'lam1'])):
        for kk, code in enumerate(codes):
            y = .95 - .15 * kk
            ax.plot([.42, .5], [y, y], transform=ax.transAxes, color='.3', lw=.8, ls=SIM_LS[code])
            ax.text(.52, y, SIM_LABEL[code], transform=ax.transAxes, fontsize=5.5, va='center',
                    color='.3')
    fig.suptitle(title, fontsize=7)
    letter(axes[0, 0], 'a')
    letter(axes[0, 1], 'b')
    fig.savefig(out)
    print(f'saved {out}')


def main(skip_sweep=False):
    src, ideal = load_sources()
    print('sources:', {k: v.subject.nunique() for k, v in src.items()})
    gsrc = {k: group(v) for k, v in src.items()}
    pd.concat([g.assign(source=k) for k, g in gsrc.items()]).to_csv(
        RES / 'precision_vs_value_group.tsv', sep='\t', index=False, float_format='%.4g')
    pd.concat([c.assign(source=k) for k, c in src.items()]).to_csv(
        RES / 'precision_vs_value_cells_by_subject.tsv.gz', sep='\t', index=False, float_format='%.4g')

    inter = []
    for k, c in src.items():
        for col in ('sd', 'abs', 'bias'):
            r = interaction_binned(c, col)
            r['source'] = k
            inter.append(r)
    inter = pd.concat(inter, ignore_index=True)
    inter.to_csv(RES / 'precision_vs_value_interaction.tsv', sep='\t', index=False,
                 float_format='%.4g')
    pd.set_option('display.width', 220)
    sdi = inter[inter.measure == 'sd']
    print(sdi.pivot_table(index=['decoder', 'roi'], columns=['source', 'space'],
                          values='rms').round(3).to_string())
    print(sdi.pivot_table(index=['decoder', 'roi'], columns=['source', 'space'],
                          values='p').round(3).to_string())

    cl = pd.concat([closeness(src, k, sp) for k in ('ideal', 'pipe')
                    for sp in ('value', 'orientation')], ignore_index=True)
    if len(cl):
        cl.to_csv(RES / 'precision_vs_value_closeness_by_subject.tsv', sep='\t', index=False,
                  float_format='%.4g')
        rows = []
        for (sim, sp, roi), g in cl.groupby(['sim', 'space', 'roi']):
            m, lo, hi, n, p = tci(g.delta)
            rows.append(dict(sim=sim, space=sp, roi=roi, delta=m, lo=lo, hi=hi, n=n, p=p,
                             n_closer_lam1=int((g.delta > 0).sum()),
                             d_lam0=g.d_lam0.mean(), d_lam1=g.d_lam1.mean(),
                             sim_lam0_vs_lam1=g.d_lam0_lam1.mean()))
        cls = pd.DataFrame(rows)
        cls.to_csv(RES / 'precision_vs_value_closeness.tsv', sep='\t', index=False,
                   float_format='%.4g')
        print(cls.round(3).to_string(index=False))

    if ideal is not None and not skip_sweep:
        dg = diagnosis(ideal)
        dg.to_csv(RES / 'precision_vs_value_noise_sweep.tsv', sep='\t', index=False,
                  float_format='%.4g')
        print(dg[dg.measure == 'post'].pivot_table(
            index=['code', 'roi', 'space'], columns='noise_scale', values='rms').round(3).to_string())
        print(dg.drop_duplicates(['code', 'roi', 'noise_scale']).pivot_table(
            index=['code', 'roi'], columns='noise_scale', values='mae_v').round(2).to_string())

    for kind, title in (('pipe', 'Data (CV-N low-rank) vs real pipeline on simulated data'),
                        ('ideal', 'Data (CV-N low-rank) vs ideal observer on simulated data')):
        if any(k.startswith(kind) for k in gsrc):
            figure(gsrc, kind, RES / f'fig_precision_vs_value_{kind}.pdf', title)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-sweep', action='store_true', help='skip the noise-scale diagnosis')
    main(ap.parse_args().skip_sweep)
