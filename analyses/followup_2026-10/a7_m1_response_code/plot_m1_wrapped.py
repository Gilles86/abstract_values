#!/usr/bin/env python3
"""Preferred value / bid of the wrapped PRF in M1 (fit_m1_wrapped.py output).

Seam = 0 = 42 CHF on the wrapped axis: a voxel tuned there responds most to
both extreme bids, i.e. tracks expected slider travel. Prediction for a
travel code: preferred values pile up at the seam in contralateral (left) M1,
flat in ipsilateral (right) M1.

Per (subject, epoch, space, hemi): density of preferred values over 7 bins
(6 CHF each, centred on the seam ... centre), and the seam fraction (|mu - seam|
< 6 CHF; 2/7 = 0.29 if flat). Group test: M1l - M1r seam fraction, and M1l
against 2/7, sign-flip / t on subjects.

    python plot_m1_wrapped.py --fits-dir /data/ds-abstractvalue/derivatives/followup_2026-10/a7_m1_response_code/fits
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 8,
    'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
COL = {'M1l': '#3B5BA5', 'M1r': '#9A9A9A'}
LAB = {'M1l': 'Left M1 (contra)', 'M1r': 'Right M1 (ipsi)'}
SEAM_W = 6.0
EDGES = np.arange(-3, 43, 6.0)          # 7 wrapped bins centred 0, 6, ..., 36


def wrap_bins(mu):
    x = np.mod(np.asarray(mu) + 3.0, 42.0) - 3.0       # [-3, 39)
    return np.histogram(x, bins=EDGES)[0]


def seam_dist(mu):
    return np.minimum(np.mod(mu, 42), 42 - np.mod(mu, 42))


def load(fits_dir):
    df = pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str})
                    for f in sorted(Path(fits_dir).glob('*_m1wrapped.tsv'))])
    df['above_null'] = df.cvr2_wrapped > df.cvr2_null
    return df


def densities(df, col, sel=None):
    rows = []
    for (s, ep, sp, h), g in df.groupby(['subject', 'epoch', 'space', 'hemi']):
        if sel is not None:
            g = g[g[sel]]
        if len(g) < 5:
            continue
        cnt = wrap_bins(g[col])
        for c, d in zip(EDGES[:-1] + 3, cnt / cnt.sum()):
            rows.append(dict(subject=s, epoch=ep, space=sp, hemi=h, centre=c, p=d))
    return pd.DataFrame(rows)


def seam_frac(df, col, sel=None):
    g = df if sel is None else df[df[sel]]
    out = (g.assign(seam=seam_dist(g[col]) < SEAM_W)    # 12 of 42 CHF: 2/7 if flat
           .groupby(['subject', 'epoch', 'space', 'hemi']).agg(
               seam=('seam', 'mean'), n=('seam', 'size'),
               dcv=('cvr2_wrapped', 'mean'), null=('cvr2_null', 'mean')).reset_index())
    out['dcv'] = out.dcv - out.null
    return out


def stats_table(sf):
    rows = []
    for (ep, sp), g in sf.groupby(['epoch', 'space']):
        w = g.pivot(index='subject', columns='hemi', values='seam').dropna()
        for name, d in (('M1l-M1r', w.M1l - w.M1r), ('M1l-flat', w.M1l - 2 / 7),
                        ('M1r-flat', w.M1r - 2 / 7)):
            t = stats.ttest_1samp(d, 0)
            rows.append(dict(epoch=ep, space=sp, contrast=name, mean=d.mean(),
                             sem=d.sem(), t=t.statistic, p=t.pvalue,
                             n_pos=int((d > 0).sum()), n=len(d)))
        c = g.pivot(index='subject', columns='hemi', values='dcv').dropna()
        for h in ('M1l', 'M1r'):
            t = stats.ttest_1samp(c[h], 0)
            rows.append(dict(epoch=ep, space=sp, contrast=f'cvr2-null {h}',
                             mean=c[h].mean(), sem=c[h].sem(), t=t.statistic,
                             p=t.pvalue, n_pos=int((c[h] > 0).sum()), n=len(c)))
    return pd.DataFrame(rows)


def panel(ax, dens, ep, sp, title):
    for h in ('M1r', 'M1l'):
        g = dens[(dens.epoch == ep) & (dens.space == sp) & (dens.hemi == h)]
        m = g.groupby('centre').p.agg(['mean', 'sem'])
        x = np.r_[m.index, 42.0]                           # repeat the seam bin at 42
        y, e = np.r_[m['mean'], m['mean'].iloc[0]], np.r_[m['sem'], m['sem'].iloc[0]]
        ax.fill_between(x, y - e, y + e, color=COL[h], alpha=.2, lw=0)
        ax.plot(x, y, color=COL[h], marker='o', ms=2.5)
    ax.axhline(1 / 7, color='k', lw=.6, ls=':')
    ax.set_xticks([0, 21, 42])
    ax.set_title(title, loc='left')


def main(fits_dir, out):
    df = load(fits_dir)
    print(df.groupby(['epoch', 'space', 'hemi']).subject.nunique())
    dens_all = densities(df, 'wrapped_mu_chf')
    dens_sel = densities(df, 'wrapped_mu_chf', sel='above_null')
    sf = seam_frac(df, 'wrapped_mu_chf')
    sf_sel = seam_frac(df, 'wrapped_mu_chf', sel='above_null')
    st = stats_table(sf).assign(voxels='all')
    st = pd.concat([st, stats_table(sf_sel).assign(voxels='cvR2 > null')])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    st.to_csv(out.with_suffix('.stats.tsv'), sep='\t', index=False)
    sf.to_csv(out.with_suffix('.subjects.tsv'), sep='\t', index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(st.round(4).to_string(index=False))

    fig, axes = plt.subplots(2, 4, figsize=(7.25, 3.4), constrained_layout=True,
                             sharey='row')
    for r, ep in enumerate(('response', 'gabor')):
        ep_lab = 'Response epoch' if ep == 'response' else 'Stimulus epoch'
        panel(axes[r, 0], dens_all, ep, 'bid', f'{ep_lab} · bid · all voxels')
        panel(axes[r, 1], dens_sel, ep, 'bid', 'Bid · cvR² > null')
        panel(axes[r, 2], dens_all, ep, 'value', 'Value · all voxels')
        panel(axes[r, 3], dens_sel, ep, 'value', 'Value · cvR² > null')
        axes[r, 0].set_ylabel('Fraction of voxels')
    for ax in axes[1]:
        ax.set_xlabel('Preferred CHF (0 = 42)')
    ax = axes[0, 0]
    for i, h in enumerate(('M1l', 'M1r')):
        ax.text(.04, .96 - i * .1, LAB[h], color=COL[h], transform=ax.transAxes,
                va='top', fontsize=6.5)
    ax.text(.04, .76, 'Flat', transform=ax.transAxes, va='top', fontsize=6.5)
    for ax in axes.flat:
        ax.spines['left'].set_position(('outward', 4))
        ax.spines['bottom'].set_position(('outward', 4))
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    print(f'saved {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fits-dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.fits_dir, a.out)
