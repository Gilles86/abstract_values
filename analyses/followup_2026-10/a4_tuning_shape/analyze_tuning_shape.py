#!/usr/bin/env python3
"""A4: group statistics + figures from the per-subject fit_tuning_shape outputs.

Reads   <bids>/derivatives/followup_2026-10/a4_tuning_shape/sub-*/  (rsync from cluster)
Writes  results/*.tsv and results/fig_*.{pdf,png} next to this script, and copies
        them into the derivatives folder.

    ~/mambaforge/envs/abstract_values/bin/python analyze_tuning_shape.py [--bids /data/ds-abstractvalue]
"""
import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

HERE = Path(__file__).parent
ROIS = [('BensonV1', 'V1'), ('BensonV2', 'V2'), ('BensonV3', 'V3'),
        ('IPS0', 'IPS0'), ('IPS12', 'IPS1–2'), ('IPS345', 'IPS3–5')]
TICKFS = 6.5
RL = dict(ROIS)
WARPS = ['-0.50', '+0.00', '+0.25', '+0.50', '+0.75']
BINS = ['near0', 'oblique', 'near90']
BIN_LABEL = {'near90': '90° ± 15°', 'near0': '0° ± 15°', 'oblique': 'Other'}
C_REF, C_GVM, C_WARP = '0.35', '#3B5BA5', '#C44E52'

mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.sans-serif': ['Helvetica', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'lines.linewidth': 1.2, 'lines.markersize': 4, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': .02})


def paired(x, label, **kw):
    """Mean, 95% t-CI, t, p, Wilcoxon p, n>0 for a vector of per-subject values."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(.975, n - 1) * se
    t, p = stats.ttest_1samp(x, 0)
    pw = stats.wilcoxon(x).pvalue if n > 5 and np.any(x != 0) else np.nan
    return dict(contrast=label, n=n, mean=m, ci_lo=m - h, ci_hi=m + h, t=t, p=p,
                p_wilcoxon=pw, n_positive=int((x > 0).sum()), **kw)


def load(root):
    vox, dec, idx = [], [], []
    for d in sorted(root.glob('sub-*')):
        s = d.name[4:]
        f = list(d.glob('*_desc-a4_voxels.tsv.gz'))
        if not f:
            print('incomplete', s)
            continue
        vox.append(pd.read_csv(f[0], sep='\t').assign(subject=s))
        dec.append(pd.read_csv(next(d.glob('*_desc-a4_decoding.tsv.gz')), sep='\t')
                   .assign(subject=s))
        idx.append(pd.read_csv(next(d.glob('*_desc-a4_index.tsv')), sep='\t')
                   .assign(subject=s))
    vox = pd.concat(vox)
    inc = [c for c in vox if c.startswith('in_')]
    vox[inc] = vox[inc].fillna(False).astype(bool)      # ROI mask missing (sub-36: no Wang IPS)
    return vox, pd.concat(dec), pd.concat(idx)


def summarise_cvr2(vox):
    models = [c[5:] for c in vox if c.startswith('cvr2_') and c != 'cvr2_null']
    rows, binrows, par = [], [], []
    for (s, roi), d in ((k, v) for r, _ in ROIS for k, v in
                        vox[vox[f'in_{r}']].assign(roi=r).groupby(['subject', 'roi'])):
        row = dict(subject=s, roi=roi, n_vox=len(d))
        for m in models:
            row[m] = (d[f'cvr2_{m}'] - d.cvr2_null).mean()
        sig = d[(d.cvr2_vm > d.cvr2_null) | (d['cvr2_basis_a+0.00'] > d.cvr2_null)]
        row['n_signal'] = len(sig)
        rows.append(row)
        for b in BINS:
            br = dict(subject=s, roi=roi, bin=b)
            for m in models:       # fraction of the null's held-out error removed
                br[m] = 1 - sig[f'sse_{b}_{m}'].sum() / sig[f'sse_{b}_null'].sum()
            binrows.append(br)
        if len(sig) >= 10:
            w = sig.cvr2_vm - sig.cvr2_null
            g = sig[w > 0]
            par.append(dict(subject=s, roi=roi, n=len(g),
                            median_log2_shape=np.median(np.log2(g.gvm_shape)),
                            frac_shape_lt1=(g.gvm_shape < 1).mean(),
                            median_warp_a=np.median(g.warp_a),
                            frac_warp_pos=(g.warp_a > 0).mean()))
    return pd.DataFrame(rows), pd.DataFrame(binrows), pd.DataFrame(par)


def summarise_decoding(dec):
    dec = dec.copy()
    deg = dec.true_deg.round(3)
    dec['bin'] = np.where((deg - 90).abs() <= 15, 'near90',
                          np.where((deg <= 15) | (deg >= 165), 'near0', 'oblique'))
    dec['abs_error'] = dec.error.abs()
    for lev in (.5, .8, .95):
        dec[f'cov{int(lev * 100)}'] = dec.hpd_rank < lev
    g = dec.groupby(['subject', 'roi', 'decoder', 'bin'])
    out = g[['abs_error', 'post_sd', 'logdens_true', 'cov50', 'cov80', 'cov95']].mean()
    out['n_trials'] = g.size()
    out = out.reset_index()
    allb = dec.groupby(['subject', 'roi', 'decoder'])[
        ['abs_error', 'post_sd', 'logdens_true', 'cov50', 'cov80', 'cov95']].mean()
    out = pd.concat([out, allb.reset_index().assign(bin='all')])
    out['cov_err'] = ((out.cov50 - .5).abs() + (out.cov80 - .8).abs()
                      + (out.cov95 - .95).abs()) / 3
    return out


def contrasts(cv, binned, decs, idx):
    rows = []
    pairs = [('gvm', 'vm'), ('warp', 'vm')] + [(f'basis_a{w}', 'basis_a+0.00')
                                                for w in WARPS if w != '+0.00']
    pairs += [('basis_a+0.00', 'vm')]
    for roi, _ in ROIS:
        c = cv[cv.roi == roi].set_index('subject')
        for a, b in pairs:
            rows.append(paired(c[a] - c[b], f'{a} - {b}', roi=roi, measure='cvR2 (all voxels)'))
        for bn in BINS:
            c = binned[(binned.roi == roi) & (binned.bin == bn)].set_index('subject')
            for a, b in pairs:
                rows.append(paired(c[a] - c[b], f'{a} - {b}', roi=roi,
                                   measure=f'error reduction {bn}'))
        # does the gain concentrate near 90? (near90 - oblique) of the difference
        c9 = binned[(binned.roi == roi) & (binned.bin == 'near90')].set_index('subject')
        co = binned[(binned.roi == roi) & (binned.bin == 'oblique')].set_index('subject')
        for a, b in pairs:
            rows.append(paired((c9[a] - c9[b]) - (co[a] - co[b]), f'{a} - {b}', roi=roi,
                               measure='error reduction near90 minus oblique'))
        for met in ('abs_error', 'logdens_true', 'cov_err', 'post_sd'):
            for bn in ['near90', 'all']:
                d = decs[(decs.roi == roi) & (decs.bin == bn)].pivot(
                    index='subject', columns='decoder', values=met)
                for a, b in [('gvm', 'vm'), ('warp', 'vm'),
                             ('basis_a+0.50', 'basis_a+0.00'),
                             ('basis_a-0.50', 'basis_a+0.00'), ('basis_a+0.00', 'vm')]:
                    if a in d and b in d:
                        rows.append(paired(d[a] - d[b], f'{a} - {b}', roi=roi,
                                           measure=f'decoding {met} {bn}'))
        for val in ('ci_90', 'ratio_90'):
            d = idx[idx.roi == roi].pivot(index='subject', columns='model', values=val)
            for a, b in [('gvm', 'vm'), ('warp', 'vm'), ('basis_a+0.50', 'basis_a+0.00'),
                         ('basis_a+0.75', 'basis_a+0.00'), ('basis_a+0.00', 'vm')]:
                rows.append(paired(d[a] - d[b], f'{a} - {b}', roi=roi,
                                   measure=f'model-implied {val}'))
            ref = 0 if val.startswith('ci') else 1
            d45 = idx[idx.roi == roi].pivot(index='subject', columns='model',
                                            values=val.replace('90', '45'))
            for m in ('vm', 'gvm', 'warp', 'basis_a+0.00', 'basis_a+0.50'):
                rows.append(paired(d[m] - ref, m, roi=roi, measure=f'{val} - {ref}'))
                rows.append(paired(d[m] - d45[m], m, roi=roi, measure=f'{val} - 45 deg'))
    return pd.DataFrame(rows)


def dot_ci(ax, x, vals, color, mfc, marker='o', jitter=True):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if jitter:
        ax.scatter(x + np.random.default_rng(0).uniform(-.06, .06, len(v)), v, s=4,
                   color=color, alpha=.25, lw=0, zorder=2)
    m = v.mean()
    h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    ax.errorbar(x, m, yerr=h, fmt=marker, color=color, mfc=mfc, mec=color, ms=4.5,
                mew=1.2, lw=1.2, capsize=0, zorder=4)


def fig_cvr2(cv, binned, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.5), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.3, 1.1, 1]))
    x = np.arange(len(ROIS))
    # a: single-PRF shape families vs von Mises
    ax = axes[0]
    ax.axhline(0, color='0.7', lw=.6, ls='--', zorder=0)
    for k, (m, col, lab) in enumerate((('gvm', C_GVM, 'Generalized vM'),
                                       ('warp', C_WARP, 'Cardinal-warped vM'))):
        for i, (r, _) in enumerate(ROIS):
            c = cv[cv.roi == r]
            dot_ci(ax, i + (k - .5) * .3, 100 * (c[m] - c.vm), col, col)
        ax.text(.02, .97 - k * .09, lab, color=col, transform=ax.transAxes, va='top')
    ax.set_xticks(x, [l for _, l in ROIS], fontsize=TICKFS)
    ax.set_ylabel('ΔcvR² vs single vM (pp)')
    # b: basis warp sweep vs uniform basis
    ax = axes[1]
    ax.axhline(0, color='0.7', lw=.6, ls='--', zorder=0)
    cols = sns.color_palette('mako', as_cmap=True)(np.linspace(.2, .8, len(ROIS)))
    wv = np.array([float(w) for w in WARPS])
    ends = []
    for i, (r, l) in enumerate(ROIS):
        c = cv[cv.roi == r]
        y = np.array([(100 * (c[f'basis_a{w}'] - c['basis_a+0.00'])).mean() for w in WARPS])
        ax.plot(wv, y, '-o', color=cols[i], ms=3)
        ends.append([y[-1], l, cols[i]])
    ends.sort(key=lambda e: e[0])                     # de-collide end labels
    lo, hi = ax.get_ylim()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + .065 * (hi - lo))
    for y, l, col in ends:
        ax.text(wv[-1] + .05, y, l, color=col, va='center')
    ax.set_xticks([-.5, 0, .25, .5, .75])
    ax.set_xlabel('Basis warp a (> 0: denser at cardinals)')
    ax.set_ylabel('ΔcvR² vs uniform basis (pp)')
    # c: where does the shape gain sit (binned, V1, error reduction)
    ax = axes[2]
    ax.axhline(0, color='0.7', lw=.6, ls='--', zorder=0)
    b = binned[binned.roi == 'BensonV1']
    for k, (m, ref, col) in enumerate((('gvm', 'vm', C_GVM), ('warp', 'vm', C_WARP),
                                       ('basis_a+0.50', 'basis_a+0.00', '0.3'))):
        for i, bn in enumerate(BINS):
            bb = b[b.bin == bn]
            dot_ci(ax, i + (k - 1) * .25, 100 * (bb[m] - bb[ref]), col,
                   col if k < 2 else 'white', jitter=False)
    for k, (lab, col, fc) in enumerate((('Generalized vM − vM', C_GVM, C_GVM),
                                        ('Warped vM − vM', C_WARP, C_WARP),
                                        ('Basis a = 0.5 − uniform', '0.3', 'white'))):
        yy = .97 - k * .085
        ax.plot(.04, yy, 'o', transform=ax.transAxes, color=col, mfc=fc, mec=col, ms=4,
                mew=1.2, clip_on=False)
        ax.text(.08, yy, lab, color=col, transform=ax.transAxes, va='center')
    ax.set_xticks(range(3), [BIN_LABEL[bn] for bn in BINS])
    ax.set_ylabel('Δ held-out error removed, V1 (pp)')
    for a, lt in zip(axes, 'abc'):
        a.text(-.18, 1.04, lt, transform=a.transAxes, fontweight='bold', fontsize=8)
    sns.despine(fig=fig, offset=4, trim=False)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig_a4_cvr2.{ext}')
    plt.close(fig)


def fig_params(par, out):
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.1), constrained_layout=True)
    for ax, col, lab, ref, c in ((axes[0], 'median_log2_shape', 'Median log₂ shape', 0, C_GVM),
                                 (axes[1], 'median_warp_a', 'Median warp a', 0, C_WARP)):
        ax.axhline(ref, color='0.7', lw=.6, ls='--', zorder=0)
        for i, (r, _) in enumerate(ROIS):
            dot_ci(ax, i, par[par.roi == r][col], c, c)
        ax.set_xticks(range(len(ROIS)), [l for _, l in ROIS], fontsize=TICKFS)
        ax.set_ylabel(lab)
    axes[0].text(.02, .03, 'Below 0: pointed peak', transform=axes[0].transAxes, color='0.4')
    axes[1].text(.02, .97, 'Above 0: narrower at cardinals', transform=axes[1].transAxes,
                 color='0.4', va='top')
    for a, lt in zip(axes, 'ab'):
        a.text(-.2, 1.04, lt, transform=a.transAxes, fontweight='bold', fontsize=8)
    sns.despine(fig=fig, offset=4)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig_a4_params.{ext}')
    plt.close(fig)


def fig_decoding(decs, out):
    dd = [('vm', 'Single vM', C_REF, 'white'), ('gvm', 'Generalized vM', C_GVM, C_GVM),
          ('warp', 'Warped vM', C_WARP, C_WARP), ('basis_a+0.00', 'Uniform basis', C_REF, C_REF),
          ('basis_a+0.50', 'Cardinal basis', '#5D8C3F', '#5D8C3F'),
          ('basis_a-0.50', 'Oblique basis', '#8172B2', '#8172B2')]
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.2), constrained_layout=True)
    for col, roi in enumerate(('BensonV1', 'IPS12')):
        for row, (met, lab) in enumerate((('abs_error', 'Absolute error (°)'),
                                          ('cov80', '80% HPD coverage'))):
            ax = axes[row, col]
            if met == 'cov80':
                ax.axhline(.8, color='0.7', lw=.6, ls='--', zorder=0)
                ax.set_ylim(0, 1)
                ax.text(2.45, .82, 'Nominal', color='0.5', ha='right', va='bottom')
            d = decs[decs.roi == roi]
            for k, (dn, dl, c, fc) in enumerate(dd):
                for i, bn in enumerate(BINS):
                    dot_ci(ax, i + (k - 2.5) * .12, d[(d.decoder == dn) & (d.bin == bn)][met],
                           c, fc, jitter=False)
                if row == 1 and col == 0:               # key in the empty upper band
                    kx, ky = .05 + (k // 3) * .45, .68 - (k % 3) * .075
                    ax.plot(kx, ky, 'o', transform=ax.transAxes, color=c, mfc=fc, mec=c,
                            ms=4.5, mew=1.2, clip_on=False)
                    ax.text(kx + .03, ky, dl, color=c, transform=ax.transAxes, va='center')
            ax.set_xticks(range(3), [BIN_LABEL[b] for b in BINS])
            ax.set_ylabel(lab)
            if row == 0:
                ax.set_title(RL[roi])
    for a, lt in zip(axes.ravel(), 'abcd'):
        a.text(-.14, 1.04, lt, transform=a.transAxes, fontweight='bold', fontsize=8)
    sns.despine(fig=fig, offset=4)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig_a4_decoding.{ext}')
    plt.close(fig)


def fig_index(idx, out):
    mm = [('vm', 'Single vM', C_REF, 'white'), ('gvm', 'Generalized vM', C_GVM, C_GVM),
          ('warp', 'Warped vM', C_WARP, C_WARP), ('basis_a+0.00', 'Uniform basis', C_REF, C_REF),
          ('basis_a+0.50', 'Cardinal basis', '#5D8C3F', '#5D8C3F')]
    fig, ax = plt.subplots(figsize=(5.4, 2.2), constrained_layout=True)
    ax.axhline(0, color='0.7', lw=.6, ls='--', zorder=0)
    for k, (m, lab, c, fc) in enumerate(mm):
        for i, (r, _) in enumerate(ROIS):
            v = idx[(idx.roi == r) & (idx.model == m)].ci_90
            dot_ci(ax, i + (k - 2) * .14, v, c, fc, jitter=False)
        yy = .95 - k * .1
        ax.plot(1.04, yy, 'o', transform=ax.transAxes, color=c, mfc=fc, mec=c, ms=4.5,
                mew=1.2, clip_on=False)
        ax.text(1.08, yy, lab, color=c, transform=ax.transAxes, va='center')
    ax.set_xticks(range(len(ROIS)), [l for _, l in ROIS], fontsize=TICKFS)
    ax.set_ylabel('Model-implied 90° index')
    sns.despine(fig=fig, offset=4)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig_a4_index.{ext}')
    plt.close(fig)


def main(bids):
    root = Path(bids) / 'derivatives' / 'followup_2026-10' / 'a4_tuning_shape'
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    vox, dec, idx = load(root)
    print(vox.subject.nunique(), 'subjects')
    cv, binned, par = summarise_cvr2(vox)
    decs = summarise_decoding(dec)
    con = contrasts(cv, binned, decs, idx)
    for name, df in (('cvr2_by_subject_roi', cv), ('binned_error_by_subject_roi', binned),
                     ('shape_params_by_subject_roi', par), ('decoding_by_subject_roi', decs),
                     ('index_by_subject_roi', idx), ('contrasts', con)):
        df.to_csv(out / f'a4_{name}.tsv', sep='\t', index=False)
    fig_cvr2(cv, binned, out)
    fig_params(par, out)
    fig_decoding(decs, out)
    fig_index(idx, out)
    for f in out.iterdir():
        shutil.copy(f, root / f.name)
    print('wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids', default='/data/ds-abstractvalue')
    main(ap.parse_args().bids)
