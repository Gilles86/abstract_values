#!/usr/bin/env python3
"""Does NPCr track the value the participant REPORTED rather than the true one?

Two leave-one-run-out aPRF fits per subject, on exactly the same trials (those
with a usable bid) and in the same mask (NPCr + the stimulated V1 band):

    aprf-bid.cv          stimulus = the participant's bid on that trial
    aprf-bidmatched.cv   stimulus = the objective CHF value

Prediction: in NPCr the reported value should fit at least as well as the
objective one, because the bid is what the subjective quantity produced; in V1,
which only ever sees the gabor, the objective value (a deterministic function of
orientation) should win.

Per subject and ROI: mean cvR2 for each model, their difference, and the
fraction of voxels where the bid model wins. Signal voxels = either model beats
the per-voxel null model (aprf-null.cv); the ROI mean over all voxels is
reported too.

Writes notes/data/bid_vs_objective_aprf.tsv and
notes/figures/bid_vs_objective_aprf.pdf.

    python -m abstract_values.visualize.bid_vs_objective_aprf
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.maskers import NiftiMasker
from scipy import stats

from abstract_values.connective_fields.plot_gates import letter  # sets rcParams
from abstract_values.utils.data import BIDS_FOLDER, Subject

REPO = Path(__file__).resolve().parents[2]
ENC = BIDS_FOLDER / 'derivatives' / 'encoding_models'
ROIS = {'NPCr': 'NPCr', 'V1': 'BensonV1ecc075-375'}
BID_C, OBJ_C = '#C44E52', '#3B5BA5'


def cvr2_path(model, subject):
    return (ENC / model / f'sub-{subject}' / 'func'
            / f'sub-{subject}_task-abstractvalue_space-T1w_desc-cvr2_pe.nii.gz')


def main():
    subs = sorted(p.name[4:] for p in (ENC / 'aprf-bid.cv').glob('sub-*'))
    rows = []
    for s in subs:
        paths = {m: cvr2_path(m, s) for m in ('aprf-bid.cv', 'aprf-bidmatched.cv',
                                              'aprf-null.cv')}
        if not all(p.exists() for p in paths.values()):
            print(f'  skip sub-{s}: missing {[m for m, p in paths.items() if not p.exists()]}')
            continue
        sub = Subject(s)
        for roi, desc in ROIS.items():
            masker = NiftiMasker(mask_img=sub.get_roi_mask(desc, hemi=None)).fit()
            v = {m: masker.transform(str(p)).ravel() for m, p in paths.items()}
            ok = np.isfinite(v['aprf-bid.cv']) & np.isfinite(v['aprf-bidmatched.cv'])
            best = np.maximum(v['aprf-bid.cv'], v['aprf-bidmatched.cv'])
            sig = ok & np.isfinite(v['aprf-null.cv']) & (best > v['aprf-null.cv'])
            for sel, tag in ((ok, 'all voxels'), (sig, 'signal voxels')):
                if sel.sum() < 10:
                    continue
                rows.append(dict(subject=s, roi=roi, selection=tag, n=int(sel.sum()),
                                 bid=v['aprf-bid.cv'][sel].mean(),
                                 objective=v['aprf-bidmatched.cv'][sel].mean(),
                                 delta=(v['aprf-bid.cv'] - v['aprf-bidmatched.cv'])[sel].mean(),
                                 frac_bid_wins=float((v['aprf-bid.cv'][sel]
                                                      > v['aprf-bidmatched.cv'][sel]).mean())))
    res = pd.DataFrame(rows)
    res.to_csv(REPO / 'notes' / 'data' / 'bid_vs_objective_aprf.tsv', sep='\t', index=False)
    print(f'n = {res.subject.nunique()} subjects')
    for (roi, sel), g in res.groupby(['roi', 'selection']):
        t = stats.ttest_1samp(g.delta, 0)
        print(f'  {roi:5s} {sel:14s} cvR2 bid {g.bid.mean():+.4f}  objective '
              f'{g.objective.mean():+.4f}  Δ {g.delta.mean():+.4f} ± {g.delta.sem():.4f} '
              f'(t = {t.statistic:+.2f}, p = {t.pvalue:.2g}); bid wins in '
              f'{g.frac_bid_wins.mean():.0%} of voxels')
    for sel, g in res.groupby('selection'):
        w = g.pivot(index='subject', columns='roi', values='delta').dropna()
        t = stats.ttest_rel(w.NPCr, w.V1)
        print(f'  NPCr − V1 ({sel}): {(w.NPCr - w.V1).mean():+.4f}, t = {t.statistic:+.2f}, '
              f'p = {t.pvalue:.2g}')

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.4), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.3, 1, 1]})
    rng = np.random.default_rng(0)
    g = res[res.selection == 'signal voxels']
    ax = axes[0]
    for i, roi in enumerate(ROIS):
        h = g[g.roi == roi]
        for j, (col, c) in enumerate((('objective', OBJ_C), ('bid', BID_C))):
            x = i + (j - .5) * .34
            ax.scatter(x + rng.uniform(-.06, .06, len(h)), h[col], s=6, color=c, alpha=.4, lw=0)
            ax.errorbar(x, h[col].mean(), h[col].sem(), color='.15', lw=.9, zorder=3)
            ax.plot(x, h[col].mean(), 'D', ms=5, mfc=c, mec='.15', mew=1, zorder=4)
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks(range(len(ROIS)), list(ROIS))
    ax.set_ylabel('cvR² (signal voxels)')
    ax.text(.03, .98, 'Reported value (bid)', color=BID_C, transform=ax.transAxes, va='top')
    ax.text(.03, .89, 'Objective value', color=OBJ_C, transform=ax.transAxes, va='top')
    letter(ax, 'a')

    ax = axes[1]
    for i, roi in enumerate(ROIS):
        h = g[g.roi == roi]
        ax.scatter(i + rng.uniform(-.08, .08, len(h)), h.delta, s=6, color=BID_C, alpha=.4, lw=0)
        ax.errorbar(i, h.delta.mean(), h.delta.sem(), color='.15', lw=.9, zorder=3)
        ax.plot(i, h.delta.mean(), 'D', ms=5, mfc=BID_C, mec='.15', mew=1, zorder=4)
        p = stats.ttest_1samp(h.delta, 0).pvalue
        ax.text(i, ax.get_ylim()[1], f'p = {p:.1g}'.replace('0.', '.'), ha='center',
                va='bottom', fontsize=6, color='.3')
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks(range(len(ROIS)), list(ROIS))
    ax.set_xlim(-.6, len(ROIS) - .4)
    ax.set_ylabel('Δ cvR² (bid − objective)')
    letter(ax, 'b')

    ax = axes[2]
    for i, roi in enumerate(ROIS):
        h = g[g.roi == roi]
        ax.scatter(i + rng.uniform(-.08, .08, len(h)), h.frac_bid_wins, s=6, color=BID_C,
                   alpha=.4, lw=0)
        ax.errorbar(i, h.frac_bid_wins.mean(), h.frac_bid_wins.sem(), color='.15', lw=.9, zorder=3)
        ax.plot(i, h.frac_bid_wins.mean(), 'D', ms=5, mfc=BID_C, mec='.15', mew=1, zorder=4)
    ax.axhline(.5, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks(range(len(ROIS)), list(ROIS))
    ax.set_xlim(-.6, len(ROIS) - .4)
    ax.set_ylabel('Voxels where the bid model wins')
    letter(ax, 'c')
    for a in axes:
        sns.despine(ax=a, offset=3, trim=True)
    out = REPO / 'notes' / 'figures' / 'bid_vs_objective_aprf.pdf'
    fig.savefig(out)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
