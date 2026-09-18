#!/usr/bin/env python3
"""Stimulus epoch vs report epoch: is V1 stimulus-driven and NPCr value-carrying?

Within a session, value and orientation are one-to-one, so the gabor epoch
cannot separate a stimulus code from a value code. The report epoch can: the
gabor is gone, the participant is moving a slider to say what it was worth. A
representation of the stimulus should fade; a representation of the value the
participant is about to report need not.

Per subject, ROI (NPCr, the stimulated V1 band) and epoch, cross-validated
cvR2 of two one-bump encoding models, minus the per-voxel null model fitted on
the same betas:

    value        aprf.cv          (log-Gaussian over CHF)
    orientation  vonmises-prf.cv  (axial von Mises over gabor orientation)

Gabor-epoch fits are the existing whole-brain runs; report-epoch fits are the
``--desc response`` runs in the NPCr + V1 mask.

    python -m abstract_values.visualize.epoch_dissociation
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
EPOCHS = {'Gabor': '', 'Report': '-response'}
MODELS = {'value': 'aprf.cv', 'orientation': 'vonmises-prf.cv'}
VAL_C, ORI_C = '#C44E52', '#3B5BA5'


def cvr2(model, subject, suffix, masker):
    f = (ENC / f'{model}{suffix}' / f'sub-{subject}' / 'func'
         / f'sub-{subject}_task-abstractvalue_space-T1w_desc-cvr2_pe.nii.gz')
    return masker.transform(str(f)).ravel() if f.exists() else None


def main():
    subs = sorted(p.name[4:] for p in (ENC / 'aprf.cv-response').glob('sub-*'))
    rows = []
    for s in subs:
        sub = Subject(s)
        for roi, desc in ROIS.items():
            masker = NiftiMasker(mask_img=sub.get_roi_mask(desc, hemi=None)).fit()
            for epoch, suffix in EPOCHS.items():
                null = cvr2('aprf-null.cv', s, suffix, masker)
                if null is None:
                    continue
                for model, sub_dir in MODELS.items():
                    v = cvr2(sub_dir, s, suffix, masker)
                    if v is None:
                        continue
                    ok = np.isfinite(v) & np.isfinite(null)
                    rows.append(dict(subject=s, roi=roi, epoch=epoch, model=model,
                                     n=int(ok.sum()), delta=(v - null)[ok].mean(),
                                     frac=float((v > null)[ok].mean())))
    res = pd.DataFrame(rows)
    res.to_csv(REPO / 'notes' / 'data' / 'epoch_dissociation.tsv', sep='\t', index=False)
    print(f'n = {res.subject.nunique()} subjects')
    for (roi, epoch, model), g in res.groupby(['roi', 'epoch', 'model']):
        t = stats.ttest_1samp(g.delta, 0)
        print(f'  {roi:5s} {epoch:7s} {model:12s} cvR² − null = {g.delta.mean():+.4f} ± '
              f'{g.delta.sem():.4f} (t = {t.statistic:+.2f}, p = {t.pvalue:.2g}); '
              f'{g.frac.mean():.0%} of voxels above null')
    w = res.pivot_table(index='subject', columns=['roi', 'epoch', 'model'], values='delta')
    for roi in ROIS:
        for epoch in EPOCHS:
            d = w[(roi, epoch, 'value')] - w[(roi, epoch, 'orientation')]
            t = stats.ttest_1samp(d.dropna(), 0)
            print(f'  {roi:5s} {epoch:7s} value − orientation = {d.mean():+.4f} '
                  f'(t = {t.statistic:+.2f}, p = {t.pvalue:.2g})')
    inter = ((w[('NPCr', 'Report', 'value')] - w[('NPCr', 'Report', 'orientation')])
             - (w[('V1', 'Report', 'value')] - w[('V1', 'Report', 'orientation')])).dropna()
    t = stats.ttest_1samp(inter, 0)
    print(f'  report epoch, (value − orientation) NPCr − V1 = {inter.mean():+.4f} '
          f'(t = {t.statistic:+.2f}, p = {t.pvalue:.2g})')

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4), constrained_layout=True, sharey=True)
    rng = np.random.default_rng(0)
    for ax, epoch in zip(axes, EPOCHS):
        for i, roi in enumerate(ROIS):
            for j, (model, c) in enumerate((('value', VAL_C), ('orientation', ORI_C))):
                g = res[(res.roi == roi) & (res.epoch == epoch) & (res.model == model)]
                x = i + (j - .5) * .34
                ax.scatter(x + rng.uniform(-.06, .06, len(g)), g.delta, s=5, color=c,
                           alpha=.35, lw=0)
                ax.errorbar(x, g.delta.mean(), g.delta.sem(), color='.15', lw=.9, zorder=3)
                ax.plot(x, g.delta.mean(), 'D', ms=4.5, mfc=c, mec='.15', mew=1, zorder=4)
        ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
        ax.set_xticks(range(len(ROIS)), list(ROIS))
        ax.set_xlim(-.6, len(ROIS) - .4)
        ax.set_title(f'{epoch} epoch', fontsize=7.5)
        sns.despine(ax=ax, offset=3, trim=True)
    axes[0].set_ylabel('cvR² above the null model')
    axes[0].text(.03, .97, 'Value tuning', color=VAL_C, transform=axes[0].transAxes, va='top')
    axes[0].text(.03, .87, 'Orientation tuning', color=ORI_C, transform=axes[0].transAxes,
                 va='top')
    letter(axes[0], 'a')
    letter(axes[1], 'b')
    out = REPO / 'notes' / 'figures' / 'epoch_dissociation.pdf'
    fig.savefig(out)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
