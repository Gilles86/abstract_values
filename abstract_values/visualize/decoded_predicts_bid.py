#!/usr/bin/env python3
"""Does the decoded value predict what the participant bid, beyond the truth?

Per subject and ROI, the partial correlation between the decoded value of a
trial (leave-one-run-out, spherical noise) and the bid on that trial, with the
objective value regressed out of both. A positive value means the brain signal
carries the trial-by-trial part of the bid -- the subjective component -- and
not just the stimulus.

Includes a shift control: the same partial correlation against the NEXT trial's
bid, which shares the subject's slow drifts but not the trial's percept.

Decoders: 50 voxels, 250 voxels (single log-Gaussian bump) and the basis-set
('weighted') decoder at 100 voxels; NPCr and V1.

Writes notes/data/decoded_predicts_bid.tsv and
notes/figures/decoded_predicts_bid.pdf.

    python -m abstract_values.visualize.decoded_predicts_bid
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.behavior.data import get_all_behavioral_data
from abstract_values.connective_fields.plot_gates import letter  # sets rcParams

REPO = Path(__file__).resolve().parents[2]
SETS = (('50 voxels', 'decoding_trials_nv50.tsv', None),
        ('250 voxels', 'decoding_trials_nv250.tsv', None),
        ('Basis set, 100 voxels', 'decoding_trials_weighted.tsv', 'weighted'))
ROI_COL = {'NPCr': '#C44E52', 'BensonV1': '#3B5BA5'}


def partial_r(x, y, z):
    """corr(x, y) with z (and an intercept) regressed out of both."""
    Z = np.c_[np.ones(len(z)), z]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def behaviour():
    b = get_all_behavioral_data()
    b = b[b.event_type == 'feedback'].copy()
    b['bid'] = pd.to_numeric(b['response'], errors='coerce')
    return b.reset_index()[['subject', 'session', 'run', 'trial_nr', 'bid']]


def main():
    beh = behaviour()
    rows = []
    for label, fn, model in SETS:
        d = pd.read_csv(REPO / 'notes' / 'data' / fn, sep='\t', dtype={'subject': str})
        d = d[d.quantity == 'value'] if 'quantity' in d else d[d.model == model]
        d['subject_n'] = [int(''.join(c for c in s if c.isdigit())) for s in d.subject]
        m = d.merge(beh, left_on=['subject_n', 'session', 'run', 'trial_nr'],
                    right_on=['subject', 'session', 'run', 'trial_nr'], suffixes=('', '_b'))
        m = m.sort_values(['subject', 'roi', 'session', 'run', 'trial_nr'])
        m['bid_next'] = m.groupby(['subject', 'roi', 'session', 'run']).bid.shift(-1)
        for (roi, s), h in m.groupby(['roi', 'subject']):
            k = h.dropna(subset=['bid'])
            kn = h.dropna(subset=['bid_next'])
            rows.append(dict(decoder=label, roi=roi, subject=s, n_trials=len(k),
                             same_trial=partial_r(k.post_mean.values, k.bid.values, k.true.values),
                             next_trial=partial_r(kn.post_mean.values, kn.bid_next.values,
                                                  kn.true.values)))
    res = pd.DataFrame(rows)
    res.to_csv(REPO / 'notes' / 'data' / 'decoded_predicts_bid.tsv', sep='\t', index=False)
    for (label, roi), g in res.groupby(['decoder', 'roi']):
        t = stats.ttest_1samp(g.same_trial, 0)
        tn = stats.ttest_1samp(g.next_trial, 0)
        print(f'{label:22s} {roi:9s} same trial r = {g.same_trial.mean():+.3f} ± '
              f'{g.same_trial.sem():.3f} (p = {t.pvalue:.2g}) | next trial '
              f'{g.next_trial.mean():+.3f} (p = {tn.pvalue:.2g})')
    for label, g in res.groupby('decoder'):
        w = g.pivot(index='subject', columns='roi', values='same_trial').dropna()
        t = stats.ttest_rel(w.NPCr, w.BensonV1)
        print(f'{label:22s} NPCr − V1: {(w.NPCr - w.BensonV1).mean():+.3f}, p = {t.pvalue:.2g}')

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, col, title in ((axes[0], 'same_trial', 'This trial'),
                           (axes[1], 'next_trial', 'Next trial (control)')):
        for i, (label, _, _) in enumerate(SETS):
            for j, roi in enumerate(ROI_COL):
                g = res[(res.decoder == label) & (res.roi == roi)]
                x = i + (j - .5) * .34
                ax.scatter(x + rng.uniform(-.06, .06, len(g)), g[col], s=5,
                           color=ROI_COL[roi], alpha=.35, lw=0)
                ax.errorbar(x, g[col].mean(), g[col].sem(), color='.15', lw=.9, zorder=3)
                ax.plot(x, g[col].mean(), 'D', ms=4.5, mfc=ROI_COL[roi], mec='.15', mew=1,
                        zorder=4)
        ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
        ax.set_xticks(range(len(SETS)), [s[0].replace(', ', ',\n') for s in SETS], fontsize=6)
        ax.set_ylabel('Partial r (decoded vs bid,\nobjective value removed)' if col == 'same_trial' else '')
        ax.set_title(title, fontsize=7.5)
        sns.despine(ax=ax, offset=3, trim=True)
    axes[0].text(.03, .97, 'NPCr', color=ROI_COL['NPCr'], transform=axes[0].transAxes, va='top')
    axes[0].text(.03, .87, 'V1', color=ROI_COL['BensonV1'], transform=axes[0].transAxes, va='top')
    letter(axes[0], 'a')
    letter(axes[1], 'b')
    out = REPO / 'notes' / 'figures' / 'decoded_predicts_bid.pdf'
    fig.savefig(out)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
