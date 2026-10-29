#!/usr/bin/env python3
"""Per-participant metrics of the decode_jehee_cv.py arms (spherical, jehee at
N = 50, 250, whole ROI), same definitions as analyze.py.

Input: derivatives/decoding_jeheecv/lowrank_trials.tsv.gz (collect.py --tree
decoding_jeheecv), rsynced to results/jehee_trials.tsv.gz.

    python analyses/followup_2026-10/lowrank_decoding/jehee_metrics.py
"""
import pandas as pd

from analyze import RES, metrics

tr = pd.read_csv(RES / 'jehee_trials.tsv.gz', sep='\t', dtype={'subject': str, 'n_voxels': str})
per = (tr.groupby(['quantity', 'roi', 'n_voxels', 'noise', 'subject'])
       .apply(lambda g: metrics(g, g.name[0])).reset_index())
per.to_csv(RES / 'jehee_metrics_by_subject.tsv', sep='\t', index=False, float_format='%.5g')
print(per.groupby(['quantity', 'roi', 'n_voxels', 'noise'])[['r', 'mae', 'cov80']].mean().round(3))
