#!/usr/bin/env python3
"""Collect decode_lowrank_cv.py outputs into two tables (run on the cluster).

Reads derivatives/decoding_lowrankcv/{gabor,value}/sub-*/func/*_{trials,meta}.tsv
and writes, next to them,

    derivatives/decoding_lowrankcv/lowrank_trials.tsv.gz   one row per trial x N x noise
    derivatives/decoding_lowrankcv/lowrank_meta.tsv.gz     one row per fold x N (k, T, dof, CV)

    python analyses/followup_2026-10/lowrank_decoding/collect.py \
        --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue
"""
import argparse
from pathlib import Path

import pandas as pd


def main(bids_folder, tree='decoding_lowrankcv'):
    root = Path(bids_folder) / 'derivatives' / tree
    for kind in ('trials', 'meta'):
        frames = []
        for sub_dir, quantity in (('gabor', 'orientation'), ('value', 'value')):
            for fn in sorted((root / sub_dir).glob(f'sub-*/func/*_{kind}.tsv')):
                d = pd.read_csv(fn, sep='\t', dtype={'subject': str, 'n_voxels': str})
                d.insert(0, 'quantity', quantity)
                frames.append(d)
        out = pd.concat(frames, ignore_index=True)
        fn = root / f'lowrank_{kind}.tsv.gz'
        out.to_csv(fn, sep='\t', index=False, float_format='%.5g')
        n = out.groupby(['quantity', 'roi']).subject.nunique()
        print(f'{fn}: {len(out)} rows\n{n.to_string()}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-abstractvalue')
    p.add_argument('--tree', default='decoding_lowrankcv')
    a = p.parse_args()
    main(a.bids_folder, a.tree)
