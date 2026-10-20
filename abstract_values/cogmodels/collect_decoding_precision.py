"""Per-subject decoding precision for each (quantity, ROI) decoder, from the trial-wise
decoding posteriors in derivatives/decoding/{gabor,value}/.

One row per subject x decoder: number of decoded trials, fidelity (orientation:
resultant length of the axial decoding error; value: Pearson r of true vs posterior
mean), mean absolute error (deg / CHF) and mean posterior SD. All decoders are
leave-one-run-out, spherical noise, lambda 0.1.

Run on the cluster (reads ~thousands of pars files):

    python -m abstract_values.cogmodels.collect_decoding_precision \
        --out notes/data/decoding_precision_subjects.tsv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DECODERS = [  # (quantity dir, roi, n voxels)
    ('gabor', 'BensonV1', 250),
    ('gabor', 'BensonV1', 100),
    ('gabor', 'IPS0', 100),
    ('value', 'IPS0', 100),
    ('value', 'IPS12', 100),
    ('value', 'IPS345', 100),
    ('value', 'NPCr', 100),
    ('value', 'BensonV1', 100),
]


def summarise(pars, quantity):
    grid = pars.columns[4:].astype(float).values
    p = pars.iloc[:, 4:].values.astype(float)
    p = p / p.sum(1, keepdims=True)
    true = pars.iloc[:, 3].values.astype(float)
    if quantity == 'gabor':  # axial: work on the doubled angle
        z = p @ np.exp(2j * grid)
        mean = np.angle(z) / 2
        post_sd = np.rad2deg(np.sqrt(-2 * np.log(np.abs(z))) / 2)
        err = np.angle(np.exp(2j * (mean - true))) / 2
        fidelity = np.abs(np.exp(2j * err).mean())
        mae = np.rad2deg(np.abs(err)).mean()
    else:
        mean = p @ grid
        post_sd = np.sqrt(p @ grid ** 2 - mean ** 2)
        fidelity = np.corrcoef(true, mean)[0, 1]
        mae = np.abs(mean - true).mean()
    return dict(n_trials=len(true), fidelity=fidelity, mae=mae, post_sd=post_sd.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decoding-dir',
                    default='/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/decoding')
    ap.add_argument('--out', default='notes/data/decoding_precision_subjects.tsv')
    a = ap.parse_args()

    rows = []
    for quantity, roi, nv in DECODERS:
        pattern = f'sub-*/func/sub-*_mask-{roi}_nvoxels-{nv}_noise-spherical_lambda-0.1_pars.tsv'
        for f in sorted(Path(a.decoding_dir, quantity).glob(pattern)):
            subject = f.name.split('_')[0][4:]
            rows.append(dict(subject=subject, quantity=quantity, roi=roi, n_voxels=nv,
                             **summarise(pd.read_csv(f, sep='\t'), quantity)))
    out = pd.DataFrame(rows)
    out.to_csv(a.out, sep='\t', index=False)
    print(out.groupby(['quantity', 'roi', 'n_voxels'])[['fidelity', 'mae', 'post_sd']]
             .agg(['count', 'median']).round(3).to_string())


if __name__ == '__main__':
    main()
