"""Per-participant value-decoding accuracy per ROI -- cluster-side reduce.

Reads decode_value.py's per-trial likelihoods
(derivatives/decoding/<value|value-linear>/sub-*/func/
 sub-*_mask-<roi>_nvoxels-100_noise-spherical_lambda-0.1_pars.tsv)
and writes one row per (subject, roi, decoder): the within-run Pearson r
between presented CHF value and posterior-mean decoded value (flat prior),
averaged across runs -- the metric of compare_linear_vs_prf_decoding.py.

    python -m abstract_values.visualize.collect_roi_decoding_r \\
        --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue \\
        --out notes/data/vmpfc_decoding_r.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from abstract_values.visualize.compare_linear_vs_prf_decoding import _decode_metric

ROIS = ["vmPFCOFCaparc", "vmPFCOFC", "BensonV1", "IPS0", "IPS12", "IPS345"]
DECODERS = {"Tuned (aPRF)": "value", "Linear": "value-linear"}


def main(bids, out):
    root = Path(bids) / "derivatives" / "decoding"
    rows = []
    for dec, sub_dir in DECODERS.items():
        for fn in sorted((root / sub_dir).glob(
                "sub-*/func/sub-*_mask-*_nvoxels-100_noise-spherical_lambda-0.1_pars.tsv")):
            subject = fn.name.split("_")[0][4:]
            roi = fn.name.split("_mask-")[1].split("_")[0]
            if roi not in ROIS:
                continue
            df = pd.read_csv(fn, sep="\t", index_col=[0, 1, 2, 3])
            rows.append(dict(subject=subject, roi=roi, decoder=dec,
                             r=_decode_metric(df, "uniform"), n_trials=len(df)))
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)
    print(df.groupby(["roi", "decoder"]).r.agg(["mean", "sem", "count"]).round(3))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", required=True)
    p.add_argument("--out", default="notes/data/vmpfc_decoding_r.tsv")
    a = p.parse_args()
    main(a.bids_folder, a.out)
