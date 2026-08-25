"""Per-subject NPCr cross-condition preferred-value shift.

Reduces the per-voxel (mode_cdf, mode_invcdf) table that
``shifted_preferred_value.collect_subject`` builds -- voxels gated by the
out-of-sample criterion cvR2(session-shift) > cvR2(null) -- to one row per
subject, so ``brain_behavior_summary.py`` can relate it to behaviour.

Run on the cluster (it reads the session-shift parameter volumes), then rsync
the TSV back:

    ssh sciencecluster 'cd ~/git/abstract_values && \
        BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue srun -c2 --mem 24G \
        --time 40 --account=zne.uzh python -u -m \
        abstract_values.visualize.aggregate_condition_shift \
        --subjects 03 ... 28 --out notes/data/npcr_condition_shift.tsv'
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from abstract_values.visualize.shifted_preferred_value import collect_subject


def aggregate(subjects, roi="NPCr", smoothed=False):
    rows = []
    for s in subjects:
        try:
            d = collect_subject(s, roi=roi, smoothed=smoothed)
        except Exception as e:                       # missing fits / null cv
            print(f"  sub-{s}: {type(e).__name__}: {e}")
            continue
        if d is None or len(d) < 10:
            print(f"  sub-{s}: {0 if d is None else len(d)} voxels, skipping")
            continue
        shift = d["mode_invcdf"].values - d["mode_cdf"].values
        r = stats.pearsonr(d["mode_cdf"], d["mode_invcdf"])[0]
        rows.append(dict(subject=int(s), n_shift_vox=len(d),
                         mean_shift=float(np.mean(shift)),
                         mean_abs_shift=float(np.mean(np.abs(shift))),
                         median_abs_shift=float(np.median(np.abs(shift))),
                         mode_r=float(r)))
        print(f"  sub-{s}: {len(d):5d} vox  "
              f"mean|shift|={rows[-1]['mean_abs_shift']:.2f} CHF  r={r:+.3f}")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--roi", default="NPCr")
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--out", default="notes/data/npcr_condition_shift.tsv")
    a = p.parse_args()
    df = aggregate(a.subjects, roi=a.roi, smoothed=a.smoothed)
    df.to_csv(a.out, sep="\t", index=False)
    print(f"wrote {a.out}  ({len(df)} subjects)")


if __name__ == "__main__":
    main()
