"""Does any code retune between the two mappings? Fixed vs session-shift encoding models.

For each encoding-model family, the session-shift twin lets the preferred
value (or orientation) differ between the two sessions, with everything else
shared; the fixed model uses one tuning for both. Both are scored by
leave-one-run-out cvR² on the fsaverage surface. Per participant and region
(disjoint dorsal regions, as in the gradient figures), the mean of
cvR²(shift) − cvR²(fixed) over vertices where the fixed model beats the null.
Positive = retuning between mappings explains held-out data better.

  aPRF       log-Gaussian over value (aprf.cv vs aprf-shift.cv)
  wPRF       weighted log-Gaussian basis over value (aprf-weighted.cv vs
             aprf-weighted-shift.cv)
  von Mises  orientation tuning (vonmises.cv vs vonmises-shift.cv)

    python -m abstract_values.visualize.retuning_summary          # pycortex2 env
    -> notes/data/retuning_summary.tsv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize.figure_gradient import DORSAL, exclusive_rois, roi_verts
from abstract_values.visualize.group_surface_maps import discover_subjects, load_fsaverage

REPO = Path(__file__).resolve().parents[2]
FAMILIES = {"aPRF": ("aprf.cv", "aprf-shift.cv"),
            "wPRF": ("aprf-weighted.cv", "aprf-weighted-shift.cv"),
            "von Mises": ("vonmises.cv", "vonmises-shift.cv")}


def main():
    deriv = Path(BIDS_FOLDER) / "derivatives"
    verts = exclusive_rois(roi_verts())
    rows = []
    for s in discover_subjects(deriv):
        null = load_fsaverage(deriv, "aprf-null.cv", s, "cvr2")
        if null is None:
            continue
        for fam, (fixed, shift) in FAMILIES.items():
            a = load_fsaverage(deriv, fixed, s, "cvr2")
            b = load_fsaverage(deriv, shift, s, "cvr2")
            if a is None or b is None:
                continue
            for r in DORSAL:
                vx = verts[r]
                ok = np.isfinite(a[vx]) & np.isfinite(b[vx]) & (a[vx] > null[vx])
                if ok.sum() < 10:
                    continue
                rows.append(dict(subject=s, family=fam, roi=r, n_vertices=int(ok.sum()),
                                 fixed=float(np.mean(a[vx][ok] - null[vx][ok])),
                                 shift=float(np.mean(b[vx][ok] - null[vx][ok])),
                                 delta=float(np.mean(b[vx][ok] - a[vx][ok]))))
    d = pd.DataFrame(rows)
    d.to_csv(REPO / "notes/data/retuning_summary.tsv", sep="\t", index=False)
    s = d.groupby(["family", "roi"]).delta.agg(["mean", "sem", "count"]).mul(
        [100, 100, 1]).round(3)
    print(s.unstack(0).reindex(DORSAL).to_string())


if __name__ == "__main__":
    main()
