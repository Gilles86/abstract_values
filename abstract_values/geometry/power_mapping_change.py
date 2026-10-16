"""Power of the crossnobis mapping-change test: would we see a code that re-scales?

The parietal RDMs do not change between mappings the way value distances do
(analyze_crossnobis: slope ~ 0). Before reading that as "the code does not
re-scale", ask whether the test could have seen it. Into each participant's
REAL single-trial betas (so real noise, real covariance, real geometry) we
inject a population code in value space: every voxel gets a Gaussian tuning
curve over CHF (preferred value uniform on 2-42, SD 8 CHF, random gain) and
each trial adds W . g(value of that trial in that session). Because the value
of an orientation differs between sessions, this code re-scales with the
mapping by construction.

Injection size is set relative to the region's own structure: the injected
code alone (after the same whitening) adds `frac` x the region's observed mean
crossnobis distance. frac = 0 is the real data (the null result itself).
Control: an ordered ORIENTATION code (tuning over orientation, same in both
sessions) -- it must not produce a mapping-change slope.

    python -m abstract_values.geometry.power_mapping_change 03
    -> derivatives/geometry/power/sub-<s>/sub-<s>_desc-mappingchange_power.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image, masking

from abstract_values.geometry.crossnobis import ROIS, crossnobis, trial_labels
from abstract_values.utils.data import BIDS_FOLDER, Subject

FRACS = [0.0, 0.1, 0.25, 0.5, 1.0]
TARGETS = ["IPS0", "IPS12", "IPS345", "BensonV1"]


def slope(lab, X):
    """Mapping-change slope: (d_cdf - d_inv) on (|dv|_cdf - |dv|_inv)."""
    out = {}
    for ses, g in lab.groupby("session"):
        idx = g.index.to_numpy()
        D, oris = crossnobis(X[idx], g.run.to_numpy(), g.orientation.to_numpy())
        val = g.groupby("orientation").value.first().reindex(oris).to_numpy()
        out[g.mapping.iloc[0]] = (D, np.abs(val[:, None] - val[None]) / 40)
    i, j = np.triu_indices(len(oris), 1)
    dd = out["cdf"][0][i, j] - out["inverse_cdf"][0][i, j]
    dv = out["cdf"][1][i, j] - out["inverse_cdf"][1][i, j]
    mean_d = np.mean([out[m][0][i, j].mean() for m in out])
    return change_slope(dd, dv, oris[i], oris[j]), mean_d


def change_slope(dd, dv, oi, oj):
    """Coefficient of the value-distance change, with the FIXED geometries as
    covariates. A session-wise gain change (SNR, whitening) multiplies the
    fixed orientation geometry, and linear orientation distance correlates
    with the mapping-induced change in value distance -- so without these
    covariates a plain gain difference between sessions shows up as a
    'mapping change'."""
    dt = np.abs(oi - oj)
    X = np.column_stack([dv, dt / 165, np.minimum(dt, 180 - dt) / 90, np.ones(len(dd))])
    return np.linalg.lstsq(X, dd, rcond=None)[0][0]


def main(subject, bids_folder=BIDS_FOLDER, seed=0, rois=TARGETS):
    sub = Subject(subject, bids_folder=bids_folder)
    lab = trial_labels(sub)
    betas = sub.get_single_trial_estimates(sorted(sub.get_sessions()))
    rng = np.random.default_rng(seed)
    rows = []
    for roi in rois:
        mask = image.resample_to_img(sub.get_roi_mask(roi, hemi=ROIS[roi]),
                                     image.index_img(betas, 0), interpolation="nearest")
        X = masking.apply_mask(betas, mask)
        X = X[:, np.all(np.isfinite(X), 0) & (X.std(0) > 0)]
        X = X - X.mean(0)
        p = X.shape[1]
        s0, d0 = slope(lab, X)
        for code, feat, centres, width in (
                ("value", lab.value.to_numpy(), (2, 42), 8.0),
                ("orientation", lab.orientation.to_numpy(), (0, 180), 36.0)):
            pref = rng.uniform(*centres, p)
            gain = rng.gamma(2, 0.5, p)
            diff = feat[:, None] - pref[None]           # orientation: ordered, not circular
            G = gain * np.exp(-0.5 * (diff / width) ** 2)
            G = (G - G.mean(0)) / X.std(0).mean()
            _, d1 = slope(lab, X + G)
            unit = max(d1 - d0, 1e-9)                 # added distance at amplitude 1
            for frac in FRACS:
                a = np.sqrt(frac * d0 / unit) if frac > 0 else 0.0
                s, d = slope(lab, X + a * G)
                rows.append(dict(subject=subject, roi=roi, code=code, frac=frac,
                                 amplitude=a, slope=s, mean_d=d, n_voxels=p))
                print(f"sub-{subject} {roi} {code} frac={frac}: slope {s:+.4f}")
    odir = Path(bids_folder) / "derivatives" / "geometry" / "power" / f"sub-{subject}"
    odir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(odir / f"sub-{subject}_desc-mappingchange_power.tsv",
                              sep="\t", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject")
    ap.add_argument("--bids-folder", default=BIDS_FOLDER)
    ap.add_argument("--rois", nargs="+", default=TARGETS)
    a = ap.parse_args()
    main(a.subject, a.bids_folder, rois=a.rois)
