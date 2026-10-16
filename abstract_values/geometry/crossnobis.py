"""Cross-validated Mahalanobis (crossnobis) distances between the 23 orientations.

The design makes this unusually clean: both sessions show the SAME 23
orientations (7.5..172.5 deg), each once per run, 8 runs per session -- only the
orientation -> CHF mapping differs between sessions. So per (subject, ROI,
session) we get a 23 x 23 representational distance matrix (RDM) that

  * takes the noise covariance into account: patterns are whitened with a
    Ledoit-Wolf-shrunk voxel covariance estimated, fold by fold, from the
    TRAINING runs' residuals
    (pattern minus the condition's mean over runs), i.e. the distance is the
    squared Mahalanobis distance = linear Fisher discriminability (LDA d'^2)
    under a shared Gaussian noise model;
  * is unbiased: the difference vector from one run is multiplied with the
    mean difference vector of the OTHER runs, so noise contributes zero in
    expectation (a distance can be < 0 when there is no signal).

For neighbouring orientations, d / delta^2 estimates the linear Fisher
information of the population code at that point -- the quantity efficient
coding makes predictions about. Distances are per voxel (divided by the number
of voxels) so ROIs of different size are comparable.

What it does NOT capture: information only a nonlinear readout can use, and
stimulus-dependent noise covariance (one shared covariance per session).

    python -m abstract_values.geometry.crossnobis 03 [--rois ...] [--smoothed]
    -> derivatives/geometry/crossnobis/sub-<s>/sub-<s>_desc-crossnobis_rdm.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image, masking
from sklearn.covariance import LedoitWolf

from abstract_values.utils.data import BIDS_FOLDER, Subject

ROIS = {"BensonV1": "LR", "BensonV2": "LR", "BensonV3": "LR", "V3ab": None,
        "IPS0": None, "IPS12": None, "IPS345": None, "NPCr": None}


def trial_labels(sub):
    """One row per GLMsingle gabor beta, in the beta file's order
    (session -> run -> gabor events sorted by onset)."""
    rows = []
    for ses in sorted(sub.get_sessions()):
        mapping = sub.get_mapping(ses)
        runs = sub.get_runs(ses)
        ev = sub.get_events(ses, runs)
        for run in runs:
            g = ev.loc[run].reset_index().sort_values("onset")
            g = g[g.event_type == "gabor"]
            for r in g.itertuples():
                rows.append(dict(session=ses, mapping=mapping, run=run,
                                 orientation=float(r.orientation), value=float(r.value)))
    return pd.DataFrame(rows)


def crossnobis(B, runs, conds):
    """B: (trials, voxels). Returns (cond x cond) crossnobis RDM per voxel.

    Each (run, cond) must occur exactly once."""
    ur, uc = np.unique(runs), np.unique(conds)
    P = np.full((len(ur), len(uc), B.shape[1]), np.nan)
    for i, r in enumerate(ur):
        for j, c in enumerate(uc):
            sel = (runs == r) & (conds == c)
            if sel.sum() != 1:
                raise ValueError(f"run {r} cond {c}: {sel.sum()} trials")
            P[i, j] = B[sel][0]
    D = np.zeros((len(uc), len(uc)))
    for k in range(len(ur)):
        # The noise covariance comes from the training runs only: whitening
        # with a covariance that has seen run k makes run k's noise correlate
        # with the training mean and biases every distance upward.
        train = np.delete(P, k, 0)
        resid = (train - train.mean(0, keepdims=True)).reshape(-1, B.shape[1])
        w, V = np.linalg.eigh(LedoitWolf().fit(resid).covariance_)
        W = V @ np.diag(1 / np.sqrt(np.maximum(w, w.max() * 1e-8))) @ V.T
        G = (P[k] @ W) @ (train.mean(0) @ W).T
        dg = np.diag(G)
        D += dg[:, None] + dg[None, :] - G - G.T
    return D / len(ur) / B.shape[1], uc


def main(subject, rois, smoothed=False, bids_folder=BIDS_FOLDER):
    sub = Subject(subject, bids_folder=bids_folder)
    lab = trial_labels(sub)
    betas = sub.get_single_trial_estimates(sorted(sub.get_sessions()), smoothed=smoothed)
    if betas.shape[-1] != len(lab):
        raise ValueError(f"{betas.shape[-1]} betas vs {len(lab)} labelled trials")
    out = []
    for roi in rois:
        try:
            mask = sub.get_roi_mask(roi, hemi=ROIS.get(roi, None))
        except FileNotFoundError as e:
            print(f"skip {roi}: {e}")
            continue
        mask = image.resample_to_img(mask, image.index_img(betas, 0),
                                     interpolation="nearest")
        X = masking.apply_mask(betas, mask)               # (trials, voxels)
        X = X[:, np.all(np.isfinite(X), 0) & (X.std(0) > 0)]
        for ses, g in lab.groupby("session"):
            idx = g.index.to_numpy()
            D, oris = crossnobis(X[idx], g.run.to_numpy(), g.orientation.to_numpy())
            val = g.groupby("orientation").value.first().reindex(oris).to_numpy()
            i, j = np.triu_indices(len(oris), 1)
            out.append(pd.DataFrame(dict(
                subject=subject, roi=roi, session=ses, mapping=g.mapping.iloc[0],
                n_voxels=X.shape[1], ori_i=oris[i], ori_j=oris[j], val_i=val[i],
                val_j=val[j], d=D[i, j])))
            print(f"sub-{subject} {roi} ses-{ses}: {X.shape[1]} voxels, "
                  f"mean d = {D[i, j].mean():.4f}")
    res = pd.concat(out)
    odir = Path(bids_folder) / "derivatives" / "geometry" / "crossnobis" / f"sub-{subject}"
    odir.mkdir(parents=True, exist_ok=True)
    tag = "_smoothed" if smoothed else ""
    fn = odir / f"sub-{subject}_desc-crossnobis{tag}_rdm.tsv"
    res.to_csv(fn, sep="\t", index=False)
    print(f"Wrote {fn}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subject")
    p.add_argument("--rois", nargs="+", default=list(ROIS))
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--bids-folder", default=BIDS_FOLDER)
    a = p.parse_args()
    main(a.subject, a.rois, a.smoothed, a.bids_folder)
