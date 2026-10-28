"""A3 control (cluster step): is the 90 deg distinctiveness a univariate amplitude effect?

Per subject, ROI and session:
  * ROI-mean beta per orientation (mean over voxels, then over the 8 runs)
    -> sub-<s>_desc-a3roimean.tsv
  * crossnobis RDM after removing each trial's ROI-mean (pattern mean-centring across
    voxels), so a 90-deg trial that is simply larger/smaller overall no longer counts
    -> sub-<s>_desc-a3demeaned_rdm.tsv  (kind = crossnobis_demeaned, split = all)

    python compute_controls.py 03 [--bids-folder ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image, masking

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_rdms import ROIS, roi_mask  # noqa: E402

from abstract_values.geometry.crossnobis import crossnobis, trial_labels
from abstract_values.utils.data import BIDS_FOLDER, Subject


def main(subject, bids_folder=BIDS_FOLDER):
    sub = Subject(subject, bids_folder=bids_folder)
    lab = trial_labels(sub)
    betas = sub.get_single_trial_estimates(sorted(sub.get_sessions()))
    ref = image.index_img(betas, 0)
    rdm, mean_rows = [], []
    for roi, parts in ROIS.items():
        try:
            mask = roi_mask(sub, parts, ref)
        except FileNotFoundError as e:
            print(f"skip {roi}: {e}")
            continue
        X = masking.apply_mask(betas, mask)
        X = X[:, np.all(np.isfinite(X), 0) & (X.std(0) > 0)]
        m = X.mean(1)
        Xd = X - m[:, None]
        for ses, g in lab.groupby("session"):
            idx = g.index.to_numpy()
            mm = pd.DataFrame(dict(orientation=g.orientation.values, b=m[idx])).groupby("orientation").b.mean()
            mean_rows.append(pd.DataFrame(dict(subject=subject, roi=roi, session=ses,
                                               mapping=g.mapping.iloc[0], orientation=mm.index,
                                               mean_beta=mm.values)))
            D, oris = crossnobis(Xd[idx], g.run.to_numpy(), g.orientation.to_numpy())
            val = g.groupby("orientation").value.first().reindex(oris).to_numpy()
            i, j = np.triu_indices(len(oris), 1)
            rdm.append(pd.DataFrame(dict(
                subject=subject, roi=roi, session=ses, mapping=g.mapping.iloc[0],
                kind="crossnobis_demeaned", split="all", n_voxels=X.shape[1],
                ori_i=oris[i], ori_j=oris[j], val_i=val[i], val_j=val[j], d=D[i, j])))
            print(f"sub-{subject} {roi} ses-{ses}", flush=True)
    odir = Path(bids_folder) / "derivatives" / "followup_2026-10" / "a3_categorical_90" / f"sub-{subject}"
    odir.mkdir(parents=True, exist_ok=True)
    pd.concat(rdm).to_csv(odir / f"sub-{subject}_desc-a3demeaned_rdm.tsv", sep="\t", index=False)
    pd.concat(mean_rows).to_csv(odir / f"sub-{subject}_desc-a3roimean.tsv", sep="\t", index=False)
    print("done")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subject")
    p.add_argument("--bids-folder", default=BIDS_FOLDER)
    a = p.parse_args()
    main(a.subject, a.bids_folder)
