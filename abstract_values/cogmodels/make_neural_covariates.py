"""Per-participant neural covariates for the cognitive model (``--covariates``).

The same code-strength measures as the gradient figure
(``visualize/figure_gradient.py``): per region, the mean over its fsaverage
vertices of the best von Mises (orientation) or aPRF (value) cvR² above the
null model. Written for the model's behavioural subject ids (pil01 -> 1).

    v1_ori      orientation-code strength in V1
    ips_val     value-code strength, mean of IPS0, IPS1-2 and IPS3-5
    fd          mean framewise displacement (head motion)
    g_ori/g_val whole-cortex orientation / value code strength

All columns are z-scored across participants; the raw values are kept with a
``_raw`` suffix.

    python -m abstract_values.cogmodels.make_neural_covariates   # pycortex2 env
    -> notes/data/cogmodel_neural_covariates.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize import figure_gradient as fg
from abstract_values.visualize.group_surface_maps import discover_subjects
from abstract_values.visualize.qc_tuning_maps import collect

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "notes" / "data" / "cogmodel_neural_covariates.tsv"
IPS = ["IPS0", "IPS1–2", "IPS3–5"]


def main(out=OUT):
    deriv = Path(BIDS_FOLDER) / "derivatives"
    d = collect(deriv, discover_subjects(deriv), True)
    subs = [int(fg.PIL.get(s, s)) for s in d["subjects"]]
    ori, val = d["ori_gain"], d["val_gain"]
    verts = fg.exclusive_rois(fg.roi_verts())
    roi_ori = {r: np.nanmean(ori[:, vx], 1) for r, vx in verts.items()}
    roi_val = {r: np.nanmean(val[:, vx], 1) for r, vx in verts.items()}

    mo = pd.read_csv(REPO / "notes/data/motion_summary.tsv", sep="\t", dtype={"subject": str})
    mo["subject"] = mo.subject.str.replace("sub-", "").replace(fg.PIL).astype(int)
    fd = mo.groupby("subject").mean_fd.mean()

    raw = pd.DataFrame({"v1_ori": roi_ori["V1"],
                        "ips_val": np.mean([roi_val[r] for r in IPS], 0),
                        "g_ori": np.nanmean(ori, 1), "g_val": np.nanmean(val, 1)},
                       index=pd.Index(subs, name="subject"))
    raw["fd"] = fd.reindex(raw.index)
    raw = raw.dropna().sort_index()
    cov = (raw - raw.mean()) / raw.std()
    cov = cov.join(raw.add_suffix("_raw"))
    cov.to_csv(out, sep="\t", float_format="%.6g")
    print(f"{len(cov)} participants -> {out}")
    print(raw.corr("spearman").round(2).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(OUT))
    main(p.parse_args().out)
