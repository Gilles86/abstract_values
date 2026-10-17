"""ROI-mean BOLD time courses, confounds and events for deconvolution (one subject).

Same ROIs as collect_vmpfc_dropout (mPFC subregions, PCC, left M1, V1, IPS,
white matter). Per run: the ROI-mean preprocessed signal as % of its run mean
(averaging before scaling, so near-empty dropout voxels cannot blow up), the
fmriprep cosine + 6 motion confounds, and the gabor / response-bar events with
value, bid and RT. Onsets are shifted by -TR/2 so they line up with fmriprep's
slice-timing reference (mid-volume) when sample i is taken to sit at i * TR.

    python -m abstract_values.visualize.collect_roi_timeseries 03 \\
        --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue --out-dir /scratch/gdehol/vmpfc_roi_ts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image as nimage

from abstract_values.utils.data import BIDS_FOLDER, Subject
from abstract_values.visualize.collect_vmpfc_dropout import (BAD_TIMING, ROIS, anat_file,
                                                             confounds, label_volume, roi_masks)

MOTION = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]


def main(s, bids, out_dir):
    bids, out_dir = Path(bids), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = Subject(s, bids_folder=bids)
    sessions = sorted(sub.get_sessions())
    ref3 = nimage.index_img(str(sub.get_preprocessed_bold(sessions[0])[0]), 0)
    masks = roi_masks(sub, ref3, label_volume(nib.load(str(anat_file(
        bids, s, "desc-aparcaseg_dseg.nii.gz")))))
    ts_rows, conf_rows, ev_rows = [], [], []
    for ses in sessions:
        runs = sorted(sub.get_runs(ses))
        ev = sub.get_events(ses, runs)
        paths = sub.get_preprocessed_bold(ses, runs)
        func_dir = Path(paths[0]).parent
        mapping = sub.get_mapping(ses)
        for run, path in zip(runs, paths):
            if (s, ses, run) in BAD_TIMING:
                continue
            print(f"sub-{s} ses-{ses} run-{run}", flush=True)
            img = nib.load(str(path))
            tr = float(json.loads(Path(str(path).replace(".nii.gz", ".json")).read_text())
                       .get("RepetitionTime", img.header.get_zooms()[3]))
            data = np.asarray(img.dataobj, dtype=np.float32)
            ts = {}
            for r in ROIS:
                x = data[masks[r]].mean(0) if masks[r].any() else np.full(data.shape[-1], np.nan)
                ts[r] = (x / x.mean() - 1) * 100
            d = pd.DataFrame(ts)
            d.insert(0, "volume", np.arange(len(d)))
            d.insert(0, "run", run)
            d.insert(0, "session", ses)
            d.insert(0, "subject", s)
            ts_rows.append(d)
            c = confounds(func_dir, s, ses, run)
            cn = pd.read_csv(func_dir / f"sub-{s}_ses-{ses}_task-abstractvalue_run-{run}"
                             f"_desc-confounds_timeseries.tsv", sep="\t", nrows=0).columns
            cols = [k for k in cn if k.startswith("cosine")] + MOTION
            cdf = pd.DataFrame(c, columns=cols)
            cdf.insert(0, "volume", np.arange(len(cdf)))
            cdf.insert(0, "run", run)
            cdf.insert(0, "session", ses)
            cdf.insert(0, "subject", s)
            conf_rows.append(cdf)
            e = ev.loc[run].reset_index()
            e["onset"] = e["onset"] - tr / 2
            e = e[e.event_type.isin(["gabor", "response_bar"])]
            bids_ = e[e.event_type == "response_bar"].set_index("trial_nr")["bid"]
            e["bid"] = e.trial_nr.map(bids_)
            ev_rows.append(e.assign(subject=s, session=ses, run=run, mapping=mapping, tr=tr)
                           [["subject", "session", "run", "trial_nr", "event_type", "onset",
                             "duration", "orientation", "value", "bid", "mapping", "tr"]])
            del data
    pd.concat(ts_rows).to_csv(out_dir / f"sub-{s}_roi_ts.tsv.gz", sep="\t", index=False)
    pd.concat(conf_rows).to_csv(out_dir / f"sub-{s}_confounds.tsv.gz", sep="\t", index=False)
    pd.concat(ev_rows).to_csv(out_dir / f"sub-{s}_events.tsv", sep="\t", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subject")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--out-dir", default="/scratch/gdehol/vmpfc_roi_ts")
    a = p.parse_args()
    main(a.subject, a.bids_folder, a.out_dir)
