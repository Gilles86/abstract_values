"""Combine the per-subject outputs of collect_vmpfc_dropout.py (cluster side).

Writes, into --out-dir (default notes/data):

  vmpfc_dropout_rois.tsv       one row per (subject, ROI)
  vmpfc_dropout_tsnr_hist.tsv  voxel tSNR histogram per (subject, ROI)
  vmpfc_dropout_trials.tsv     one row per trial (the epochs' rows)
  vmpfc_dropout_epochs.npz     gabor- and bar-locked ROI epochs, all trials;
                               per-subject display slices
  vmpfc_dropout_mni.npz        group mean tSNR / mean EPI in MNI152NLin2009cAsym
                               (2 mm), per-label coverage probability, template

    python -m abstract_values.visualize.combine_vmpfc_dropout \\
        --in-dir /scratch/gdehol/vmpfc_dropout \\
        --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

TPL = Path("/shares/zne.uzh/containers/templateflow/tpl-MNI152NLin2009cAsym")


def main(in_dir, bids, out_dir):
    in_dir, out_dir, bids = Path(in_dir), Path(out_dir), Path(bids)
    subs = sorted(p.name.split("_")[0][4:] for p in in_dir.glob("sub-*_data.npz"))
    print(f"{len(subs)} subjects", flush=True)
    for kind in ("rois", "tsnr_hist", "trials"):
        d = pd.concat([pd.read_csv(in_dir / f"sub-{s}_{kind}.tsv", sep="\t", dtype={"subject": str})
                       for s in subs])
        d.to_csv(out_dir / f"vmpfc_dropout_{kind}.tsv", sep="\t", index=False)
    eg, eb, sl = [], [], {}
    for s in subs:
        z = np.load(in_dir / f"sub-{s}_data.npz")
        eg.append(z["epochs_gabor"])
        eb.append(z["epochs_bar"])
        for k in z.files:
            if k.split("_")[0] in ("t1", "epi", "tsnr", "lab") or k in ("slice_xyz", "voxsize"):
                sl[f"sub{s}__{k}"] = z[k]
    np.savez_compressed(out_dir / "vmpfc_dropout_epochs.npz",
                        epochs_gabor=np.concatenate(eg), epochs_bar=np.concatenate(eb),
                        t_gabor=z["t_gabor"], t_bar=z["t_bar"], rois=z["rois"],
                        tsnr_bins=z["tsnr_bins"], subjects=np.array(subs), **sl)

    maps = {"tsnr": [], "meanepi": [], "lab": []}
    used = []
    for s in subs:
        qa = bids / "derivatives" / "qa" / "tsnr" / f"sub-{s}"
        f = {"tsnr": qa / f"sub-{s}_task-abstractvalue_space-MNI152NLin2009cAsym_res-2_desc-tsnr_map.nii.gz",
             "meanepi": qa / f"sub-{s}_task-abstractvalue_space-MNI152NLin2009cAsym_res-2_desc-meanepi_map.nii.gz",
             "lab": qa / f"sub-{s}_space-MNI152NLin2009cAsym_res-2_desc-mpfclabels_dseg.nii.gz"}
        if not all(p.exists() for p in f.values()):
            print(f"sub-{s}: MNI maps missing", flush=True)
            continue
        used.append(s)
        for k, p in f.items():
            maps[k].append(np.asarray(nib.load(str(p)).dataobj, dtype=np.float32))
    lab = np.stack(maps["lab"]).round().astype(np.int8)
    tpl = nib.load(str(TPL / "tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz"))
    bm = np.asarray(nib.load(str(TPL / "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz")).dataobj) > 0
    np.savez_compressed(out_dir / "vmpfc_dropout_mni.npz",
                        tsnr=np.mean(maps["tsnr"], 0), meanepi=np.mean(maps["meanepi"], 0),
                        tsnr_lowfrac=np.mean(np.stack(maps["tsnr"]) < 20, 0),
                        labprob=np.stack([(lab == i).mean(0) for i in range(1, 8)]),
                        template=np.asarray(tpl.dataobj, dtype=np.float32), brain=bm,
                        affine=tpl.affine, subjects=np.array(used))
    print(f"MNI maps from {len(used)} subjects", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", default="/scratch/gdehol/vmpfc_dropout")
    p.add_argument("--bids-folder", default="/shares/zne.uzh/gdehol/ds-abstractvalue")
    p.add_argument("--out-dir", default="notes/data")
    a = p.parse_args()
    main(a.in_dir, a.bids_folder, a.out_dir)
