#!/usr/bin/env python3
"""
Estimate a run's stimulus-to-BOLD temporal offset by an onset-shift sweep.

For runs where scanner triggers were dropped during the dummy/start sync,
the experiment clock can start k volumes late -> all event onsets are
shifted by an integer k*TR relative to the BOLD (a constant offset; the
sub-TR residual is < 1/2 TR and negligible). This finds k empirically:

  for each candidate shift delta (in TRs):
    build a gabor-stimulus GLM with onsets shifted by delta*TR,
    fit it to V1 (which the gabor drives strongly),
    record mean V1 R^2.
  The delta that MAXIMISES V1 R^2 is the true offset.

Interpretation:
  * sharp peak at delta=0  -> clock was fine (only pulses failed to LOG);
                              run is usable as-is.
  * sharp peak at delta=k  -> clock started k volumes late; shift onsets by
                              +k*TR (or drop k leading volumes) to recover.
  * no clear peak (flat)   -> not a clean constant offset (e.g. progressive
                              drift); the run is probably unrecoverable.

V1 is the anchor on purpose: strong, fast, stimulus-locked signal gives a
sharp R^2(delta) curve. (NPC value tuning is far too weak to localise it.)

Usage
-----
  python -m abstract_values.glm.estimate_run_offset 18 1 1
  python -m abstract_values.glm.estimate_run_offset 18 1 1 --shifts -3 45
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from nilearn.glm.first_level import make_first_level_design_matrix

from abstract_values.utils.data import Subject, BIDS_FOLDER

TR = 0.996


def estimate(subject, session, run, shift_lo=-3, shift_hi=45,
             roi="BensonV1", bids_folder=BIDS_FOLDER):
    sub = Subject(subject, bids_folder=bids_folder)
    bold_path = sub.get_preprocessed_bold(session, [run])[0]
    masker = NiftiMasker(mask_img=sub.get_roi_mask(roi, hemi="LR")).fit()
    Y = masker.transform(str(bold_path)).astype(np.float64)      # (n_vols, n_vox)
    n_vols = Y.shape[0]
    frame_times = np.arange(n_vols) * TR

    ev = sub.get_events(session, [run]).reset_index()
    gab = ev[ev["event_type"] == "gabor"]
    onsets = gab["onset"].to_numpy(dtype=float)
    dur = (gab["duration"].to_numpy(dtype=float)
           if "duration" in gab and gab["duration"].notna().all()
           else np.full(len(onsets), 1.0))
    print(f"sub-{subject} ses-{session} run-{run}: {n_vols} vols, "
          f"{Y.shape[1]} {roi} voxels, {len(onsets)} gabor events")

    SStot = ((Y - Y.mean(0)) ** 2).sum(0)
    rows = []
    for d in range(int(shift_lo), int(shift_hi) + 1):
        on = onsets + d * TR
        keep = (on >= 0) & (on < n_vols * TR)
        events = pd.DataFrame({"onset": on[keep], "duration": dur[keep],
                               "trial_type": "stim"})
        dm = make_first_level_design_matrix(
            frame_times, events, hrf_model="spm",
            drift_model="cosine", high_pass=0.01)
        X = dm.values
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        SSres = ((Y - X @ beta) ** 2).sum(0)
        r2 = 1 - SSres / SStot
        rows.append((d, float(np.nanmean(r2)), float(np.nanmedian(r2))))
    df = pd.DataFrame(rows, columns=["shift_TR", "mean_r2", "median_r2"])

    best = df.loc[df["mean_r2"].idxmax()]
    base = df["mean_r2"].median()
    peak_z = (best["mean_r2"] - base) / (df["mean_r2"].std() + 1e-9)
    print(df.to_string(index=False))
    print(f"\nPEAK shift = {int(best.shift_TR)} TR ({best.shift_TR*TR:+.1f} s)  "
          f"mean V1 R2={best.mean_r2:.4f}  (peak prominence ~{peak_z:.1f} SD)")
    if int(best.shift_TR) == 0:
        print("=> peak at 0: run aligned as-is (pulses just unlogged).")
    elif peak_z > 3:
        print(f"=> clean peak at {int(best.shift_TR)} TR: shift onsets by "
              f"+{int(best.shift_TR)}*TR (or drop {int(best.shift_TR)} leading vols).")
    else:
        print("=> no sharp peak: likely drift, not a constant offset — "
              "run probably unrecoverable.")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subject"); p.add_argument("session", type=int)
    p.add_argument("run", type=int)
    p.add_argument("--shifts", type=int, nargs=2, default=[-3, 45],
                   metavar=("LO", "HI"))
    p.add_argument("--roi", default="BensonV1")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--out", default=None, help="optional TSV output path")
    args = p.parse_args()
    df = estimate(args.subject, args.session, args.run,
                  shift_lo=args.shifts[0], shift_hi=args.shifts[1],
                  roi=args.roi, bids_folder=args.bids_folder)
    if args.out:
        df.to_csv(args.out, sep="\t", index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
