#!/usr/bin/env python3
"""Extract per-trial gaze dispersion during the response_bar (estimation)
phase from raw EyeLink .edf files.

Built for the eye-movement confound check on the NPCr signal-voxel-fraction
brain-behavior correlation (see ``visualize/eu_vs_behavior.py``), but the
output TSV is general-purpose (one row per trial).

Requires ``edf2asc`` (SR Research EyeLink Developer's Kit) — a Linux x86-64
binary, so this only runs on sciencecluster, not locally on macOS. Reuses
the copy already checked into ``~/git/risk_experiment`` /
``~/git/multlearn-sns`` rather than shipping a new one (proprietary SR
Research binary, not something to commit to this repo).

Approach (same pattern as riskeye/riskeye/eyetracking/prepare_data.py):
run edf2asc twice per .edf — once for events (``-e``), once for samples
(``-s``) — then parse both as plain text:

  - MSG lines ``start_type-stim_trial-<n>_phase-<p>`` (written by
    exptools2's ``Trial.log_phase_info``, see
    ``libs/exptools2/exptools2/core/trial.py``) give phase onsets. Phase 4
    = response_bar start, phase 5 = feedback start (see
    ``experiment/task.py``'s ``phase_names``) — the window in between is
    the value-estimation response period.
  - Sample lines ``<time> <x> <y> <pupil> ...`` (tab-separated, 500 Hz,
    EyeLink '.' marks missing/blink samples -> NaN).

Per trial: gaze_dispersion = sqrt(var(x) + var(y)) over samples in the
response_bar window, in EyeLink screen pixels (1920x1080).

Usage (on sciencecluster, abstract_values conda env):
    python -m abstract_values.eyetracking.extract_gaze_dispersion \\
        --out /tmp/gaze_dispersion_all.tsv
    python -m abstract_values.eyetracking.extract_gaze_dispersion \\
        --subjects 03 04 --out /tmp/gaze_test.tsv
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EDF2ASC = str(Path.home() / "git" / "risk_experiment" / "edf2asc")
PHASE_RESPONSE = 4
PHASE_FEEDBACK = 5
MSG_RE = re.compile(r"start_type-stim_trial-(-?\d+)_phase-(\d+)")
RUN_RE = re.compile(r"sub-(.+)_ses-(\d+)_run-(\d+)_task-estimate\.(cdf|inverse_cdf)")


def convert(edf_path: Path, tmp_dir: Path) -> tuple[Path, Path]:
    msg_asc = tmp_dir / (edf_path.stem + "_msg.asc")
    samp_asc = tmp_dir / (edf_path.stem + "_samp.asc")
    subprocess.run([EDF2ASC, "-t", "-y", "-z", "-v", "-e", str(edf_path), str(msg_asc)],
                   capture_output=True, check=False)
    subprocess.run([EDF2ASC, "-t", "-y", "-z", "-v", "-s", str(edf_path), str(samp_asc)],
                   capture_output=True, check=False)
    return msg_asc, samp_asc


def parse_phase_onsets(msg_asc: Path) -> dict[tuple[int, int], int]:
    onsets: dict[tuple[int, int], int] = {}
    with open(msg_asc, errors="ignore") as f:
        for line in f:
            if not line.startswith("MSG"):
                continue
            m = MSG_RE.search(line)
            if not m:
                continue
            trial_nr, phase = int(m.group(1)), int(m.group(2))
            if trial_nr < 0:                            # pre-run calibration probe trial
                continue
            onsets[(trial_nr, phase)] = int(line.split("\t")[1])
    return onsets


def parse_samples(samp_asc: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, x, y) arrays; EyeLink '.' (missing/blink) -> NaN."""
    times, xs, ys = [], [], []
    with open(samp_asc, errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            t = parts[0].strip()
            if not t.lstrip("-").isdigit():
                continue
            times.append(int(t))
            for col, dst in ((1, xs), (2, ys)):
                v = parts[col].strip()
                dst.append(np.nan if v in (".", "") else float(v))
    return np.array(times), np.array(xs), np.array(ys)


def process_run(edf_path: Path, tmp_dir: Path, subject: str, session: int,
                run: int, mapping: str) -> list[dict]:
    msg_asc, samp_asc = convert(edf_path, tmp_dir)
    rows = []
    try:
        onsets = parse_phase_onsets(msg_asc)
        times, xs, ys = parse_samples(samp_asc)
        trial_nrs = sorted({t for t, p in onsets if p == PHASE_RESPONSE})
        for trial_nr in trial_nrs:
            t0 = onsets.get((trial_nr, PHASE_RESPONSE))
            t1 = onsets.get((trial_nr, PHASE_FEEDBACK))
            if t0 is None or t1 is None or t1 <= t0:
                continue
            mask = (times >= t0) & (times < t1)
            n = int(mask.sum())
            x, y = xs[mask], ys[mask]
            valid = np.isfinite(x) & np.isfinite(y)
            frac_valid = float(valid.mean()) if n else 0.0
            disp = (float(np.sqrt(np.nanvar(x[valid]) + np.nanvar(y[valid])))
                    if valid.sum() >= 5 else np.nan)
            rows.append(dict(subject=subject, session=session, mapping=mapping,
                            run=run, trial_nr=trial_nr, gaze_dispersion=disp,
                            n_samples=n, frac_valid=frac_valid))
    finally:
        msg_asc.unlink(missing_ok=True)
        samp_asc.unlink(missing_ok=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default="/shares/zne.uzh/gdehol/ds-abstractvalue")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--tmp-dir", default="/tmp/gaze_extract")
    args = p.parse_args()

    src = Path(args.bids_folder) / "sourcedata" / "behavior"
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or sorted(
        d.name.removeprefix("sub-") for d in src.glob("sub-*") if d.is_dir())

    all_rows = []
    for s in subjects:
        edf_files = sorted((src / f"sub-{s}").glob("ses-*/*.edf"))
        print(f"sub-{s}: {len(edf_files)} edf files", file=sys.stderr, flush=True)
        for edf_path in edf_files:
            m = RUN_RE.match(edf_path.stem)
            if not m:
                print(f"  skip (name mismatch): {edf_path.name}", file=sys.stderr)
                continue
            _, session, run, mapping = m.groups()
            all_rows.extend(process_run(edf_path, tmp_dir, s, int(session),
                                        int(run), mapping))
        print(f"  -> {len(all_rows)} trial rows so far", file=sys.stderr, flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}  ({len(df)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
