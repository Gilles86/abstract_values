#!/usr/bin/env python3
"""Detect microsaccades during the pre-stimulus fixation window (trial
onset -> gabor onset: green_fixation + white_fixation, ~1.0s) using the
Engbert & Kliegl (2003, Vision Research) velocity-threshold algorithm.

Unlike extract_gaze_trajectories.py, this needs the RAW, full-sampling-
rate sample sequence — microsaccades are ~10-30ms events, destroyed by
that script's 20-point-per-trial resampling. So this is a separate
extraction, sharing geometry/QC helpers with extract_gaze_trajectories.py
(pix2deg, reject_offscreen, load_geometry, load_orientations) but not its
resample-to-fixed-grid step.

Algorithm (Engbert & Kliegl 2003; this is THE standard method in the
saccade/microsaccade literature, not an ad hoc peak detector)
----------------------------------------------------------------
1. Velocity via a smoothed 5-sample differentiator (not raw sample-to-
   sample differencing, which is too noisy):
       v(n) = (x(n+2) + x(n+1) - x(n-1) - x(n-2)) / (6*dt)
2. A per-trial, per-axis noise estimate from the MEDIAN (robust to the
   saccades themselves, which are outliers in the velocity distribution):
       sigma_x = sqrt(median((v_x - median(v_x))^2))
3. Elliptic threshold in (v_x, v_y) space: flag samples where
       (v_x / (lambda*sigma_x))^2 + (v_y / (lambda*sigma_y))^2 > 1
   lambda=6 is Engbert & Kliegl's standard choice.
4. Contiguous flagged runs >= MIN_DURATION_MS become candidate events;
   events closer together than MIN_ISI_MS are merged (avoids splitting
   one saccade's velocity profile into two events at a brief dip).
5. Amplitude/direction = the (start -> end) displacement vector across
   the event; NOT a "micro"saccade filter yet (amplitude < ~1 deg is the
   usual cutoff) — that's applied downstream at analysis time, same
   extraction/filtering split as extract_gaze_trajectories.py's
   frac_valid.

Sample-quality requirement: the ENTIRE window must be blink-free and
on-screen (reject_offscreen already applied) for detection to run on
that trial at all. Interpolating across a blink before differentiating
would create a spurious velocity spike that looks exactly like a
saccade — safer to just drop the trial (reported as a QC count) than
risk manufacturing fake events. Given the window is only ~1s this drops
a meaningful fraction of trials; that's the correct trade, not a bug.

Requires ``edf2asc`` (SR Research EyeLink Developer's Kit) — a Linux
x86-64 binary, so this only runs on sciencecluster, not locally on macOS.

Usage (on sciencecluster, abstract_values conda env):
    python -m abstract_values.eyetracking.extract_microsaccades \\
        --out /tmp/microsaccades_all.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abstract_values.eyetracking.extract_gaze_dispersion import (
    RUN_RE,
    convert,
    parse_phase_onsets,
    parse_samples,
)
from abstract_values.eyetracking.extract_gaze_trajectories import (
    PHASE_NAMES,
    load_geometry,
    load_orientations,
    pix2deg,
    reject_offscreen,
)

LAMBDA = 6.0            # Engbert & Kliegl's standard velocity-threshold multiplier
MIN_DURATION_MS = 6.0   # minimum event duration to count as a saccade, not noise
MIN_ISI_MS = 20.0       # events closer than this are merged into one


def detect_microsaccades(t_ms: np.ndarray, x_deg: np.ndarray, y_deg: np.ndarray) -> list[dict]:
    """t_ms strictly increasing, x_deg/y_deg finite and gap-free (caller's
    responsibility — this function does not handle NaNs or missing samples,
    by design: interpolating across gaps before differentiating fabricates
    velocity artifacts). Returns one dict per detected event."""
    n = len(t_ms)
    if n < 5:
        return []
    dt_ms = np.median(np.diff(t_ms))
    if dt_ms <= 0:
        return []
    dt_s = dt_ms / 1000.0

    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    vx[2:-2] = (x_deg[4:] + x_deg[3:-1] - x_deg[1:-3] - x_deg[0:-4]) / (6 * dt_s)
    vy[2:-2] = (y_deg[4:] + y_deg[3:-1] - y_deg[1:-3] - y_deg[0:-4]) / (6 * dt_s)
    valid = np.isfinite(vx)
    if valid.sum() < 5:
        return []

    med_vx, med_vy = np.median(vx[valid]), np.median(vy[valid])
    sigma_x = np.sqrt(np.median((vx[valid] - med_vx) ** 2))
    sigma_y = np.sqrt(np.median((vy[valid] - med_vy) ** 2))
    if sigma_x == 0 or sigma_y == 0:
        return []

    ellipse = np.where(valid, (vx / (LAMBDA * sigma_x)) ** 2 + (vy / (LAMBDA * sigma_y)) ** 2, 0.0)
    is_event = ellipse > 1.0

    min_dur_samples = max(1, int(round(MIN_DURATION_MS / dt_ms)))
    min_isi_samples = int(round(MIN_ISI_MS / dt_ms))

    runs = []
    i = 0
    while i < n:
        if is_event[i]:
            j = i
            while j < n and is_event[j]:
                j += 1
            if (j - i) >= min_dur_samples:
                runs.append([i, j - 1])
            i = j
        else:
            i += 1

    merged = []
    for run in runs:
        if merged and (run[0] - merged[-1][1]) < min_isi_samples:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    events = []
    for i0, i1 in merged:
        dx, dy = x_deg[i1] - x_deg[i0], y_deg[i1] - y_deg[i0]
        speed = np.hypot(vx[i0:i1 + 1], vy[i0:i1 + 1])
        events.append(dict(
            onset_ms=float(t_ms[i0] - t_ms[0]),
            duration_ms=float(t_ms[i1] - t_ms[i0]),
            amplitude_deg=float(np.hypot(dx, dy)),
            direction_deg=float(np.degrees(np.arctan2(dy, dx))),
            peak_vel_deg_s=float(np.nanmax(speed)) if np.isfinite(speed).any() else np.nan,
        ))
    return events


def process_run(edf_path: Path, tmp_dir: Path, subject: str, session: int,
                run: int, mapping: str, phase_start: int, phase_end: int) -> tuple:
    msg_asc, samp_asc = convert(edf_path, tmp_dir)
    rows: list[dict] = []
    n_trials_seen = n_trials_clean = 0
    try:
        stem = edf_path.stem
        events_path = edf_path.with_name(stem + "_events.tsv")
        expsettings_path = edf_path.with_name(stem + "_expsettings.yml")
        if not events_path.exists() or not expsettings_path.exists():
            print(f"  skip (missing sibling events/expsettings): {edf_path.name}", file=sys.stderr)
            return rows, n_trials_seen, n_trials_clean

        width_cm, distance_cm, w_px, h_px = load_geometry(expsettings_path)
        orientations = load_orientations(events_path)

        onsets = parse_phase_onsets(msg_asc)
        times, xs, ys = parse_samples(samp_asc)
        xs, ys = reject_offscreen(xs, ys, w_px, h_px)
        x_deg = pix2deg(xs - w_px / 2, width_cm, w_px, distance_cm)
        y_deg = pix2deg(h_px / 2 - ys, width_cm, w_px, distance_cm)

        trial_nrs = sorted({t for t, p in onsets if p == phase_start})
        for trial_nr in trial_nrs:
            t0 = onsets.get((trial_nr, phase_start))
            t1 = onsets.get((trial_nr, phase_end))
            if t0 is None or t1 is None or t1 <= t0 or trial_nr not in orientations.index:
                continue
            n_trials_seen += 1
            mask = (times >= t0) & (times < t1)
            t_win, x_win, y_win = times[mask].astype(float), x_deg[mask], y_deg[mask]
            if len(t_win) < 5 or not (np.isfinite(x_win).all() and np.isfinite(y_win).all()):
                continue  # any blink/track-loss in this ~1s window -> drop the trial, don't interpolate
            order = np.argsort(t_win)
            t_win, x_win, y_win = t_win[order], x_win[order], y_win[order]
            n_trials_clean += 1

            orientation = float(orientations.loc[trial_nr])
            for ev in detect_microsaccades(t_win, x_win, y_win):
                rows.append(dict(subject=subject, session=session, mapping=mapping, run=run,
                                  trial_nr=trial_nr, orientation=orientation, **ev))
    finally:
        msg_asc.unlink(missing_ok=True)
        samp_asc.unlink(missing_ok=True)
    return rows, n_trials_seen, n_trials_clean


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default="/shares/zne.uzh/gdehol/ds-abstractvalue")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--tmp-dir", default="/tmp/microsaccade_extract")
    p.add_argument("--epoch-start", default="green_fixation", choices=PHASE_NAMES)
    p.add_argument("--epoch-end", default="gabor", choices=PHASE_NAMES)
    args = p.parse_args()
    phase_start, phase_end = PHASE_NAMES.index(args.epoch_start), PHASE_NAMES.index(args.epoch_end)
    if phase_end <= phase_start:
        p.error(f"--epoch-end ({args.epoch_end}) must come after --epoch-start ({args.epoch_start})")

    src = Path(args.bids_folder) / "sourcedata" / "behavior"
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or sorted(
        d.name.removeprefix("sub-") for d in src.glob("sub-*") if d.is_dir())

    print(f"Epoch: {args.epoch_start} (phase {phase_start}) -> "
          f"{args.epoch_end} (phase {phase_end})", file=sys.stderr)
    all_rows = []
    total_seen = total_clean = 0
    for s in subjects:
        edf_files = sorted((src / f"sub-{s}").glob("ses-*/*.edf"))
        print(f"sub-{s}: {len(edf_files)} edf files", file=sys.stderr, flush=True)
        for edf_path in edf_files:
            m = RUN_RE.match(edf_path.stem)
            if not m:
                print(f"  skip (name mismatch): {edf_path.name}", file=sys.stderr)
                continue
            _, session, run, mapping = m.groups()
            rows, n_seen, n_clean = process_run(edf_path, tmp_dir, s, int(session),
                                                int(run), mapping, phase_start, phase_end)
            all_rows.extend(rows)
            total_seen += n_seen
            total_clean += n_clean
        print(f"  -> {len(all_rows)} events so far ({total_clean}/{total_seen} trials clean)",
              file=sys.stderr, flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}  ({len(df)} events, {total_clean}/{total_seen} trials had a full "
          f"blink-free window, {df['trial_nr'].groupby([df['subject'], df['trial_nr']]).ngroups if len(df) else 0} "
          f"trials contributed >=1 event)", file=sys.stderr)


if __name__ == "__main__":
    main()
