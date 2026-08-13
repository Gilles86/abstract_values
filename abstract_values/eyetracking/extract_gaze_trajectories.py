#!/usr/bin/env python3
"""Extract per-trial gaze trajectories (x/y in degrees of visual angle,
resampled to a fixed number of points) during a configurable trial phase
window, for plotting gaze paths coloured by orientation.

Epoch is a CLI choice (--epoch-start/--epoch-end, phase names from
experiment/task.py's TaskTrial.phase_names), not hardcoded — this dataset
has two natural windows to look at: 'gabor'->'isi' (during stimulus
presentation, 1.5s, phase_names index 2->3) and 'response_bar'->'feedback'
(during value estimation, up to 3.5s, index 4->5). Re-run once per epoch
of interest; each run's --out is a separate TSV.

Sibling of ``extract_gaze_dispersion.py`` (same repo) — reuses its EDF
parsing helpers (msg/sample .asc parsing, phase-window convention) rather
than duplicating them. Where that script collapses each trial to a single
scalar (dispersion), this one keeps the trajectory shape.

Monitor geometry — READ PER RUN, NEVER HARDCODED
--------------------------------------------------
Screen width/distance/resolution differ across rooms (the fMRI suite is
55 cm / 100 cm / nominally 1920x1200 per ``experiment/settings/sns_fmri.yml``,
the behaviour-only rooms use different HP/Philips monitors — see
``experiment/settings/{hp_screen,single_subject,sns_multisubject}.yml``)
and can vary even within a template across actual runs. The only source
that is guaranteed correct for a *given* .edf is the exptools2 log written
alongside it at acquisition time, e.g.::

    sub-16_ses-2_run-01_task-estimate.inverse_cdf_expsettings.yml
        monitor: {width: 55, distance: 100, ...}   # cm
        window:  {size: [1920, 1080], ...}         # px

This script always derives pix2deg from that sibling file (edf stem +
``_expsettings.yml``) — never from ``experiment/settings/*.yml`` — so
screen geometry never needs to be re-derived by hand again. Conversion is
the exact (non-small-angle) flat-screen formula, per axis offset from
screen centre:

    deg = degrees(arctan((offset_px * width_cm / window_width_px) / distance_cm))

y is flipped (EyeLink pixel y grows downward; plots want up = positive).
Verified against this dataset's own ``DISPLAY_COORDS``/``GAZE_COORDS`` .asc
messages (``0 0 1920 1080``, top-left origin) rather than assumed.

Sample-level QC
----------------
EyeLink occasionally emits numeric (non-``.``) samples during partial
blinks/track loss that are wildly outside the physical display — this
dataset had raw pixel x in [-1130, 3270] against a 1920 px wide screen.
``reject_offscreen`` NaNs out anything outside [-10%, 110%] of the screen
bounds before interpolation, so a track-loss artifact can't drag a
straight-line spike across an otherwise-clean trajectory. Per-trial
``n_samples``/``frac_valid`` are also written out (same convention as
extract_gaze_dispersion.py) so a downstream min-validity filter (e.g.
frac_valid >= 0.5) can drop trials that were mostly blinks — filtering
happens at load time, not here, matching how eu_vs_behavior.py's
load_gaze_dispersion() does it for the dispersion TSV.

Orientation is read from the sibling events.tsv (edf stem + ``_events.tsv``),
column ``orientation`` — constant per trial_nr (see CLAUDE.md "clean
per-trial behavior" recipe; same trial key as extract_gaze_dispersion.py:
subject, session, mapping, run, trial_nr).

Each trial's response_bar-window samples are linearly interpolated onto
``N_RESAMPLE`` normalized-time points (0..1) so trials of different
durations/sampling gaps align into one flat table. No cross-trial
averaging happens here — that's a plotting-time decision (see
``abstract_values/visualize/plot_gaze_trajectories.py``).

Requires ``edf2asc`` (SR Research EyeLink Developer's Kit) — a Linux
x86-64 binary, so this only runs on sciencecluster, not locally on macOS.
Reuses the copy already checked into ``~/git/risk_experiment``.

Usage (on sciencecluster, abstract_values conda env):
    python -m abstract_values.eyetracking.extract_gaze_trajectories \\
        --out /tmp/gaze_trajectories_all.tsv
    python -m abstract_values.eyetracking.extract_gaze_trajectories \\
        --subjects 03 04 --out /tmp/gaze_traj_test.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from abstract_values.eyetracking.extract_gaze_dispersion import (
    RUN_RE,
    convert,
    parse_phase_onsets,
    parse_samples,
)

# Matches experiment/task.py TaskTrial.phase_names exactly — do not reorder
# without checking that file, phase MSG numbers on disk are baked in as ints.
PHASE_NAMES = ["green_fixation", "white_fixation", "gabor", "isi",
               "response_bar", "feedback", "iti"]

N_RESAMPLE = 20         # points per trial, normalized time 0..1 across the chosen epoch
OFFSCREEN_MARGIN = 0.1  # samples beyond [-10%, 110%] of screen bounds are tracker artifacts, not gaze


def pix2deg(offset_px: np.ndarray, width_cm: float, width_px: int, distance_cm: float) -> np.ndarray:
    """Visual angle (deg) of a pixel offset from screen centre (flat-screen exact formula)."""
    offset_cm = offset_px * (width_cm / width_px)
    return np.degrees(np.arctan(offset_cm / distance_cm))


def reject_offscreen(xs: np.ndarray, ys: np.ndarray, w_px: int, h_px: int) -> tuple:
    """NaN-out samples physically impossible on the actual screen.

    EyeLink occasionally emits numeric (non-'.') samples during partial
    blinks / track loss that are wildly outside the display — e.g. a real
    file in this dataset had x in [-1130, 3270] against a 1920 px wide
    screen. These are not gaze, and unlike '.' (already NaN) they were
    silently passing through resample_trial's interpolation, dragging
    straight-line artifacts across otherwise-clean trajectories.
    """
    x_lo, x_hi = -OFFSCREEN_MARGIN * w_px, w_px * (1 + OFFSCREEN_MARGIN)
    y_lo, y_hi = -OFFSCREEN_MARGIN * h_px, h_px * (1 + OFFSCREEN_MARGIN)
    bad = (xs < x_lo) | (xs > x_hi) | (ys < y_lo) | (ys > y_hi)
    xs, ys = xs.copy(), ys.copy()
    xs[bad], ys[bad] = np.nan, np.nan
    return xs, ys


def load_geometry(expsettings_path: Path) -> tuple[float, float, int, int]:
    with open(expsettings_path) as f:
        s = yaml.safe_load(f)
    width_cm = float(s["monitor"]["width"])
    distance_cm = float(s["monitor"]["distance"])
    w_px, h_px = s["window"]["size"]
    return width_cm, distance_cm, int(w_px), int(h_px)


def load_orientations(events_path: Path) -> pd.Series:
    d = pd.read_csv(events_path, sep="\t")
    return d.groupby("trial_nr")["orientation"].first()


def resample_trial(t: np.ndarray, x: np.ndarray, y: np.ndarray, n: int = N_RESAMPLE):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5:
        return None
    t, x, y = t[valid], x[valid], y[valid]
    order = np.argsort(t)
    t, x, y = t[order], x[order], y[order]
    span = max(t.max() - t.min(), 1)
    t_norm = (t - t.min()) / span
    t_grid = np.linspace(0, 1, n)
    return np.interp(t_grid, t_norm, x), np.interp(t_grid, t_norm, y)


def process_run(edf_path: Path, tmp_dir: Path, subject: str, session: int,
                run: int, mapping: str, phase_start: int, phase_end: int) -> list[dict]:
    msg_asc, samp_asc = convert(edf_path, tmp_dir)
    rows: list[dict] = []
    try:
        stem = edf_path.stem  # e.g. sub-19_ses-2_run-08_task-estimate.cdf
        events_path = edf_path.with_name(stem + "_events.tsv")
        expsettings_path = edf_path.with_name(stem + "_expsettings.yml")
        if not events_path.exists() or not expsettings_path.exists():
            print(f"  skip (missing sibling events/expsettings): {edf_path.name}", file=sys.stderr)
            return rows

        width_cm, distance_cm, w_px, h_px = load_geometry(expsettings_path)
        orientations = load_orientations(events_path)

        onsets = parse_phase_onsets(msg_asc)
        times, xs, ys = parse_samples(samp_asc)
        xs, ys = reject_offscreen(xs, ys, w_px, h_px)
        x_deg = pix2deg(xs - w_px / 2, width_cm, w_px, distance_cm)
        y_deg = pix2deg(h_px / 2 - ys, width_cm, w_px, distance_cm)  # flip: pixel y grows downward

        trial_nrs = sorted({t for t, p in onsets if p == phase_start})
        for trial_nr in trial_nrs:
            t0 = onsets.get((trial_nr, phase_start))
            t1 = onsets.get((trial_nr, phase_end))
            if t0 is None or t1 is None or t1 <= t0 or trial_nr not in orientations.index:
                continue
            mask = (times >= t0) & (times < t1)
            n = int(mask.sum())
            valid = np.isfinite(x_deg[mask]) & np.isfinite(y_deg[mask])
            frac_valid = float(valid.mean()) if n else 0.0
            resampled = resample_trial(times[mask].astype(float), x_deg[mask], y_deg[mask])
            if resampled is None:
                continue
            x_rs, y_rs = resampled
            orientation = float(orientations.loc[trial_nr])
            for i in range(N_RESAMPLE):
                rows.append(dict(subject=subject, session=session, mapping=mapping, run=run,
                                  trial_nr=trial_nr, orientation=orientation,
                                  sample_idx=i, x_deg=x_rs[i], y_deg=y_rs[i],
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
    p.add_argument("--tmp-dir", default="/tmp/gaze_traj_extract")
    p.add_argument("--epoch-start", default="response_bar", choices=PHASE_NAMES,
                   help="Phase whose onset starts the extracted window.")
    p.add_argument("--epoch-end", default="feedback", choices=PHASE_NAMES,
                   help="Phase whose onset ends the extracted window (exclusive).")
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
                                        int(run), mapping, phase_start, phase_end))
        print(f"  -> {len(all_rows)} sample rows so far", file=sys.stderr, flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}  ({len(df)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
