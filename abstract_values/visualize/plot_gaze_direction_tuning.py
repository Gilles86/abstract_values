#!/usr/bin/env python3
"""Gaze-direction tuning curves: endpoint direction angle and displacement
magnitude as a function of grating orientation, for two groups of trials
(default: gabor presentation vs. response_bar epoch), with uncertainty
bands, paired per-orientation significance testing, and (optionally)
theoretical prediction curves for the angle panel.

Distils the "fan" pattern visible in plot_gaze_trajectories.py's grand-
average panel into an explicit function of orientation — easier to read
the *shape* of the orientation -> gaze-direction relationship off a
tuning curve than off overlapping 2D paths.

Endpoint = mean of the trajectory's last END_SAMPLES resampled points
(robust tail estimate, same idea as ORIGIN_SAMPLES in
plot_gaze_trajectories.recenter_trials). angle_deg = atan2(y, x) of that
endpoint relative to the recentred trial-onset origin (0 = rightward,
90 = up, +-180 = left, -90 = down). magnitude_deg = its length.

Uncertainty
-----------
Magnitude: mean +/- 1 SEM across subjects (linear stat, no circular
issues).
Angle: circular. The grand-average line is the angle of the mean (x, y)
endpoint vector across subjects (matches how the "grand" 2D trajectory is
computed everywhere else in this codebase), not a separate circular mean
of per-subject angles — the two agree when the resultant vector length R
is not tiny, and using the same definition keeps every figure in this
family internally consistent. The shaded band is a subject-level
bootstrap 95% CI (resample subjects with replacement, recompute the
resultant-vector angle each time, take the percentile CI of the
wrapped bootstrap-minus-observed differences — avoids branch-cut
artifacts from naively percentiling raw angles).

Difference panels: difference of the means, not the mean of the vector
difference
--------------------------------------------------------------------------
Per orientation, paired across subjects present in BOTH groups, each
subject contributes their OWN direction (angle_a_i, angle_b_i) and
magnitude (mag_a_i, mag_b_i) — one well-defined number per condition —
and the difference is taken PER SUBJECT: delta_angle_i = angle_a_i -
angle_b_i (wrapped), delta_mag_i = mag_a_i - mag_b_i. The panels show the
mean of these per-subject differences (circular mean for angle, plain
mean for magnitude), not the angle/magnitude of the mean (x, y)
DIFFERENCE VECTOR.

This distinction matters a lot in practice: the difference-VECTOR
approach (an earlier version of this script) degenerates whenever the
two conditions are similar — the difference vector shrinks toward (0, 0),
and the angle of a near-zero vector is essentially uniform noise on the
circle, regardless of how well-defined each condition's own direction
is. The difference-of-MEANS approach stays well-behaved as long as each
subject's own per-condition direction is reasonably reliable, which is
usually true even when the two conditions barely differ.

Two independent tests, each its own FDR family across orientations
(they ask genuinely different questions — do NOT pool their p-values):
  - Direction: bootstrap two-sided test (resample subjects, recompute
    the circular mean of delta_angle_i, compare the bootstrap
    distribution's position relative to 0 deg) against H0: mean
    direction difference = 0 deg.
  - Magnitude: same bootstrap logic on delta_mag_i (a plain linear
    scalar, no circular machinery needed) against H0: mean magnitude
    difference = 0.
A point's own 95% CI (error bar) is shown regardless of significance —
only the CONNECTING LINE in the direction panel is restricted to
contiguous significant runs, since a line implies a trend the
non-significant points don't support.

Two modes, same as compare_gaze_epochs.py:
  1. Two files (default): different trial epochs.
  2. One file, split by a column (--split-col), e.g. `mapping`.

Circular-data caveat: connecting per-orientation points into a line uses
np.unwrap on the angle sequence (sorted by orientation) so a genuine
smooth drift doesn't get chopped into a fake +/-360 deg jump — but this
can misrepresent a *real* large jump between adjacent orientations as
smooth. Per-subject points are always shown unconnected (scatter only)
to avoid compounding that assumption across noisier individual data.

Usage (local, no cluster needed):
    python -m abstract_values.visualize.plot_gaze_direction_tuning \\
        --tsv-a notes/data/gaze_trajectories_gabor.tsv --label-a "Gabor presentation" \\
        --tsv-b notes/data/gaze_trajectories_all.tsv   --label-b "Response bar" \\
        --show-bar-prediction \\
        --out notes/figures/gaze_direction_tuning.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from abstract_values.visualize.plot_gaze_trajectories import aggregate_df, load_and_aggregate

END_SAMPLES = 3
N_BOOT = 2000
ALPHA = 0.05
PALETTE = ["#3B5BA5", "#C44E52"]  # blue, red — hue is free here (x-axis is orientation)


def endpoint_stats(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """One row per (group_cols..., orientation): endpoint angle (deg,
    atan2 convention) and magnitude (deg), from the mean of each trial's
    (or each pre-aggregated group's) last END_SAMPLES resampled points."""
    n_max = df["sample_idx"].max()
    end = df[df["sample_idx"] > n_max - END_SAMPLES]
    end = end.groupby(group_cols + ["orientation"])[["x_deg", "y_deg"]].mean().reset_index()
    end["angle_deg"] = np.degrees(np.arctan2(end["y_deg"], end["x_deg"]))
    end["magnitude_deg"] = np.hypot(end["x_deg"], end["y_deg"])
    return end


def unwrap_by_orientation(sub: pd.DataFrame) -> np.ndarray:
    sub = sub.sort_values("orientation")
    return np.degrees(np.unwrap(np.radians(sub["angle_deg"].to_numpy())))


def bootstrap_vector_angle_ci(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, ci: float = 95,
                              seed: int = 0) -> tuple:
    """Subject-level bootstrap CI for atan2(mean(y), mean(x)) — the EXACT
    estimator the plotted grand-average direction line uses (mean of the
    actual (x, y) endpoint vectors across subjects, then its angle).

    Deliberately NOT a circular mean of pre-computed per-subject angles
    (atan2(mean(sin theta_i), mean(cos theta_i))) — that discards each
    subject's magnitude and is a *different* estimator from the
    magnitude-weighted one used for the centre line. Mixing a magnitude-
    weighted line with a magnitude-blind CI is internally inconsistent;
    resampling the raw vectors keeps both on the same definition, so
    `obs` returned here matches the plotted line to floating-point
    precision, with no separate re-anchoring step needed.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    obs = float(np.degrees(np.arctan2(y.mean(), x.mean())))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.degrees(np.arctan2(y[idx].mean(), x[idx].mean()))
    wrapped_diff = ((boot - obs + 180) % 360) - 180  # avoid branch-cut artifacts
    lo, hi = np.percentile(wrapped_diff, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return obs, obs + lo, obs + hi


def sem(x: np.ndarray) -> float:
    x = np.asarray(x)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def paired_per_subject_diffs(agg_a: dict, agg_b: dict) -> pd.DataFrame:
    """Per orientation, per subject present in both groups: each subject's
    OWN direction and magnitude in each group, plus their difference
    (A - B). This is the "difference of the mean directions, not the mean
    of the differences" quantity per subject: delta_angle_i is the
    difference between subject i's own well-defined direction in each
    condition, not the angle of a (possibly tiny, noise-dominated)
    difference vector. delta_mag_i is a plain scalar difference — no
    circular concerns for magnitude."""
    end_a = endpoint_stats(agg_a["per_subj"], ["subject"]).set_index(["orientation", "subject"])
    end_b = endpoint_stats(agg_b["per_subj"], ["subject"]).set_index(["orientation", "subject"])
    rows = []
    for ori in sorted(set(end_a.index.get_level_values(0)) & set(end_b.index.get_level_values(0))):
        a, b = end_a.loc[ori], end_b.loc[ori]
        for subj in a.index.intersection(b.index):
            aa, bb = a.loc[subj], b.loc[subj]
            delta_angle = ((aa["angle_deg"] - bb["angle_deg"] + 180) % 360) - 180
            rows.append(dict(orientation=ori, subject=subj,
                              angle_a=aa["angle_deg"], angle_b=bb["angle_deg"],
                              mag_a=aa["magnitude_deg"], mag_b=bb["magnitude_deg"],
                              delta_angle=delta_angle, delta_mag=aa["magnitude_deg"] - bb["magnitude_deg"]))
    return pd.DataFrame(rows)


def circular_diff_stats(deltas_deg: np.ndarray, n_boot: int = N_BOOT, ci: float = 95,
                        seed: int = 1) -> tuple:
    """Circular mean of per-subject angular differences, bootstrap CI, and
    a bootstrap two-sided p-value against H0: mean = 0 deg. Also returns R
    (mean resultant length, 0-1) as a reliability diagnostic — small R
    means the per-subject differences are scattered around the circle
    rather than agreeing on a rotation, so the mean itself is noisy (same
    role the endpoint-magnitude played for the old vector-difference
    approach, but now a direct property of the angles themselves rather
    than a side effect of a shrinking vector)."""
    n = len(deltas_deg)
    rad = np.radians(deltas_deg)
    c, s = np.cos(rad).mean(), np.sin(rad).mean()
    mean_deg = float(np.degrees(np.arctan2(s, c)))
    R = float(np.hypot(c, s))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.degrees(np.arctan2(np.sin(rad[idx]).mean(), np.cos(rad[idx]).mean()))
    wrapped = ((boot - mean_deg + 180) % 360) - 180  # CI around the observed mean
    lo_off, hi_off = np.percentile(wrapped, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    signed_vs_zero = ((boot + 180) % 360) - 180       # p-value: bootstrap distribution vs the fixed value 0
    p = float(min(1.0, 2 * min((signed_vs_zero <= 0).mean(), (signed_vs_zero >= 0).mean())))
    return mean_deg, mean_deg + lo_off, mean_deg + hi_off, R, p


def linear_diff_stats(deltas: np.ndarray, n_boot: int = N_BOOT, ci: float = 95, seed: int = 2) -> tuple:
    """Mean of per-subject scalar differences, bootstrap CI, bootstrap
    two-sided p-value against H0: mean = 0. No circular handling needed —
    magnitude is a plain non-negative linear quantity."""
    n = len(deltas)
    mean_val = float(np.mean(deltas))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = deltas[idx].mean()
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    p = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
    return mean_val, float(lo), float(hi), p


def _fdr(p: pd.Series) -> np.ndarray:
    q = np.full(len(p), np.nan)
    valid = p.notna().to_numpy()
    if valid.sum() > 0:
        q[valid] = multipletests(p[valid], method="fdr_bh")[1]
    return q


ORI_PERIOD = 180.0  # grating orientation is axis-symmetric (0 deg == 180 deg)


def circular_mean_deg(values_deg: np.ndarray) -> float:
    rad = np.radians(values_deg)
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())))


def circular_orientation_distance(a: np.ndarray, b: float, period: float = ORI_PERIOD) -> np.ndarray:
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def _pooled_per_subject(per_subj_diffs: pd.DataFrame, ori: float, col: str,
                        orientation_window_deg: float, circular: bool) -> pd.Series:
    """Values of `col` at a single orientation, or — if orientation_window_deg > 0 —
    pooled across neighbouring orientations within that window (circular
    distance, wrapped at ORI_PERIOD, so 172.5 and 7.5 deg count as close).
    Averaged WITHIN subject first so each subject contributes exactly one
    value regardless of how many orientations fell in the window — pooling
    without this step would let subjects with more usable orientations
    dominate the bootstrap (pseudoreplication)."""
    if orientation_window_deg > 0:
        dist = circular_orientation_distance(per_subj_diffs["orientation"].to_numpy(), ori)
        window = per_subj_diffs.loc[dist <= orientation_window_deg]
        if circular:
            return window.groupby("subject")[col].apply(lambda x: circular_mean_deg(x.to_numpy()))
        return window.groupby("subject")[col].mean()
    g = per_subj_diffs.loc[per_subj_diffs["orientation"] == ori]
    return g.set_index("subject")[col]


def direction_difference_tuning(per_subj_diffs: pd.DataFrame, n_boot: int = N_BOOT,
                                ci: float = 95, orientation_window_deg: float = 0.0) -> pd.DataFrame:
    """Per orientation: circular mean of each subject's OWN direction
    difference (A - B) — i.e. the difference of the mean directions,
    not the angle of the mean difference vector (see module docstring).
    Well-defined even when the two conditions are nearly identical, unlike
    the vector-difference approach this replaced, which degenerated to
    noise whenever the difference vector shrank toward zero.

    orientation_window_deg > 0 pools neighbouring orientations (circular
    distance, mod ORI_PERIOD) before averaging — each independent
    per-orientation point has substantial sampling noise (n~25 subjects,
    ~15 trials each), which can obscure a real, smooth trend across
    orientation; 0 (default) reproduces the original unsmoothed behaviour.
    """
    rows = []
    for ori in sorted(per_subj_diffs["orientation"].unique()):
        per_subject = _pooled_per_subject(per_subj_diffs, ori, "delta_angle",
                                          orientation_window_deg, circular=True)
        if len(per_subject) < 5:
            continue
        mean_deg, lo, hi, R, p = circular_diff_stats(per_subject.to_numpy(), n_boot, ci, seed=1)
        rows.append(dict(orientation=ori, n=len(per_subject), delta_deg=mean_deg,
                          delta_lo=lo, delta_hi=hi, R=R, p=p))
    res = pd.DataFrame(rows)
    res["q"] = _fdr(res["p"]) if len(res) else []
    return res


def magnitude_difference_tuning(per_subj_diffs: pd.DataFrame, n_boot: int = N_BOOT,
                                ci: float = 95, orientation_window_deg: float = 0.0) -> pd.DataFrame:
    """Per orientation: mean of each subject's own magnitude difference
    (A - B), bootstrap CI and bootstrap p-value. A plain paired
    comparison of a linear scalar — no circular machinery needed.
    orientation_window_deg: see direction_difference_tuning."""
    rows = []
    for ori in sorted(per_subj_diffs["orientation"].unique()):
        per_subject = _pooled_per_subject(per_subj_diffs, ori, "delta_mag",
                                          orientation_window_deg, circular=False)
        if len(per_subject) < 5:
            continue
        mean_val, lo, hi, p = linear_diff_stats(per_subject.to_numpy(), n_boot, ci, seed=2)
        rows.append(dict(orientation=ori, n=len(per_subject), delta_mag=mean_val, mag_lo=lo, mag_hi=hi, p=p))
    res = pd.DataFrame(rows)
    res["q"] = _fdr(res["p"]) if len(res) else []
    return res


def annotate_significance(ax, sig: pd.DataFrame, y_pos: float, color: str):
    hit = sig[sig["q"] < ALPHA]
    miss = sig[sig["q"] >= ALPHA]
    ax.scatter(miss["orientation"], np.full(len(miss), y_pos), marker=".", s=10,
              color="0.8", zorder=1, clip_on=False)
    ax.scatter(hit["orientation"], np.full(len(hit), y_pos), marker="*", s=30,
              color=color, zorder=4, clip_on=False)


def bar_prediction_deg(orientation: np.ndarray, perpendicular: bool = False) -> np.ndarray:
    """Predicted gaze-direction angle if gaze were biased along the
    grating bars themselves (default) or, for contrast, along the
    perpendicular / motion-energy axis. Orientation convention verified
    against PsychoPy's own rotation matrix — see extract_gaze_trajectories.py
    and plot_gaze_trajectories.render_gabor()."""
    theta = np.radians(orientation)
    if perpendicular:
        return np.degrees(np.arctan2(-np.sin(theta), np.cos(theta)))
    return np.degrees(np.arctan2(np.cos(theta), np.sin(theta)))


def plot_group(ax_angle, ax_mag, agg: dict, label: str, color: str):
    grand_end = endpoint_stats(agg["grand"], group_cols=[]).sort_values("orientation")
    subj_end = endpoint_stats(agg["per_subj"], group_cols=["subject"])

    orientations = grand_end["orientation"].to_numpy()
    angle_y = unwrap_by_orientation(grand_end)
    ci_lo, ci_hi = [], []
    mag_mean, mag_sem = [], []
    for ori, ay in zip(orientations, angle_y):
        sub = subj_end.loc[subj_end["orientation"] == ori]
        # bootstrap the raw (x, y) vectors, not pre-computed angles — obs
        # is guaranteed to equal ay (both are atan2(mean_y, mean_x) over
        # the same subject set), so no separate re-anchoring is needed,
        # just the unwrap-continuity shift (ay - obs, wrapped) also
        # applied to the raw grand line above.
        obs, lo, hi = bootstrap_vector_angle_ci(sub["x_deg"].to_numpy(), sub["y_deg"].to_numpy())
        shift = ((ay - obs + 180) % 360) - 180
        ci_lo.append(lo + shift)
        ci_hi.append(hi + shift)
        mag_mean.append(sub["magnitude_deg"].mean())
        mag_sem.append(sem(sub["magnitude_deg"].to_numpy()))
    ci_lo, ci_hi = np.array(ci_lo), np.array(ci_hi)
    mag_mean, mag_sem = np.array(mag_mean), np.array(mag_sem)

    ax_angle.scatter(subj_end["orientation"], subj_end["angle_deg"], s=6, color=color,
                      alpha=0.2, linewidth=0, zorder=2)
    ax_angle.fill_between(orientations, ci_lo, ci_hi, color=color, alpha=0.2, linewidth=0, zorder=2)
    ax_angle.plot(orientations, angle_y, color=color, lw=2, marker="o", ms=4, zorder=3)
    ax_angle.annotate(label, xy=(orientations[-1], angle_y[-1]),
                      xytext=(6, 0), textcoords="offset points", color=color,
                      fontsize=7.5, ha="left", va="center", fontweight="bold")

    ax_mag.scatter(subj_end["orientation"], subj_end["magnitude_deg"], s=6, color=color,
                   alpha=0.2, linewidth=0, zorder=2)
    ax_mag.fill_between(orientations, mag_mean - mag_sem, mag_mean + mag_sem,
                        color=color, alpha=0.2, linewidth=0, zorder=2)
    ax_mag.plot(orientations, mag_mean, color=color, lw=2, marker="o", ms=4, zorder=3)
    ax_mag.annotate(label, xy=(orientations[-1], mag_mean[-1]),
                    xytext=(6, 0), textcoords="offset points", color=color,
                    fontsize=7.5, ha="left", va="center", fontweight="bold")


def unwrap_reliable_runs(angle_deg: np.ndarray, reliable: np.ndarray) -> np.ndarray:
    """np.unwrap only within contiguous reliable stretches. np.unwrap
    assumes a *continuously drifting* underlying signal — applied across
    points that are actually independent noise (unreliable, near-zero
    difference magnitude), it accumulates spurious +-360 deg corrections
    from essentially random consecutive jumps and can drift the line
    arbitrarily far from the true (-180, 180] range before it gets clipped
    at the axis limits. Unreliable points are left at their raw (already
    principal-range, atan2 output) value instead."""
    out = angle_deg.copy()
    n = len(out)
    i = 0
    while i < n:
        if reliable[i]:
            j = i
            while j < n and reliable[j]:
                j += 1
            out[i:j] = np.degrees(np.unwrap(np.radians(out[i:j])))
            i = j
        else:
            i += 1
    return out


def plot_direction_difference(ax, dstats: pd.DataFrame, color: str = "0.15"):
    d = dstats.sort_values("orientation")
    reliable = (d["q"] < ALPHA).to_numpy()
    raw_angle = d["delta_deg"].to_numpy()  # already in (-180, 180]

    angle_y = unwrap_reliable_runs(raw_angle, reliable)
    shift = angle_y - raw_angle  # zero outside reliable runs, by construction
    lo = d["delta_lo"].to_numpy() + shift
    hi = d["delta_hi"].to_numpy() + shift

    # Every point gets its own independent error bar — a single point's CI
    # is just two numbers and needs no unwrap logic across neighbours
    # (that was only ever a problem for *connecting* noisy points with a
    # line). Only the connecting line is restricted to contiguous
    # reliable (individually-significant) stretches, since that implies
    # a trend the data may not actually support.
    ax.errorbar(d["orientation"], angle_y, yerr=[angle_y - lo, hi - angle_y],
               fmt="none", ecolor=color, elinewidth=1, alpha=0.5, capsize=0, zorder=3)
    line_y = np.where(reliable, angle_y, np.nan)
    ax.plot(d["orientation"], line_y, color=color, lw=2, zorder=4)
    ax.scatter(d.loc[reliable, "orientation"], angle_y[reliable], color=color, s=16, zorder=5)
    ax.scatter(d.loc[~reliable, "orientation"], angle_y[~reliable],
              facecolor="none", edgecolor=color, linewidth=0.8, s=16, zorder=5)
    annotate_significance(ax, dstats, y_pos=-196, color=color)


def plot_magnitude_difference(ax, mstats: pd.DataFrame, color: str = "0.15"):
    d = mstats.sort_values("orientation")
    ax.axhline(0, color="0.7", lw=0.7, ls="--", zorder=0)
    ax.errorbar(d["orientation"], d["delta_mag"], yerr=[d["delta_mag"] - d["mag_lo"], d["mag_hi"] - d["delta_mag"]],
               fmt="o-", color=color, lw=2, ms=4, ecolor=color, elinewidth=1, capsize=0, zorder=3)
    y0, y1 = ax.get_ylim()
    annotate_significance(ax, mstats, y_pos=min(0, y0) - 0.06 * (y1 - y0), color=color)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv-a", default="notes/data/gaze_trajectories_gabor.tsv")
    p.add_argument("--label-a", default=None)
    p.add_argument("--tsv-b", default="notes/data/gaze_trajectories_all.tsv")
    p.add_argument("--label-b", default=None)
    p.add_argument("--split-col", default=None,
                   help="Instead of comparing --tsv-a vs --tsv-b, split --tsv-a's trials "
                        "by this column (must have exactly 2 unique values), e.g. 'mapping'.")
    p.add_argument("--title", default=None)
    p.add_argument("--show-bar-prediction", action="store_true",
                   help="Overlay the theoretical angle-vs-orientation curves predicted if gaze "
                        "is biased parallel / perpendicular to the grating bars.")
    p.add_argument("--orientation-smooth-deg", type=float, default=0.0,
                   help="Pool neighbouring orientations within this circular distance (deg) before "
                        "computing the difference-of-directions/magnitude curves. 0 (default) = "
                        "no smoothing (independent per-orientation points, as before).")
    p.add_argument("--out", default="notes/figures/gaze_direction_tuning.pdf")
    args = p.parse_args()

    if args.split_col:
        df = pd.read_csv(args.tsv_a, sep="\t", dtype={"subject": str})
        vals = sorted(df[args.split_col].unique())
        if len(vals) != 2:
            p.error(f"--split-col {args.split_col!r} must have exactly 2 unique values, got {vals}")
        label_a = args.label_a or str(vals[0])
        label_b = args.label_b or str(vals[1])
        agg_a = aggregate_df(df[df[args.split_col] == vals[0]], qc_label=f"{args.tsv_a} [{label_a}]")
        agg_b = aggregate_df(df[df[args.split_col] == vals[1]], qc_label=f"{args.tsv_a} [{label_b}]")
        title = args.title or f"Gaze-direction tuning by {args.split_col}"
    else:
        agg_a, agg_b = load_and_aggregate(args.tsv_a), load_and_aggregate(args.tsv_b)
        label_a, label_b = args.label_a or "Group A", args.label_b or "Group B"
        title = args.title or "Gaze-direction tuning by trial epoch"

    n_subj = len(set(agg_a["subjects"]) & set(agg_b["subjects"]))
    per_subj_diffs = paired_per_subject_diffs(agg_a, agg_b)
    dir_stats = direction_difference_tuning(per_subj_diffs, orientation_window_deg=args.orientation_smooth_deg)
    mag_stats = magnitude_difference_tuning(per_subj_diffs, orientation_window_deg=args.orientation_smooth_deg)
    n_sig_dir = int((dir_stats["q"] < ALPHA).sum())
    n_sig_mag = int((mag_stats["q"] < ALPHA).sum())
    print(f"Direction difference, bootstrap test per orientation (n={len(dir_stats)} tested, "
          f"FDR q<{ALPHA}): {n_sig_dir}/{len(dir_stats)} significant")
    print(f"Magnitude difference, bootstrap test per orientation (n={len(mag_stats)} tested, "
          f"FDR q<{ALPHA}): {n_sig_mag}/{len(mag_stats)} significant")

    fig, ((ax_angle, ax_mag), (ax_dangle, ax_dmag)) = plt.subplots(
        2, 2, figsize=(9, 8), constrained_layout=True)

    plot_group(ax_angle, ax_mag, agg_a, label_a, PALETTE[0])
    plot_group(ax_angle, ax_mag, agg_b, label_b, PALETTE[1])

    if args.show_bar_prediction:
        ori_grid = np.linspace(1, 179, 200)
        ax_angle.plot(ori_grid, bar_prediction_deg(ori_grid), color="0.3", lw=1.1, ls="--", zorder=1)
        ax_angle.annotate("Predicted: parallel to bars", xy=(20, bar_prediction_deg(np.array([20]))[0]),
                          xytext=(-8, 55), textcoords="offset points", fontsize=6.5, color="0.3",
                          ha="left",
                          arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=70",
                                          color="0.3", lw=0.8, mutation_scale=9, shrinkA=2, shrinkB=6))
        ax_angle.plot(ori_grid, bar_prediction_deg(ori_grid, perpendicular=True),
                      color="0.6", lw=1.1, ls=":", zorder=1)
        ax_angle.annotate("Predicted: perpendicular (motion axis)",
                          xy=(20, bar_prediction_deg(np.array([20]), perpendicular=True)[0]),
                          xytext=(-8, -60), textcoords="offset points", fontsize=6.5, color="0.6",
                          ha="left",
                          arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=-70",
                                          color="0.6", lw=0.8, mutation_scale=9, shrinkA=2, shrinkB=6))

    ori_ticks = [0, 45, 90, 135, 180]
    ax_angle.set_xticks(ori_ticks)
    ax_angle.set_xlabel("Grating orientation (deg)")
    ax_angle.set_ylabel("Gaze direction (deg)")
    ax_angle.set_yticks([-180, -90, 0, 90, 180])
    for y, ref in ((0, "Right"), (90, "Up"), (-90, "Down"), (180, "Left")):
        ax_angle.axhline(y, color="0.75", lw=0.6, ls="--", zorder=0)
        ax_angle.text(182, y, ref, fontsize=6.5, color="0.4", va="center", ha="left")
    annotate_significance(ax_angle, dir_stats, y_pos=-196, color="0.15")
    ax_angle.set_xlim(0, 205)
    ax_angle.set_ylim(-200, 200)
    ax_angle.set_title("Endpoint direction (shaded: 95% bootstrap CI)", fontsize=9)
    sns.despine(ax=ax_angle, offset=3, trim=True)

    ax_mag.set_xticks(ori_ticks)
    ax_mag.set_xlim(0, 205)
    y0, y1 = ax_mag.get_ylim()
    y_sig = -0.06 * (y1 - y0)
    annotate_significance(ax_mag, mag_stats, y_pos=y_sig, color="0.15")
    ax_mag.set_ylim(bottom=y_sig - 0.02 * (y1 - y0))
    ax_mag.set_xlabel("Grating orientation (deg)")
    ax_mag.set_ylabel("Endpoint displacement (deg)")
    ax_mag.set_title("Endpoint magnitude (shaded: +-1 SEM)", fontsize=9)
    sns.despine(ax=ax_mag, offset=3, trim=True)

    plot_direction_difference(ax_dangle, dir_stats)
    plot_magnitude_difference(ax_dmag, mag_stats)

    ax_dangle.set_xticks(ori_ticks)
    ax_dangle.set_xlabel("Grating orientation (deg)")
    ax_dangle.set_ylabel(f"{label_a} minus {label_b} (deg)")
    ax_dangle.set_yticks([-180, -90, 0, 90, 180])
    for y, ref in ((0, "Right"), (90, "Up"), (-90, "Down"), (180, "Left")):
        ax_dangle.axhline(y, color="0.75", lw=0.6, ls="--", zorder=0)
        ax_dangle.text(182, y, ref, fontsize=6.5, color="0.4", va="center", ha="left")
    ax_dangle.set_xlim(0, 205)
    ax_dangle.set_ylim(-200, 200)
    smooth_note = (f", {args.orientation_smooth_deg:g} deg orientation-pooled"
                   if args.orientation_smooth_deg > 0 else "")
    ax_dangle.set_title(f"Difference of mean directions (bootstrap 95% CI{smooth_note})\n"
                        "error bars = per-point CI; line connects only individually-significant runs",
                        fontsize=8)
    sns.despine(ax=ax_dangle, offset=3, trim=True)

    ax_dmag.set_xticks(ori_ticks)
    ax_dmag.set_xlim(0, 205)
    dy0, dy1 = ax_dmag.get_ylim()
    ax_dmag.set_ylim(bottom=min(0, dy0) - 0.08 * (dy1 - dy0))  # room for the significance markers
    ax_dmag.set_xlabel("Grating orientation (deg)")
    ax_dmag.set_ylabel(f"{label_a} minus {label_b} (deg)")
    ax_dmag.set_title(f"Difference of mean magnitudes (bootstrap 95% CI{smooth_note})", fontsize=9)
    sns.despine(ax=ax_dmag, offset=3, trim=True)

    fig.suptitle(f"{title} (N={n_subj} subjects in both; dots = individual subjects, "
                 f"line = grand average; * = FDR q<{ALPHA}, bootstrap test per orientation — "
                 f"direction {n_sig_dir}/{len(dir_stats)}, magnitude {n_sig_mag}/{len(mag_stats)})",
                 fontsize=8.2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
