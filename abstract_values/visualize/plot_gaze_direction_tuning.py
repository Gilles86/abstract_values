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

Significance
------------
Per orientation, paired across subjects present in BOTH groups: build
each subject's (x, y) endpoint difference (group A - group B), then run
a one-sample Hotelling's T^2 test against H0: mean difference = (0, 0).
This tests direction AND magnitude jointly (they're both derived from
the same (x, y) vector, so testing them as two separate marginal tests
would double-count the same underlying difference) — the panels display
the same per-orientation q-value (Benjamini-Hochberg FDR across
orientations) as a marker strip in both panels, since it's one test
about one underlying quantity shown two ways.

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
from scipy import stats as sps
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


def hotelling_t2_1sample_pvalue(diffs: np.ndarray) -> float:
    """diffs: (n, 2) paired per-subject (dx, dy). p-value for
    H0: population mean difference vector = (0, 0)."""
    n, k = diffs.shape
    if n <= k:
        return np.nan
    mean = diffs.mean(axis=0)
    cov = np.cov(diffs, rowvar=False, ddof=1)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.nan
    t2 = float(n * mean @ inv_cov @ mean)
    f_stat = t2 * (n - k) / (k * (n - 1))
    return float(1 - sps.f.cdf(f_stat, k, n - k))


def paired_diff_vectors(agg_a: dict, agg_b: dict) -> dict:
    """Per orientation: (n_paired_subjects, 2) array of each subject's
    (x, y) endpoint difference (A - B), restricted to subjects with usable
    trials in both groups at that orientation. Shared by paired_significance
    (Hotelling's T^2 needs these) and difference_tuning (the CI band needs
    the same paired set, not the full per_subj table)."""
    end_a = endpoint_stats(agg_a["per_subj"], ["subject"]).set_index(["orientation", "subject"])
    end_b = endpoint_stats(agg_b["per_subj"], ["subject"]).set_index(["orientation", "subject"])
    out = {}
    for ori in sorted(set(end_a.index.get_level_values(0)) & set(end_b.index.get_level_values(0))):
        a = end_a.loc[ori][["x_deg", "y_deg"]]
        b = end_b.loc[ori][["x_deg", "y_deg"]]
        common = a.index.intersection(b.index)
        out[ori] = (a.loc[common] - b.loc[common]).to_numpy()
    return out


def paired_significance(diff_vectors: dict) -> pd.DataFrame:
    """Per orientation: paired Hotelling's T^2 test on the (x, y) endpoint
    difference vectors against H0: mean = (0, 0). Returns orientation,
    n (paired subjects), p, q (BH-FDR across orientations)."""
    rows = []
    for ori, diffs in diff_vectors.items():
        n = len(diffs)
        p = hotelling_t2_1sample_pvalue(diffs) if n >= 5 else np.nan
        rows.append(dict(orientation=ori, n=n, p=p))
    res = pd.DataFrame(rows)
    q = np.full(len(res), np.nan)
    valid = res["p"].notna().to_numpy()
    if valid.sum() > 0:
        q[valid] = multipletests(res.loc[valid, "p"], method="fdr_bh")[1]
    res["q"] = q
    return res


def difference_tuning(diff_vectors: dict, n_boot: int = N_BOOT, ci: float = 95,
                      seed: int = 1) -> pd.DataFrame:
    """Per orientation: angle and magnitude of the mean per-subject (x, y)
    difference vector (A - B), each with a subject-level bootstrap CI.
    mean(a_i - b_i) == mean(a_i) - mean(b_i), so this is exactly the
    "grand_a minus grand_b" difference restricted to paired subjects —
    same quantity plot_group's two lines would subtract, computed once
    with its own uncertainty rather than read off two separate bands."""
    rng = np.random.default_rng(seed)
    rows = []
    for ori in sorted(diff_vectors):
        d = diff_vectors[ori]
        n = len(d)
        if n < 5:
            continue
        mean_vec = d.mean(axis=0)
        angle = float(np.degrees(np.arctan2(mean_vec[1], mean_vec[0])))
        mag = float(np.hypot(*mean_vec))
        boot_angle = np.empty(n_boot)
        boot_mag = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            bm = d[idx].mean(axis=0)
            boot_angle[i] = np.degrees(np.arctan2(bm[1], bm[0]))
            boot_mag[i] = np.hypot(*bm)
        wrapped = ((boot_angle - angle + 180) % 360) - 180
        a_lo, a_hi = np.percentile(wrapped, [(100 - ci) / 2, 100 - (100 - ci) / 2])
        m_lo, m_hi = np.percentile(boot_mag, [(100 - ci) / 2, 100 - (100 - ci) / 2])
        rows.append(dict(orientation=ori, n=n, angle_deg=angle, angle_lo=angle + a_lo,
                          angle_hi=angle + a_hi, magnitude_deg=mag, mag_lo=m_lo, mag_hi=m_hi))
    return pd.DataFrame(rows)


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


def plot_difference(ax_angle, ax_mag, diff_stats: pd.DataFrame, sig: pd.DataFrame,
                    label_a: str, label_b: str, color: str = "0.15"):
    diff_sorted = diff_stats.merge(sig[["orientation", "q"]], on="orientation", how="left").sort_values("orientation")
    reliable = (diff_sorted["q"] < ALPHA).to_numpy()
    raw_angle = diff_sorted["angle_deg"].to_numpy()  # atan2 output, already in (-180, 180]

    angle_y = unwrap_reliable_runs(raw_angle, reliable)
    shift = angle_y - raw_angle  # zero outside reliable runs, by construction
    lo = diff_sorted["angle_lo"].to_numpy() + shift
    hi = diff_sorted["angle_hi"].to_numpy() + shift

    # Only connect/shade contiguous reliable stretches — an unreliable
    # point (near-zero difference magnitude) gets a bare marker at its raw
    # angle, not a line implying a trend that isn't statistically there.
    line_y = np.where(reliable, angle_y, np.nan)
    lo_masked = np.where(reliable, lo, np.nan)
    hi_masked = np.where(reliable, hi, np.nan)

    ax_angle.fill_between(diff_sorted["orientation"], lo_masked, hi_masked,
                          color=color, alpha=0.2, linewidth=0, zorder=2)
    ax_angle.plot(diff_sorted["orientation"], line_y, color=color, lw=2, zorder=3)
    ax_angle.scatter(diff_sorted.loc[reliable, "orientation"], angle_y[reliable],
                     color=color, s=16, zorder=4)
    ax_angle.scatter(diff_sorted.loc[~reliable, "orientation"], raw_angle[~reliable],
                     facecolor="none", edgecolor=color, linewidth=0.8, s=16, zorder=4)
    annotate_significance(ax_angle, sig, y_pos=-196, color=color)

    ax_mag.axhline(0, color="0.7", lw=0.7, ls="--", zorder=0)
    ax_mag.fill_between(diff_sorted["orientation"], diff_sorted["mag_lo"], diff_sorted["mag_hi"],
                        color=color, alpha=0.2, linewidth=0, zorder=2)
    ax_mag.plot(diff_sorted["orientation"], diff_sorted["magnitude_deg"], color=color, lw=2,
               marker="o", ms=4, zorder=3)
    y0, y1 = ax_mag.get_ylim()
    annotate_significance(ax_mag, sig, y_pos=min(0, y0) - 0.06 * (y1 - y0), color=color)


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
    diff_vectors = paired_diff_vectors(agg_a, agg_b)
    sig = paired_significance(diff_vectors)
    diff_stats = difference_tuning(diff_vectors)
    n_sig = int((sig["q"] < ALPHA).sum())
    print(f"Paired Hotelling's T^2 per orientation (n={len(sig)} orientations tested, "
          f"FDR q<{ALPHA}): {n_sig}/{len(sig)} significant")

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
    annotate_significance(ax_angle, sig, y_pos=-196, color="0.15")
    ax_angle.set_xlim(0, 205)
    ax_angle.set_ylim(-200, 200)
    ax_angle.set_title("Endpoint direction (shaded: 95% bootstrap CI)", fontsize=9)
    sns.despine(ax=ax_angle, offset=3, trim=True)

    ax_mag.set_xticks(ori_ticks)
    ax_mag.set_xlim(0, 205)
    y0, y1 = ax_mag.get_ylim()
    y_sig = -0.06 * (y1 - y0)
    annotate_significance(ax_mag, sig, y_pos=y_sig, color="0.15")
    ax_mag.set_ylim(bottom=y_sig - 0.02 * (y1 - y0))
    ax_mag.set_xlabel("Grating orientation (deg)")
    ax_mag.set_ylabel("Endpoint displacement (deg)")
    ax_mag.set_title("Endpoint magnitude (shaded: +-1 SEM)", fontsize=9)
    sns.despine(ax=ax_mag, offset=3, trim=True)

    plot_difference(ax_dangle, ax_dmag, diff_stats, sig, label_a, label_b)

    ax_dangle.set_xticks(ori_ticks)
    ax_dangle.set_xlabel("Grating orientation (deg)")
    ax_dangle.set_ylabel(f"{label_a} minus {label_b} (deg)")
    ax_dangle.set_yticks([-180, -90, 0, 90, 180])
    for y, ref in ((0, "Right"), (90, "Up"), (-90, "Down"), (180, "Left")):
        ax_dangle.axhline(y, color="0.75", lw=0.6, ls="--", zorder=0)
        ax_dangle.text(182, y, ref, fontsize=6.5, color="0.4", va="center", ha="left")
    ax_dangle.set_xlim(0, 205)
    ax_dangle.set_ylim(-200, 200)
    ax_dangle.set_title("Difference in endpoint direction (95% bootstrap CI)\n"
                        "filled dots + line = individually significant; open dots = not (shown, not connected)",
                        fontsize=8)
    sns.despine(ax=ax_dangle, offset=3, trim=True)

    ax_dmag.set_xticks(ori_ticks)
    ax_dmag.set_xlim(0, 205)
    dy0, dy1 = ax_dmag.get_ylim()
    ax_dmag.set_ylim(bottom=min(0, dy0) - 0.08 * (dy1 - dy0))  # room for the significance markers
    ax_dmag.set_xlabel("Grating orientation (deg)")
    ax_dmag.set_ylabel(f"|{label_a} - {label_b}| (deg)")
    ax_dmag.set_title("Difference vector magnitude (95% bootstrap CI)", fontsize=9)
    sns.despine(ax=ax_dmag, offset=3, trim=True)

    fig.suptitle(f"{title} (N={n_subj} subjects in both; dots = individual subjects, "
                 f"line = grand average; * = FDR q<{ALPHA}, paired Hotelling's T^2 per orientation, "
                 f"{n_sig}/{len(sig)} orientations)", fontsize=8.2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
