"""V1 gabor-orientation decoding QA: per-subject + group summaries.

For each subject (and a group summary), shows how well V1 reconstructs the
presented gabor orientation, from `decode_gabor` outputs at:
    derivatives/decoding/gabor/sub-XX/func/sub-XX_mask-BensonV1_nvoxels-0_noise-full[_smoothed][_lambda-X]_pars.tsv

Pages:

1. Per-subject — two panels:
   - Average posterior aligned to true orientation (shaded ±1 SEM across trials)
   - Distribution of circular decoding errors (MAP − truth)
   Unsmoothed and smoothed variants overlaid, direct-labeled.

2. Group summary (pooled trials): same two panels.

3. Group "decoded vs true" 2D histogram: per smoothing variant, joint density
   of (truth, posterior-mean decoded). Diagonal = perfect decoding. Both
   a pooled-sessions view and a session-split (ses-1 vs ses-2) view.

4. Per-subject circular correlation swarmplot: each subject contributes one
   dot per smoothing variant. Per-subject value is the **mean** of per-run
   circular correlations (averaged within runs first, then across runs),
   matching the conventional within-subject summary.

All decoded values are out-of-sample (leave-one-run-out cross-validation in
the underlying `decode_gabor` script).

Usage:
    python -m abstract_values.visualize.check_v1_orientation_decoding
    python -m abstract_values.visualize.check_v1_orientation_decoding --subjects 08 09
    python -m abstract_values.visualize.check_v1_orientation_decoding --lambd 0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from abstract_values.utils.data import BIDS_FOLDER

# ── Vision-science house style (per scientific-figures skill) ────────────────
mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "mathtext.fontset": "stixsans",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 150,
    "savefig.dpi": 300,
})
sns.set_context("paper")

DERIV = Path(BIDS_FOLDER) / "derivatives" / "decoding" / "gabor"
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "v1_orientation_decoding.pdf"

# Gabors are π-periodic (0° ≡ 180°).
PERIOD = np.pi

# Hand-picked palette. Default is unsmoothed-only (smoothing has not moved
# any of these analyses meaningfully so far); pass --include-smoothed to add
# the smoothed variant in red.
SMOOTH_UNSMOOTHED = ("",          "Unsmoothed", "#3B5BA5")
SMOOTH_SMOOTHED   = ("_smoothed", "Smoothed",   "#C44E52")
SMOOTH_VARIANTS_DEFAULT = [SMOOTH_UNSMOOTHED]
SMOOTH_VARIANTS_ALL     = [SMOOTH_UNSMOOTHED, SMOOTH_SMOOTHED]
# Reassigned in main() if --include-smoothed is passed.
SMOOTH_VARIANTS = SMOOTH_VARIANTS_DEFAULT

# Per-condition palette: two abstract-value mappings, counterbalanced by
# subject parity. cdf ≈ "low orientations → low CHF"; inverse_cdf reverses.
# Match the Bedi et al. 2026 behavior-notebook palette so decoding figures
# read the same as the behavioral panels.
CONDITIONS = [
    ("cdf",          "#E76F51"),   # warm coral
    ("inverse_cdf",  "#2A9D8F"),   # dark teal
]


def session_to_condition(subject_id: str, session: int) -> str:
    """Which orientation→CHF mapping the subject used in that session.

    Mirrors `abstract_values.utils.data.Subject.get_mapping`: even-numbered
    subjects start with cdf (ses-1) then inverse_cdf (ses-2); odd subjects
    the other way around.
    """
    num = int("".join(c for c in subject_id if c.isdigit()))
    if num % 2 == 0:
        return "cdf" if session == 1 else "inverse_cdf"
    return "inverse_cdf" if session == 1 else "cdf"


def load_posteriors(subject: str, smoothed_suffix: str, mask: str = "BensonV1",
                    n_voxels: int = 0, lambd: float = 0.1):
    """Return dict with truth/grid/posteriors/run_keys/conditions for one (subject, smoothing)."""
    lam_tag = f"_lambda-{lambd}" if lambd != 0.0 else ""
    fn = (DERIV / f"sub-{subject}" / "func"
          / f"sub-{subject}_mask-{mask}_nvoxels-{n_voxels}"
            f"_noise-full{smoothed_suffix}{lam_tag}_pars.tsv")
    if not fn.exists():
        return None
    df = pd.read_csv(fn, sep="\t", index_col=[0, 1, 2])
    truth = df["true_orientation_rad"].to_numpy(dtype=np.float32)
    grid = np.asarray(df.columns[1:], dtype=np.float32)
    posteriors = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)
    sess = df.index.get_level_values("session").to_numpy(dtype=np.int32)
    run  = df.index.get_level_values("run").to_numpy(dtype=np.int32)
    run_keys = sess * 100 + run
    # Map each trial's session to its condition (subject-specific assignment).
    conditions = np.array([session_to_condition(subject, int(s)) for s in sess])
    return {"truth": truth, "grid": grid, "posteriors": posteriors,
            "run_keys": run_keys, "conditions": conditions}


def aligned_posterior(truth: np.ndarray, grid: np.ndarray,
                      posteriors: np.ndarray):
    """Shift each row so the true orientation lands at column N//2."""
    n_trials, n_grid = posteriors.shape
    true_idx = np.argmin(np.abs(grid[:, None] - truth[None, :]), axis=0)
    center = n_grid // 2
    aligned = np.empty_like(posteriors)
    for i in range(n_trials):
        aligned[i] = np.roll(posteriors[i], center - true_idx[i])
    x_axis = grid - grid[center]
    return aligned, x_axis


def map_estimate(grid: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
    """Maximum-a-posteriori orientation (argmax on the grid)."""
    return grid[posteriors.argmax(axis=1)]


def posterior_circular_mean(grid: np.ndarray, posteriors: np.ndarray,
                            period: float = PERIOD) -> np.ndarray:
    """Per-trial circular mean of the posterior on a `period`-periodic axis.

    For orientation (period = π) we double the angles so they live on the full
    2π circle, compute the probability-weighted resultant vector, take its
    angle, and halve to return to the half-circle. Always in [0, period).
    """
    scale = 2 * np.pi / period
    a = grid.astype(np.float64) * scale                # (N_grid,)
    sin_a = np.sin(a)
    cos_a = np.cos(a)
    s = posteriors @ sin_a
    c = posteriors @ cos_a
    mean_doubled = np.arctan2(s, c)                    # in (−π, π]
    mean = (mean_doubled / scale) % period
    return mean.astype(np.float32)


def circular_error(decoded: np.ndarray, truth: np.ndarray,
                   period: float = PERIOD) -> np.ndarray:
    d = (decoded - truth) % period
    return np.where(d > period / 2, d - period, d)


def circular_correlation(true_rad: np.ndarray, pred_rad: np.ndarray,
                         period: float = PERIOD) -> float:
    """Jammalamadaka–Sarma circular correlation for `period`-periodic angles.

    Orientations are doubled so they live on the full 2π circle, then the
    standard formula applies.
    """
    if len(true_rad) < 2:
        return float("nan")
    scale = 2 * np.pi / period
    a = (true_rad * scale).astype(np.float64)
    b = (pred_rad * scale).astype(np.float64)
    a_mean = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    b_mean = np.arctan2(np.sin(b).mean(), np.cos(b).mean())
    sa = np.sin(a - a_mean)
    sb = np.sin(b - b_mean)
    num = (sa * sb).sum()
    den = np.sqrt((sa * sa).sum() * (sb * sb).sum())
    return float(num / den) if den > 0 else float("nan")


def per_subject_circular_corr(d: dict) -> float:
    """Mean per-run circular correlation across all runs of one subject.

    Within-run circular correlation first, then average — the conventional
    within-subject summary.
    """
    truth = d["truth"]
    decoded = posterior_circular_mean(d["grid"], d["posteriors"])
    keys = d["run_keys"]
    rs = [circular_correlation(truth[keys == k], decoded[keys == k])
          for k in np.unique(keys)]
    rs = np.array(rs)
    return float(np.nanmean(rs)) if np.isfinite(rs).any() else float("nan")


# ── plotting helpers ─────────────────────────────────────────────────────────

def _annotate_chance(ax, n_grid):
    """Thin gray dashed baseline at uniform-posterior level."""
    ax.axhline(1 / n_grid, color="0.7", lw=0.6, ls="--", zorder=0)


def _draw_aligned_posterior(ax, datasets):
    """Average posterior aligned to true orientation, ±1 SEM, direct-labelled."""
    for label, color, d in datasets:
        truth, grid, posteriors = d["truth"], d["grid"], d["posteriors"]
        aligned, x = aligned_posterior(truth, grid, posteriors)
        mean = aligned.mean(axis=0)
        se = aligned.std(axis=0, ddof=1) / np.sqrt(aligned.shape[0])
        x_deg = np.degrees(x)
        ax.plot(x_deg, mean, color=color, lw=1.5, label=label)
        ax.fill_between(x_deg, mean - se, mean + se, color=color, alpha=0.22,
                        linewidth=0)
        # direct label at right edge of the curve
        ax.text(x_deg[-1] + 1.5, mean[-1], label, color=color,
                fontsize=8, va="center", ha="left")

    _annotate_chance(ax, aligned.shape[1])
    ax.axvline(0, color="black", lw=0.5, ls=":", zorder=0)
    ax.set_xlabel("Offset from true orientation (°)")
    ax.set_ylabel("Mean posterior probability")
    ax.set_ylim(bottom=0)
    # Pick clean ticks at extremes + zero
    xmin, xmax = ax.get_xlim()
    ax.set_xticks([round(xmin), 0, round(xmax)])


def _draw_error_hist(ax, datasets):
    bins = np.linspace(-90, 90, 19)
    for label, color, d in datasets:
        decoded = posterior_circular_mean(d["grid"], d["posteriors"])
        err = np.degrees(circular_error(decoded, d["truth"]))
        mae = float(np.abs(err).mean())
        ax.hist(err, bins=bins, histtype="step", color=color, lw=1.8,
                label=f"{label} · MAE={mae:.1f}°")
    ax.axvline(0, color="black", lw=0.5, ls=":", zorder=0)
    ax.set_xlabel("Decoding error (Mean − true) (°)")
    ax.set_ylabel("Trial count")
    ax.set_ylim(bottom=0)
    ax.set_xticks([-90, -45, 0, 45, 90])
    # MAE labels — direct, no legend frame
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=8,
              handlelength=1.4)


def plot_subject_page(subject: str, datasets, lambd: float):
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.7), constrained_layout=True)
    fig.suptitle(f"sub-{subject}  ·  V1 (BensonV1, all voxels, λ={lambd})",
                 fontsize=10, y=1.02)
    _draw_aligned_posterior(axes[0], datasets)
    _draw_error_hist(axes[1], datasets)
    sns.despine(fig=fig, offset=5, trim=True)
    return fig


def plot_group_summary_page(group: dict, lambd: float):
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.7), constrained_layout=True)
    fig.suptitle(f"Group  ·  V1 (pooled trials, λ={lambd})",
                 fontsize=10, y=1.02)

    # Build per-variant "merged" dataset for re-use of the same helpers
    merged_datasets = []
    for suffix, label, color in SMOOTH_VARIANTS:
        items = group.get(suffix)
        if not items:
            continue
        truth = np.concatenate([d["truth"] for d in items])
        grid = items[0]["grid"]                       # same across subjects
        posteriors = np.concatenate([d["posteriors"] for d in items], axis=0)
        merged_datasets.append((f"{label}  (n_subj={len(items)})", color,
                                {"truth": truth, "grid": grid,
                                 "posteriors": posteriors}))

    _draw_aligned_posterior(axes[0], merged_datasets)
    _draw_error_hist(axes[1], merged_datasets)
    sns.despine(fig=fig, offset=5, trim=True)
    return fig


# Cardinal-centered binning helpers. 12 bins of 15° each, centered on
# {0, 15, 30, 45, ...}. Shift data by half-a-bin so that cardinal 0°/45°/90°/
# 135° land at bin CENTERS, not bin edges — useful because V1 orientation
# decoding tends to over-represent the cardinals (vertical/horizontal bias).
_N_BINS = 24                                     # 7.5° bin width, matches grid
_HALF_BIN_DEG = 180.0 / _N_BINS / 2.0            # 3.75° shift puts 0/45/90/135 at centers


def _cardinal_bin(values_deg: np.ndarray) -> np.ndarray:
    """Shift orientation degrees so cardinal-centered bins begin at 0."""
    return (values_deg + _HALF_BIN_DEG) % 180


def _cardinal_ticks():
    """(positions, labels) for cardinal ticks in the SHIFTED display space."""
    cardinals = [0, 45, 90, 135]
    positions = [(c + _HALF_BIN_DEG) for c in cardinals]
    labels = [f"{c}°" for c in cardinals]
    return positions, labels


def _plot_one_heatmap(ax, truth_pool, decoded_pool, title, title_color):
    """Render a single decoded-vs-true 2D histogram into `ax`. Returns the im."""
    if not truth_pool:
        ax.set_axis_off()
        ax.set_title(f"{title}  (no data)", fontsize=9)
        return None
    truth = _cardinal_bin(np.degrees(np.concatenate(truth_pool)) % 180)
    decoded = _cardinal_bin(np.degrees(np.concatenate(decoded_pool)) % 180)
    edges = np.linspace(0, 180, _N_BINS + 1)
    H, xe, ye = np.histogram2d(truth, decoded, bins=edges)
    H = H / H.sum() * 100 if H.sum() > 0 else H
    im = ax.imshow(H.T, origin="lower",
                    extent=(xe[0], xe[-1], ye[0], ye[-1]),
                    cmap="mako", aspect="equal", interpolation="nearest")
    # The shifted-space identity line is still the diagonal of the box
    ax.plot([0, 180], [0, 180], color="white", lw=0.8, ls="--",
            alpha=0.5, zorder=1)
    pos, lab = _cardinal_ticks()
    ax.set_xticks(pos); ax.set_xticklabels(lab)
    ax.set_yticks(pos); ax.set_yticklabels(lab)
    ax.set_title(title, fontsize=9, color=title_color)
    return im


def plot_decoded_vs_truth_page(group: dict, lambd: float):
    """Pooled 2D histogram of (truth, decoded) per smoothing variant."""
    fig, axes = plt.subplots(1, len(SMOOTH_VARIANTS),
                              figsize=(3.4 * len(SMOOTH_VARIANTS), 3.2),
                              constrained_layout=True,
                              sharex=True, sharey=True,
                              squeeze=False)
    axes = axes[0]   # row-1, all columns
    fig.suptitle(f"Group  ·  Decoded vs presented orientation (λ={lambd})",
                 fontsize=10, y=1.02)
    im = None
    for ax, (suffix, label, color) in zip(axes, SMOOTH_VARIANTS):
        items = group.get(suffix, [])
        n_trials = sum(d["truth"].shape[0] for d in items)
        title = f"{label}  (n_subj={len(items)}, n_trials={n_trials})"
        im = _plot_one_heatmap(
            ax,
            [d["truth"] for d in items],
            [posterior_circular_mean(d["grid"], d["posteriors"]) for d in items],
            title, color,
        ) or im
    axes[0].set_ylabel("Decoded orientation (°)")
    for ax in axes:
        ax.set_xlabel("True orientation (°)")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
        cbar.set_label("Trials (%)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    return fig


def _trials_for_condition(items, condition_label):
    """Pool truth + decoded-mean across subjects for one condition slice."""
    truth_pool, decoded_pool, n_subj, n_trials = [], [], 0, 0
    for d in items:
        sel = d["conditions"] == condition_label
        if not sel.any():
            continue
        n_subj += 1
        truth_pool.append(d["truth"][sel])
        decoded_pool.append(
            posterior_circular_mean(d["grid"], d["posteriors"][sel])
        )
        n_trials += int(sel.sum())
    return truth_pool, decoded_pool, n_subj, n_trials


def _cardinal_hist(truth_pool, decoded_pool):
    """2D histogram on cardinal-centered bins, normalised to percent."""
    if not truth_pool:
        return None, None
    truth = _cardinal_bin(np.degrees(np.concatenate(truth_pool)) % 180)
    decoded = _cardinal_bin(np.degrees(np.concatenate(decoded_pool)) % 180)
    edges = np.linspace(0, 180, _N_BINS + 1)
    H, _, _ = np.histogram2d(truth, decoded, bins=edges)
    H = H / H.sum() * 100 if H.sum() > 0 else H
    return H, edges


def plot_decoded_vs_truth_by_condition_page(group: dict, lambd: float):
    smooth_variants = SMOOTH_VARIANTS
    """2D histograms split by smoothing × condition (cdf / inverse_cdf).

    Conditions are the two orientation→CHF mappings the subject is exposed
    to across the two sessions (counterbalanced by subject parity). Aligning
    by condition rather than session pools across subjects whose session→
    condition assignment is reversed.

    Note (per decode_gabor source): folds pool across sessions during
    *training*, so this slices OUT-OF-SAMPLE trials by their condition —
    a sanity check on whether V1 decoding quality differs across mappings,
    not a within-condition-trained decoder.
    """
    n_cols = len(CONDITIONS) + 1   # +1 for the difference panel
    fig, axes = plt.subplots(len(smooth_variants), n_cols,
                              figsize=(3.0 * n_cols, 3.4 * len(smooth_variants)),
                              constrained_layout=True,
                              sharex=True, sharey=True,
                              squeeze=False)
    fig.suptitle(f"Group  ·  Decoded vs presented, split by condition (λ={lambd})",
                 fontsize=10, y=1.01)

    pos, lab = _cardinal_ticks()
    im_main = None
    im_diff = None
    for row, (suffix, smooth_label, smooth_color) in enumerate(smooth_variants):
        items = group.get(suffix, [])
        hists = {}
        for col, (cond_label, cond_color) in enumerate(CONDITIONS):
            ax = axes[row, col]
            truth_pool, decoded_pool, n_subj, n_trials = _trials_for_condition(
                items, cond_label)
            H, edges = _cardinal_hist(truth_pool, decoded_pool)
            if H is None:
                ax.set_axis_off()
                ax.set_title(f"{smooth_label} · {cond_label} (no data)", fontsize=9)
                continue
            im_main = ax.imshow(H.T, origin="lower",
                                 extent=(edges[0], edges[-1],
                                         edges[0], edges[-1]),
                                 cmap="mako", aspect="equal",
                                 interpolation="nearest")
            ax.plot([0, 180], [0, 180], color="white", lw=0.8, ls="--",
                    alpha=0.5, zorder=1)
            ax.set_xticks(pos); ax.set_xticklabels(lab)
            ax.set_yticks(pos); ax.set_yticklabels(lab)
            ax.set_title(f"{smooth_label} · {cond_label}  "
                          f"(n_subj={n_subj}, n_trials={n_trials})",
                          fontsize=9, color=cond_color)
            hists[cond_label] = H

        # Difference panel (cdf − inverse_cdf)
        ax = axes[row, -1]
        if "cdf" in hists and "inverse_cdf" in hists:
            diff = hists["cdf"] - hists["inverse_cdf"]
            absmax = float(np.max(np.abs(diff))) or 1e-6
            im_diff = ax.imshow(diff.T, origin="lower",
                                 extent=(edges[0], edges[-1],
                                         edges[0], edges[-1]),
                                 cmap="RdBu_r", aspect="equal",
                                 interpolation="nearest",
                                 vmin=-absmax, vmax=absmax)
            ax.plot([0, 180], [0, 180], color="0.4", lw=0.8, ls="--",
                    alpha=0.6, zorder=1)
            ax.set_xticks(pos); ax.set_xticklabels(lab)
            ax.set_yticks(pos); ax.set_yticklabels(lab)
            ax.set_title(f"{smooth_label} · cdf − inverse_cdf", fontsize=9)
        else:
            ax.set_axis_off()

    for ax in axes[-1, :]:
        ax.set_xlabel("True orientation (°)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Decoded orientation (°)")

    # Two colorbars: one for the matched-condition panels (mako, % trials),
    # one for the difference panels (RdBu_r, %-point delta).
    if im_main is not None:
        cb1 = fig.colorbar(im_main, ax=axes[:, :-1], shrink=0.55, pad=0.02)
        cb1.set_label("Trials (%)", fontsize=8)
        cb1.ax.tick_params(labelsize=7)
    if im_diff is not None:
        cb2 = fig.colorbar(im_diff, ax=axes[:, -1:], shrink=0.55, pad=0.02)
        cb2.set_label("Δ trials (% pts)", fontsize=8)
        cb2.ax.tick_params(labelsize=7)
    return fig


def plot_map_vs_mean_page(group: dict, subject_order_per_variant: dict,
                          lambd: float):
    """Single page comparing MAP vs posterior-mean point estimators.

    Left:  pooled error histograms (4 conditions overlaid; linestyle =
           estimator, hue = smoothing).
    Right: per-subject MAE swarmplot — for each (subject, smoothing) we
           plot MAP and Mean as two dots connected by a thin line, so the
           direction of the within-subject MAP→Mean change is visible.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), constrained_layout=True)
    fig.suptitle(f"MAP vs posterior-mean point estimator  (λ={lambd})",
                 fontsize=10, y=1.02)

    # ── Panel A: pooled error histograms ────────────────────────────────────
    ax = axes[0]
    bins = np.linspace(-90, 90, 19)
    for suffix, smooth_label, color in SMOOTH_VARIANTS:
        items = group.get(suffix, [])
        if not items:
            continue
        truth = np.concatenate([d["truth"] for d in items])
        maps = np.concatenate(
            [map_estimate(d["grid"], d["posteriors"]) for d in items])
        means = np.concatenate(
            [posterior_circular_mean(d["grid"], d["posteriors"]) for d in items])
        err_map = np.degrees(circular_error(maps, truth))
        err_mean = np.degrees(circular_error(means, truth))
        ax.hist(err_map, bins=bins, histtype="step", color=color, lw=1.4,
                ls="--", label=f"{smooth_label} · MAP  MAE={np.abs(err_map).mean():.1f}°")
        ax.hist(err_mean, bins=bins, histtype="step", color=color, lw=1.6,
                ls="-",  label=f"{smooth_label} · Mean MAE={np.abs(err_mean).mean():.1f}°")
    ax.axvline(0, color="black", lw=0.5, ls=":", zorder=0)
    ax.set_xlabel("Decoding error (decoded − true) (°)")
    ax.set_ylabel("Trial count")
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False, fontsize=7, handlelength=1.6)

    # ── Panel B: per-subject MAE swarm, paired MAP/Mean dots ────────────────
    ax = axes[1]
    rows = []
    for suffix, smooth_label, color in SMOOTH_VARIANTS:
        subs = subject_order_per_variant.get(suffix, [])
        items = group.get(suffix, [])
        for sub, d in zip(subs, items):
            err_map = np.degrees(circular_error(
                map_estimate(d["grid"], d["posteriors"]), d["truth"]))
            err_mean = np.degrees(circular_error(
                posterior_circular_mean(d["grid"], d["posteriors"]), d["truth"]))
            rows.append({"subject": sub, "smoothing": smooth_label,
                         "estimator": "MAP",  "mae": float(np.abs(err_map).mean())})
            rows.append({"subject": sub, "smoothing": smooth_label,
                         "estimator": "Mean", "mae": float(np.abs(err_mean).mean())})
    df = pd.DataFrame(rows)
    if df.empty:
        return fig

    palette = {label: color for _, label, color in SMOOTH_VARIANTS}
    smooth_labels_present = [lbl for _, lbl, _ in SMOOTH_VARIANTS]
    # x positions: 2 ticks per smoothing variant (MAP, Mean), offset by 2×idx
    x_map = {(s_lbl, est): 2 * i + j
             for i, s_lbl in enumerate(smooth_labels_present)
             for j, est in enumerate(["MAP", "Mean"])}
    rng = np.random.default_rng(0)
    df["x_jitter"] = [x_map[(s, e)] + rng.uniform(-0.08, 0.08)
                       for s, e in zip(df["smoothing"], df["estimator"])]

    # Within-subject pairing lines (MAP → Mean) inside each smoothing group
    for (sub, smoothing), grp in df.groupby(["subject", "smoothing"]):
        if len(grp) < 2:
            continue
        grp = grp.set_index("estimator")
        if "MAP" not in grp.index or "Mean" not in grp.index:
            continue
        ax.plot([grp.loc["MAP", "x_jitter"], grp.loc["Mean", "x_jitter"]],
                [grp.loc["MAP", "mae"],     grp.loc["Mean", "mae"]],
                color="0.75", lw=0.5, zorder=1)

    for _, row in df.iterrows():
        ax.plot(row["x_jitter"], row["mae"], "o",
                color=palette[row["smoothing"]], markersize=4.5, zorder=2)

    ax.set_xticks(list(x_map.values()))
    ax.set_xticklabels([est for _, est in x_map.keys()])
    # Smoothing-group labels above tick row (one per variant)
    for i, s_lbl in enumerate(smooth_labels_present):
        ax.text(2 * i + 0.5, 1.01, s_lbl,
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=palette[s_lbl])
    ax.set_ylabel("Per-subject MAE (°)")
    ax.set_ylim(bottom=0)

    sns.despine(fig=fig, offset=5, trim=True)
    return fig


def plot_circular_corr_swarm_page(group: dict, subject_order: list[str], lambd: float):
    """Per-subject circular correlation (within-run avg → cross-run mean).

    One dot per (subject, smoothing). Subjects connected by thin grey lines
    showing the smoothing effect within-subject.
    """
    rows = []
    for suffix, label, _ in SMOOTH_VARIANTS:
        for sub, d in zip(subject_order, group.get(suffix, [])):
            rows.append({"subject": sub, "smoothing": label,
                         "circ_corr": per_subject_circular_corr(d)})
    df = pd.DataFrame(rows)
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(4.2, 3.5), constrained_layout=True)
    fig.suptitle(f"V1 circular correlation, within-subject  (λ={lambd})",
                 fontsize=10, y=1.02)

    # Within-subject connecting lines
    if "Unsmoothed" in df["smoothing"].unique() and "Smoothed" in df["smoothing"].unique():
        wide = df.pivot(index="subject", columns="smoothing", values="circ_corr")
        for sub, row in wide.iterrows():
            if not pd.isna(row.get("Unsmoothed")) and not pd.isna(row.get("Smoothed")):
                ax.plot([0, 1], [row["Unsmoothed"], row["Smoothed"]],
                        color="0.75", lw=0.6, zorder=1)

    palette = {label: color for _, label, color in SMOOTH_VARIANTS}
    sns.swarmplot(data=df, x="smoothing", y="circ_corr", hue="smoothing",
                  order=[label for _, label, _ in SMOOTH_VARIANTS],
                  palette=palette, size=5, ax=ax, legend=False, zorder=2)

    ax.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
    ax.set_ylabel("Circular correlation, ρ")
    ax.set_xlabel("")  # tick labels (Unsmoothed/Smoothed) carry the info
    ax.set_ylim(bottom=min(0, df["circ_corr"].min() * 1.05))
    sns.despine(fig=fig, offset=5, trim=True)
    return fig


def discover_subjects(lambd: float) -> list[str]:
    lam_tag = f"_lambda-{lambd}" if lambd != 0.0 else ""
    out = []
    for p in sorted(DERIV.glob("sub-*")):
        fn = (p / "func"
              / f"{p.name}_mask-BensonV1_nvoxels-0_noise-full{lam_tag}_pars.tsv")
        if fn.exists():
            out.append(p.name.removeprefix("sub-"))
    return out


def run(subjects, lambd: float, out: Path):
    if subjects is None:
        subjects = discover_subjects(lambd)
    if not subjects:
        raise SystemExit(f"No V1 decode_gabor files found under {DERIV}.")

    out.parent.mkdir(parents=True, exist_ok=True)
    group: dict[str, list] = {s: [] for s, _, _ in SMOOTH_VARIANTS}
    subject_order_per_variant: dict[str, list[str]] = {s: [] for s, _, _ in SMOOTH_VARIANTS}
    pages = 0

    with PdfPages(out) as pdf:
        for sub in subjects:
            datasets = []
            for suffix, label, color in SMOOTH_VARIANTS:
                loaded = load_posteriors(sub, suffix, lambd=lambd)
                if loaded is None:
                    print(f"sub-{sub} {label}: not found — skipping")
                    continue
                datasets.append((label, color, loaded))
                group[suffix].append(loaded)
                subject_order_per_variant[suffix].append(sub)
            if not datasets:
                continue
            print(f"sub-{sub}: {[lbl for lbl, _, _ in datasets]}")
            fig = plot_subject_page(sub, datasets, lambd)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages += 1

        if any(group.values()):
            page_funcs = [
                plot_group_summary_page,
                plot_decoded_vs_truth_page,
                plot_decoded_vs_truth_by_condition_page,
                lambda g, l: plot_map_vs_mean_page(g, subject_order_per_variant, l),
            ]
            for fn in page_funcs:
                fig = fn(group, lambd)
                if fig is None:
                    continue
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                pages += 1

            # Swarmplot. Use the intersection of subjects across whatever
            # smoothing variants are enabled (only "" if --include-smoothed
            # is off; "" + "_smoothed" if on).
            variant_subjects = [subject_order_per_variant[s]
                                for s, _, _ in SMOOTH_VARIANTS]
            both_order = [s for s in variant_subjects[0]
                          if all(s in vs for vs in variant_subjects[1:])]
            if both_order:
                fig = plot_circular_corr_swarm_page(
                    {s: [d for sub, d in zip(subject_order_per_variant[s], group[s])
                         if sub in both_order]
                     for s in group},
                    both_order, lambd)
                if fig is not None:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    pages += 1

    print(f"Wrote {out}  ({pages} pages)")


def main():
    global SMOOTH_VARIANTS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+",
                   help="Subject labels (default: discover all with V1 decode_gabor on disk)")
    p.add_argument("--lambd", type=float, default=0.1,
                   help="Decoder regularisation λ (default 0.1)")
    p.add_argument("--include-smoothed", action="store_true",
                   help="Also include the smoothed BOLD variant in all plots "
                        "(default: unsmoothed only — smoothing has not moved "
                        "these analyses meaningfully).")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"Output PDF (default {DEFAULT_OUT})")
    args = p.parse_args()
    SMOOTH_VARIANTS = (SMOOTH_VARIANTS_ALL
                        if args.include_smoothed else SMOOTH_VARIANTS_DEFAULT)
    run(args.subjects, args.lambd, Path(args.out))


if __name__ == "__main__":
    main()
