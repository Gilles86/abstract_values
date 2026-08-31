"""NPC abstract-value decoding QA — per-subject + group summaries.

Counterpart to ``check_v1_orientation_decoding.py``. Where V1 decodes the
gabor's physical orientation (π-periodic), NPC (Numerosity Parietal Cortex)
decodes the abstract value (CHF, positive linear axis) the orientation has
been mapped to via the BDM auction. So: same plot family, but linear
(not circular) value axis throughout.

Decode posteriors are read from:
    derivatives/decoding/value/sub-XX/func/sub-XX_mask-NPCr_nvoxels-N_noise-full[_smoothed][_lambda-X]_pars.tsv

Pages produced:

1. Per-subject — two panels:
   - Scatter / 2D-density of decoded vs true CHF with identity line and
     Pearson r.
   - Distribution of decoding errors (decoded − true).

2. Group summary (pooled trials).

3. Decoded-vs-true 2D histogram per smoothing variant (pooled).

4. Condition-split heatmap (cdf / inverse_cdf / cdf − inverse_cdf), per
   smoothing.

5. MAP vs posterior-mean comparison page.

6. Per-subject Pearson r swarmplot (within-run avg → cross-run mean).

All decoded values are out-of-sample (leave-one-run-out CV at fit time).
Default is unsmoothed-only; pass ``--include-smoothed`` to add smoothed.

Usage:
    python -m abstract_values.visualize.check_npc_value_decoding
    python -m abstract_values.visualize.check_npc_value_decoding --subjects 08 09
    python -m abstract_values.visualize.check_npc_value_decoding --nvoxels 100
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

# ── Vision-science house style ──────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "mathtext.fontset": "stixsans",
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 4,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False, "legend.handlelength": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 150, "savefig.dpi": 300,
})
sns.set_context("paper")

DERIV = Path(BIDS_FOLDER) / "derivatives" / "decoding" / "value"
GABOR_DERIV = Path(BIDS_FOLDER) / "derivatives" / "decoding" / "gabor"
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "npc_value_decoding.pdf"

SMOOTH_UNSMOOTHED = ("",          "Unsmoothed", "#3B5BA5")
SMOOTH_SMOOTHED   = ("_smoothed", "Smoothed",   "#C44E52")
SMOOTH_VARIANTS_DEFAULT = [SMOOTH_UNSMOOTHED]
SMOOTH_VARIANTS_ALL     = [SMOOTH_UNSMOOTHED, SMOOTH_SMOOTHED]
SMOOTH_VARIANTS = SMOOTH_VARIANTS_DEFAULT  # reassigned in main()

# Match the Bedi et al. 2026 behavior-notebook palette so decoding figures
# read the same as the behavioral panels.
CONDITIONS = [
    ("cdf",          "#E76F51"),   # warm coral
    ("inverse_cdf",  "#2A9D8F"),   # dark teal
]


def session_to_condition(subject_id: str, session: int) -> str:
    num = int("".join(c for c in subject_id if c.isdigit()))
    if num % 2 == 0:
        return "cdf" if session == 1 else "inverse_cdf"
    return "inverse_cdf" if session == 1 else "cdf"


def _has_decoded_trials(fn: Path) -> bool:
    """True when a decode _pars.tsv holds at least one trial.

    Since the zero-voxel fix, a cell where no fold was decodable still writes
    a header-only file so Snakemake gets its sentinel. Such a file must not be
    mistaken for a usable result.
    """
    try:
        with open(fn) as f:
            f.readline()          # header
            return bool(f.readline().strip())
    except OSError:
        return False


def load_posteriors(subject: str, smoothed_suffix: str, mask: str = "NPCr",
                    n_voxels: int = 100, lambd: float = 0.1):
    """Return dict with truth/grid/posteriors/run_keys/conditions."""
    lam_tag = f"_lambda-{lambd}" if lambd != 0.0 else ""
    fn = (DERIV / f"sub-{subject}" / "func"
          / f"sub-{subject}_mask-{mask}_nvoxels-{n_voxels}"
            f"_noise-full{smoothed_suffix}{lam_tag}_pars.tsv")
    if not fn.exists():
        return None
    df = pd.read_csv(fn, sep="\t", index_col=[0, 1, 2])
    if df.empty:
        # No fold was decodable for this cell — treat it as absent.
        return None
    truth = df["true_value_chf"].to_numpy(dtype=np.float32)
    grid = np.asarray(df.columns[1:], dtype=np.float32)
    posteriors = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)
    sess = df.index.get_level_values("session").to_numpy(dtype=np.int32)
    run = df.index.get_level_values("run").to_numpy(dtype=np.int32)
    return {
        "truth": truth, "grid": grid, "posteriors": posteriors,
        "run_keys": sess * 100 + run,
        "conditions": np.array([session_to_condition(subject, int(s)) for s in sess]),
    }


def map_estimate(grid: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
    return grid[posteriors.argmax(axis=1)]


def posterior_mean(grid: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
    """Linear (not circular) probability-weighted mean."""
    return (posteriors * grid[None, :]).sum(axis=1).astype(np.float32)


def decoding_error(decoded: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return decoded - truth


def per_subject_pearson(d: dict) -> float:
    """Mean per-run Pearson r across runs of one subject (within-run avg)."""
    truth = d["truth"]
    decoded = posterior_mean(d["grid"], d["posteriors"])
    keys = d["run_keys"]
    rs = []
    for k in np.unique(keys):
        sel = (keys == k)
        if sel.sum() < 3:
            continue
        t, p = truth[sel], decoded[sel]
        if t.std() == 0 or p.std() == 0:
            continue
        rs.append(np.corrcoef(t, p)[0, 1])
    return float(np.nanmean(rs)) if rs else float("nan")


# ── plotting helpers ─────────────────────────────────────────────────────────

def _value_range(group):
    """Single shared (vmin, vmax) for axes/bins, drawn from the grid."""
    grids = [d["grid"] for items in group.values() for d in items]
    if not grids:
        return 2.0, 42.0
    lo = float(min(g.min() for g in grids))
    hi = float(max(g.max() for g in grids))
    return lo, hi


def _hist_bins(vmin, vmax, n=20):
    return np.linspace(vmin, vmax, n + 1)


def _pearson_per_condition(truth, decoded, cond):
    """Return {cond_label: r}. NaN if cond doesn't have ≥2 points or no variance."""
    out = {}
    for cond_label, _ in CONDITIONS:
        sel = cond == cond_label
        if sel.sum() < 2:
            continue
        t, p = truth[sel], decoded[sel]
        if t.std() == 0 or p.std() == 0:
            continue
        out[cond_label] = float(np.corrcoef(t, p)[0, 1])
    return out


def plot_subject_page(subject: str, datasets, lambd: float, nvoxels: int,
                      vmin: float, vmax: float):
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.9), constrained_layout=True)
    fig.suptitle(f"sub-{subject}  ·  NPCr value decoding  (n_voxels={nvoxels}, λ={lambd})",
                 fontsize=10, y=1.02)

    cond_palette = {c: col for c, col in CONDITIONS}

    # Panel A: scatter colored by condition; one r per condition
    ax = axes[0]
    for label, color, d in datasets:
        truth = d["truth"]
        decoded = posterior_mean(d["grid"], d["posteriors"])
        cond = d["conditions"]
        rs = _pearson_per_condition(truth, decoded, cond)
        r_text = "  ".join(f"r({c[:3]})={rs[c]:.2f}"
                            for c, _ in CONDITIONS if c in rs)
        for cond_label, cond_color in CONDITIONS:
            sel = cond == cond_label
            if not sel.any():
                continue
            ax.plot(truth[sel], decoded[sel], "o", color=cond_color,
                    ms=2.8, alpha=0.45, mec="none")
        # smoothing label + 2 r's in legend slot
        ax.plot([], [], " ", label=f"{label} · {r_text}")
    ax.plot([vmin, vmax], [vmin, vmax], color="0.5", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("True value (CHF)")
    ax.set_ylabel("Decoded value (CHF)")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal")
    # condition swatch legend
    handles = [plt.Line2D([], [], marker="o", color=col, lw=0,
                            mec="none", ms=5, label=c)
               for c, col in CONDITIONS]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0],
              loc="upper left", frameon=False, fontsize=7)

    # Panel B: error histogram (per smoothing — condition is in the scatter)
    ax = axes[1]
    bins = np.linspace(-(vmax - vmin), vmax - vmin, 31)
    for label, color, d in datasets:
        err = decoding_error(posterior_mean(d["grid"], d["posteriors"]), d["truth"])
        mae = float(np.abs(err).mean())
        ax.hist(err, bins=bins, histtype="step", color=color, lw=1.6,
                label=f"{label} · MAE={mae:.2f} CHF")
    ax.axvline(0, color="black", lw=0.5, ls=":")
    ax.set_xlabel("Decoding error (decoded − true) (CHF)")
    ax.set_ylabel("Trial count")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    sns.despine(fig=fig, offset=5, trim=True)
    return fig


def plot_group_summary_page(group, lambd, nvoxels, vmin, vmax):
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.9), constrained_layout=True)
    fig.suptitle(f"Group  ·  NPCr value decoding (pooled trials, n_voxels={nvoxels}, λ={lambd})",
                 fontsize=10, y=1.02)

    cond_palette = {c: col for c, col in CONDITIONS}

    # Panel A: scatter colored by condition; two r's
    ax = axes[0]
    text_lines = []
    for suffix, label, color in SMOOTH_VARIANTS:
        items = group.get(suffix, [])
        if not items:
            continue
        truth = np.concatenate([d["truth"] for d in items])
        decoded = np.concatenate(
            [posterior_mean(d["grid"], d["posteriors"]) for d in items])
        cond = np.concatenate([d["conditions"] for d in items])
        for cond_label, cond_color in CONDITIONS:
            sel = cond == cond_label
            if not sel.any():
                continue
            ax.plot(truth[sel], decoded[sel], "o", color=cond_color,
                    ms=1.8, alpha=0.18, mec="none")
        rs = _pearson_per_condition(truth, decoded, cond)
        r_text = "  ".join(f"r({c[:3]})={rs[c]:.2f}"
                            for c, _ in CONDITIONS if c in rs)
        text_lines.append(f"{label}  (n_subj={len(items)})  {r_text}")
    ax.plot([vmin, vmax], [vmin, vmax], color="0.5", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("True value (CHF)")
    ax.set_ylabel("Decoded value (CHF)")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal")
    handles = [plt.Line2D([], [], marker="o", color=col, lw=0,
                            mec="none", ms=5, label=c)
               for c, col in CONDITIONS]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7)
    # Two-r summary as in-panel text (one line per smoothing)
    for i, line in enumerate(text_lines):
        ax.text(0.98, 0.04 + 0.06 * i, line, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7)

    ax = axes[1]
    bins = np.linspace(-(vmax - vmin), vmax - vmin, 31)
    for suffix, label, color in SMOOTH_VARIANTS:
        items = group.get(suffix, [])
        if not items:
            continue
        truth = np.concatenate([d["truth"] for d in items])
        decoded = np.concatenate([posterior_mean(d["grid"], d["posteriors"]) for d in items])
        err = decoding_error(decoded, truth)
        mae = float(np.abs(err).mean())
        ax.hist(err, bins=bins, histtype="step", color=color, lw=1.6,
                label=f"{label} · MAE={mae:.2f} CHF")
    ax.axvline(0, color="black", lw=0.5, ls=":")
    ax.set_xlabel("Decoding error (decoded − true) (CHF)")
    ax.set_ylabel("Trial count")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    sns.despine(fig=fig, offset=5, trim=True)
    return fig


def _value_hist(truth_pool, decoded_pool, edges):
    if not truth_pool:
        return None
    truth = np.concatenate(truth_pool)
    decoded = np.concatenate(decoded_pool)
    H, _, _ = np.histogram2d(truth, decoded, bins=edges)
    return H / H.sum() * 100 if H.sum() > 0 else H


def _plot_one_value_heatmap(ax, H, edges, title, title_color, cmap="mako",
                            vmin=None, vmax=None):
    if H is None:
        ax.set_axis_off()
        ax.set_title(f"{title}  (no data)", fontsize=9)
        return None
    im = ax.imshow(H.T, origin="lower",
                    extent=(edges[0], edges[-1], edges[0], edges[-1]),
                    cmap=cmap, aspect="equal", interpolation="nearest",
                    vmin=vmin, vmax=vmax)
    ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]],
            color="white" if cmap != "RdBu_r" else "0.4",
            lw=0.8, ls="--", alpha=0.6, zorder=1)
    ax.set_title(title, fontsize=9, color=title_color)
    return im


def plot_decoded_vs_truth_page(group, lambd, nvoxels, vmin, vmax):
    edges = _hist_bins(vmin, vmax, n=20)
    fig, axes = plt.subplots(1, len(SMOOTH_VARIANTS),
                              figsize=(3.4 * len(SMOOTH_VARIANTS), 3.2),
                              constrained_layout=True,
                              sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    fig.suptitle(f"Group  ·  Decoded vs true value (n_voxels={nvoxels}, λ={lambd})",
                 fontsize=10, y=1.02)
    im = None
    for ax, (suffix, label, color) in zip(axes, SMOOTH_VARIANTS):
        items = group.get(suffix, [])
        n_trials = sum(d["truth"].shape[0] for d in items)
        H = _value_hist(
            [d["truth"] for d in items],
            [posterior_mean(d["grid"], d["posteriors"]) for d in items],
            edges,
        )
        title = f"{label}  (n_subj={len(items)}, n_trials={n_trials})"
        new_im = _plot_one_value_heatmap(ax, H, edges, title, color)
        if new_im is not None:
            im = new_im
        ax.set_xlabel("True value (CHF)")
    axes[0].set_ylabel("Decoded value (CHF)")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
        cbar.set_label("Trials (%)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    return fig


def _trials_for_condition(items, condition_label):
    truth_pool, decoded_pool, n_subj, n_trials = [], [], 0, 0
    for d in items:
        sel = d["conditions"] == condition_label
        if not sel.any():
            continue
        n_subj += 1
        truth_pool.append(d["truth"][sel])
        decoded_pool.append(posterior_mean(d["grid"], d["posteriors"][sel]))
        n_trials += int(sel.sum())
    return truth_pool, decoded_pool, n_subj, n_trials


def plot_decoded_vs_truth_by_condition_page(group, lambd, nvoxels, vmin, vmax):
    edges = _hist_bins(vmin, vmax, n=20)
    n_cols = len(CONDITIONS) + 1   # + diff panel
    fig, axes = plt.subplots(len(SMOOTH_VARIANTS), n_cols,
                              figsize=(3.0 * n_cols, 3.4 * len(SMOOTH_VARIANTS)),
                              constrained_layout=True,
                              sharex=True, sharey=True, squeeze=False)
    fig.suptitle(f"Group  ·  Decoded vs true value, split by condition (n_voxels={nvoxels}, λ={lambd})",
                 fontsize=10, y=1.01)

    im_main = None
    im_diff = None
    for row, (suffix, smooth_label, smooth_color) in enumerate(SMOOTH_VARIANTS):
        items = group.get(suffix, [])
        hists = {}
        for col, (cond_label, cond_color) in enumerate(CONDITIONS):
            ax = axes[row, col]
            truth_pool, decoded_pool, n_subj, n_trials = _trials_for_condition(
                items, cond_label)
            H = _value_hist(truth_pool, decoded_pool, edges)
            title = (f"{smooth_label} · {cond_label}  "
                     f"(n_subj={n_subj}, n_trials={n_trials})")
            new_im = _plot_one_value_heatmap(ax, H, edges, title, cond_color)
            if new_im is not None:
                im_main = new_im
                hists[cond_label] = H

        ax = axes[row, -1]
        if "cdf" in hists and "inverse_cdf" in hists:
            diff = hists["cdf"] - hists["inverse_cdf"]
            absmax = float(np.max(np.abs(diff))) or 1e-6
            im_diff = _plot_one_value_heatmap(
                ax, diff, edges,
                f"{smooth_label} · cdf − inverse_cdf",
                title_color="0.2",
                cmap="RdBu_r", vmin=-absmax, vmax=absmax,
            )
        else:
            ax.set_axis_off()

    for ax in axes[-1, :]:
        ax.set_xlabel("True value (CHF)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Decoded value (CHF)")

    if im_main is not None:
        cb1 = fig.colorbar(im_main, ax=axes[:, :-1], shrink=0.55, pad=0.02)
        cb1.set_label("Trials (%)", fontsize=8)
        cb1.ax.tick_params(labelsize=7)
    if im_diff is not None:
        cb2 = fig.colorbar(im_diff, ax=axes[:, -1:], shrink=0.55, pad=0.02)
        cb2.set_label("Δ trials (% pts)", fontsize=8)
        cb2.ax.tick_params(labelsize=7)
    return fig


def plot_map_vs_mean_page(group, subject_order_per_variant, lambd, nvoxels,
                          vmin, vmax):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), constrained_layout=True)
    fig.suptitle(f"MAP vs posterior-mean point estimator  (n_voxels={nvoxels}, λ={lambd})",
                 fontsize=10, y=1.02)

    ax = axes[0]
    bins = np.linspace(-(vmax - vmin), vmax - vmin, 31)
    for suffix, smooth_label, color in SMOOTH_VARIANTS:
        items = group.get(suffix, [])
        if not items:
            continue
        truth = np.concatenate([d["truth"] for d in items])
        maps = np.concatenate([map_estimate(d["grid"], d["posteriors"]) for d in items])
        means = np.concatenate([posterior_mean(d["grid"], d["posteriors"]) for d in items])
        err_map = decoding_error(maps, truth)
        err_mean = decoding_error(means, truth)
        ax.hist(err_map, bins=bins, histtype="step", color=color, lw=1.4, ls="--",
                label=f"{smooth_label} · MAP  MAE={np.abs(err_map).mean():.2f}")
        ax.hist(err_mean, bins=bins, histtype="step", color=color, lw=1.6, ls="-",
                label=f"{smooth_label} · Mean MAE={np.abs(err_mean).mean():.2f}")
    ax.axvline(0, color="black", lw=0.5, ls=":")
    ax.set_xlabel("Decoding error (CHF)")
    ax.set_ylabel("Trial count")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False, fontsize=7)

    ax = axes[1]
    rows = []
    for suffix, smooth_label, color in SMOOTH_VARIANTS:
        subs = subject_order_per_variant.get(suffix, [])
        items = group.get(suffix, [])
        for sub, d in zip(subs, items):
            err_map = decoding_error(map_estimate(d["grid"], d["posteriors"]), d["truth"])
            err_mean = decoding_error(posterior_mean(d["grid"], d["posteriors"]), d["truth"])
            rows.append({"subject": sub, "smoothing": smooth_label,
                         "estimator": "MAP", "mae": float(np.abs(err_map).mean())})
            rows.append({"subject": sub, "smoothing": smooth_label,
                         "estimator": "Mean", "mae": float(np.abs(err_mean).mean())})
    df = pd.DataFrame(rows)
    if df.empty:
        return fig

    palette = {label: color for _, label, color in SMOOTH_VARIANTS}
    smooth_labels = [lbl for _, lbl, _ in SMOOTH_VARIANTS]
    x_map = {(s, est): 2 * i + j
             for i, s in enumerate(smooth_labels)
             for j, est in enumerate(["MAP", "Mean"])}
    rng = np.random.default_rng(0)
    df["x_jitter"] = [x_map[(s, e)] + rng.uniform(-0.08, 0.08)
                       for s, e in zip(df["smoothing"], df["estimator"])]
    for (sub, smoothing), grp in df.groupby(["subject", "smoothing"]):
        g = grp.set_index("estimator")
        if "MAP" in g.index and "Mean" in g.index:
            ax.plot([g.loc["MAP", "x_jitter"], g.loc["Mean", "x_jitter"]],
                    [g.loc["MAP", "mae"], g.loc["Mean", "mae"]],
                    color="0.75", lw=0.5, zorder=1)
    for _, row in df.iterrows():
        ax.plot(row["x_jitter"], row["mae"], "o",
                color=palette[row["smoothing"]], ms=4.5, zorder=2)

    ax.set_xticks(list(x_map.values()))
    ax.set_xticklabels([est for _, est in x_map.keys()])
    for i, s_lbl in enumerate(smooth_labels):
        ax.text(2 * i + 0.5, 1.01, s_lbl,
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=palette[s_lbl])
    ax.set_ylabel("Per-subject MAE (CHF)")
    ax.set_ylim(bottom=0)
    sns.despine(fig=fig, offset=5, trim=True)
    return fig


def plot_pearson_swarm_page(group, subject_order_per_variant, lambd, nvoxels):
    """Per-subject Pearson r (within-run avg → mean across runs) swarm."""
    rows = []
    palette = {label: color for _, label, color in SMOOTH_VARIANTS}
    for suffix, label, _ in SMOOTH_VARIANTS:
        for sub, d in zip(subject_order_per_variant.get(suffix, []),
                           group.get(suffix, [])):
            rows.append({"subject": sub, "smoothing": label,
                         "pearson_r": per_subject_pearson(d)})
    df = pd.DataFrame(rows)
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(4.2, 3.5), constrained_layout=True)
    fig.suptitle(f"NPCr Pearson r, within-subject  (n_voxels={nvoxels}, λ={lambd})",
                 fontsize=10, y=1.02)

    smooth_labels = [label for _, label, _ in SMOOTH_VARIANTS]
    if len(smooth_labels) >= 2:
        wide = df.pivot(index="subject", columns="smoothing", values="pearson_r")
        if all(s in wide.columns for s in smooth_labels[:2]):
            for sub, row in wide.iterrows():
                a, b = row.get(smooth_labels[0]), row.get(smooth_labels[1])
                if pd.notna(a) and pd.notna(b):
                    ax.plot([0, 1], [a, b], color="0.75", lw=0.6, zorder=1)

    sns.swarmplot(data=df, x="smoothing", y="pearson_r", hue="smoothing",
                  order=smooth_labels, palette=palette, size=5, ax=ax,
                  legend=False, zorder=2)
    ax.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
    ax.set_ylabel("Pearson r")
    ax.set_xlabel("")
    ymin = min(0, df["pearson_r"].min() * 1.05)
    ax.set_ylim(bottom=ymin)
    sns.despine(fig=fig, offset=5, trim=True)
    return fig


def load_orientation_aligned(subject: str, smoothed_suffix: str,
                             lambd: float, gabor_mask: str = "BensonV1",
                             gabor_nvoxels: int = 0) -> dict | None:
    """Load orientation per trial from the *gabor* decoder TSV for `subject`.

    decode_value's TSV doesn't carry orientation directly, but the
    decode_gabor TSV has true_orientation_rad on the same (session, run,
    trial_nr) index. Reading both and merging on that index gives us
    orientation + value-decoding error per trial.
    """
    lam_tag = f"_lambda-{lambd}" if lambd != 0.0 else ""
    fn = (GABOR_DERIV / f"sub-{subject}" / "func"
          / f"sub-{subject}_mask-{gabor_mask}_nvoxels-{gabor_nvoxels}"
            f"_noise-full{smoothed_suffix}{lam_tag}_pars.tsv")
    if not fn.exists():
        return None
    df = pd.read_csv(fn, sep="\t", index_col=[0, 1, 2])
    return {"index": df.index, "orientation_rad": df["true_orientation_rad"].to_numpy(np.float32)}


def _value_error_by_orientation(subject: str, d_value: dict,
                                 smoothed_suffix: str, lambd: float,
                                 n_orient_bins: int = 12,
                                 residual: bool = False):
    """Per-orientation-bin (n_bins,) arrays of mean+SD value-decoding error.

    If ``residual`` is False (default), the error is ``decoded − true``.
    If ``residual`` is True, it's ``decoded − subject_mean_value`` — i.e. the
    error minus what a constant "always predict the mean" decoder would
    produce. This removes the systematic per-orientation bias that's purely
    an artefact of the orientation→CHF mapping (low-orient maps to low CHF
    in cdf, etc.) and isolates the actual decoder's contribution. For a
    useless decoder this collapses to 0; for a working decoder the cdf and
    inverse_cdf curves should be approximate mirror images around 0.

    Returns: dict mapping condition → {orient_centers, mean_err, sd_err, n}.
    """
    orient = load_orientation_aligned(subject, smoothed_suffix, lambd)
    if orient is None:
        return None
    # Verify alignment: both should share (session, run, trial_nr) index
    val_df = pd.read_csv(
        DERIV / f"sub-{subject}" / "func"
        / f"sub-{subject}_mask-NPCr_nvoxels-100_noise-full{smoothed_suffix}"
          f"{('_lambda-' + str(lambd)) if lambd != 0.0 else ''}_pars.tsv",
        sep="\t", index_col=[0, 1, 2],
    )
    # Trim to shared trials in index order — use the value index to align.
    orient_df = pd.DataFrame({"orient_rad": orient["orientation_rad"]},
                              index=orient["index"])
    orient_aligned = orient_df.reindex(val_df.index)
    if orient_aligned["orient_rad"].isna().any():
        # Index mismatch — skip
        return None
    orient_rad = orient_aligned["orient_rad"].to_numpy(np.float32)

    # Compute error using posterior mean for the value decoder
    decoded = posterior_mean(d_value["grid"], d_value["posteriors"])
    if residual:
        # Subtract the constant-mean baseline (subject-specific mean across
        # ALL trials). Equivalent to (decoded - true) - (mean - true) =
        # decoded - mean. Removes the per-orientation bias a constant
        # predictor would already produce.
        baseline = float(d_value["truth"].mean())
        error = decoded - baseline
    else:
        error = decoded - d_value["truth"]

    # Per-trial condition
    cond = d_value["conditions"]

    # Bin by orientation (linearly across [0, π))
    edges = np.linspace(0, np.pi, n_orient_bins + 1)
    centers_deg = np.degrees(0.5 * (edges[:-1] + edges[1:]))
    bin_idx = np.clip(np.digitize(orient_rad, edges) - 1, 0, n_orient_bins - 1)

    out = {}
    for cond_label, _ in CONDITIONS:
        sel = cond == cond_label
        if not sel.any():
            continue
        mean_err = np.full(n_orient_bins, np.nan)
        sd_err = np.full(n_orient_bins, np.nan)
        n_per_bin = np.zeros(n_orient_bins, dtype=int)
        for b in range(n_orient_bins):
            mask = sel & (bin_idx == b)
            if mask.sum() >= 2:
                mean_err[b] = float(error[mask].mean())
                sd_err[b] = float(error[mask].std(ddof=1))
                n_per_bin[b] = int(mask.sum())
            elif mask.sum() == 1:
                mean_err[b] = float(error[mask].mean())
                n_per_bin[b] = 1
        out[cond_label] = {
            "orient_centers": centers_deg,
            "mean_err": mean_err,
            "sd_err": sd_err,
            "n": n_per_bin,
        }
    return out


def _aggregate_across_subjects(per_subject: list[dict], key: str
                                ) -> dict[str, np.ndarray]:
    """Stack per-subject (orient_centers, key) arrays into (n_subj, n_bins).

    Returns {cond_label: {"orient_centers", "mean", "sem"}} where mean/sem
    are computed across subjects (NaN-aware).
    """
    by_cond: dict[str, list[np.ndarray]] = {}
    centers = None
    for sub_data in per_subject:
        if sub_data is None:
            continue
        for cond_label, payload in sub_data.items():
            if centers is None:
                centers = payload["orient_centers"]
            by_cond.setdefault(cond_label, []).append(payload[key])
    out = {}
    for cond_label, stack in by_cond.items():
        arr = np.stack(stack, axis=0)
        mean = np.nanmean(arr, axis=0)
        n_valid = np.sum(~np.isnan(arr), axis=0)
        sd = np.nanstd(arr, axis=0, ddof=1)
        sem = sd / np.sqrt(np.maximum(n_valid, 1))
        out[cond_label] = {"orient_centers": centers, "mean": mean,
                            "sem": sem, "n_subj": arr.shape[0]}
    return out


def plot_value_error_vs_orientation_page(group, subject_order_per_variant,
                                          lambd, nvoxels):
    """Mean value-decoding error per orientation bin, split by condition.

    One panel per smoothing variant. Per-subject mean error per orientation
    bin is averaged across subjects (±1 SEM across subjects).
    """
    fig, axes = plt.subplots(1, len(SMOOTH_VARIANTS),
                              figsize=(4.0 * len(SMOOTH_VARIANTS), 3.0),
                              constrained_layout=True, squeeze=False,
                              sharey=True)
    axes = axes[0]
    fig.suptitle(f"NPCr  ·  Mean value-decoding error vs orientation  (n_voxels={nvoxels}, λ={lambd})",
                 fontsize=10, y=1.02)

    cond_palette = {c: col for c, col in CONDITIONS}
    for ax, (suffix, smooth_label, smooth_color) in zip(axes, SMOOTH_VARIANTS):
        per_sub_payload = []
        for sub, d in zip(subject_order_per_variant.get(suffix, []),
                           group.get(suffix, [])):
            per_sub_payload.append(_value_error_by_orientation(sub, d, suffix, lambd))
        agg = _aggregate_across_subjects(per_sub_payload, key="mean_err")
        for cond_label, _ in CONDITIONS:
            if cond_label not in agg:
                continue
            a = agg[cond_label]
            ax.plot(a["orient_centers"], a["mean"], color=cond_palette[cond_label],
                    lw=1.6, label=f"{cond_label}  (n_subj={a['n_subj']})")
            ax.fill_between(a["orient_centers"], a["mean"] - a["sem"],
                            a["mean"] + a["sem"],
                            color=cond_palette[cond_label], alpha=0.22, linewidth=0)
        ax.axhline(0, color="0.5", lw=0.6, ls="--", zorder=0)
        ax.set_title(smooth_label, fontsize=9, color=smooth_color)
        ax.set_xlabel("Orientation (°)")
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.legend(loc="upper right", frameon=False, fontsize=7)
    axes[0].set_ylabel("Mean decoding error (CHF)")
    sns.despine(fig=fig, offset=5, trim=True)
    return fig


def plot_value_error_sd_vs_orientation_page(group, subject_order_per_variant,
                                             lambd, nvoxels):
    """Variability (SD) of value-decoding error per orientation bin."""
    fig, axes = plt.subplots(1, len(SMOOTH_VARIANTS),
                              figsize=(4.0 * len(SMOOTH_VARIANTS), 3.0),
                              constrained_layout=True, squeeze=False,
                              sharey=True)
    axes = axes[0]
    fig.suptitle(f"NPCr  ·  Variability of value-decoding error vs orientation  (n_voxels={nvoxels}, λ={lambd})",
                 fontsize=10, y=1.02)

    cond_palette = {c: col for c, col in CONDITIONS}
    for ax, (suffix, smooth_label, smooth_color) in zip(axes, SMOOTH_VARIANTS):
        per_sub_payload = []
        for sub, d in zip(subject_order_per_variant.get(suffix, []),
                           group.get(suffix, [])):
            per_sub_payload.append(_value_error_by_orientation(sub, d, suffix, lambd))
        agg = _aggregate_across_subjects(per_sub_payload, key="sd_err")
        for cond_label, _ in CONDITIONS:
            if cond_label not in agg:
                continue
            a = agg[cond_label]
            ax.plot(a["orient_centers"], a["mean"], color=cond_palette[cond_label],
                    lw=1.6, label=f"{cond_label}  (n_subj={a['n_subj']})")
            ax.fill_between(a["orient_centers"], a["mean"] - a["sem"],
                            a["mean"] + a["sem"],
                            color=cond_palette[cond_label], alpha=0.22, linewidth=0)
        ax.set_title(smooth_label, fontsize=9, color=smooth_color)
        ax.set_xlabel("Orientation (°)")
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", frameon=False, fontsize=7)
    axes[0].set_ylabel("Within-subject SD of decoding error (CHF)")
    sns.despine(fig=fig, offset=5, trim=True)
    return fig


def _load_fisher_info(subject: str, nvoxels: int, smoothed: bool = False,
                       roi: str = "NPCr") -> pd.DataFrame | None:
    """Load the all-sessions FI TSV for NPCr aprf decoding."""
    smtag = "_smoothed" if smoothed else ""
    fn = (Path(BIDS_FOLDER) / "derivatives" / "encoding_models" / "aprf"
          / f"sub-{subject}" / "func"
          / f"sub-{subject}_task-abstractvalue_mask-{roi}"
            f"_nvoxels-{nvoxels}{smtag}_desc-fisherinfo_pe.tsv")
    if not fn.exists():
        return None
    df = pd.read_csv(fn, sep="\t")
    df = df.rename(columns={"Unnamed: 0": "value_chf"})
    return df


def plot_fisher_information_page(group, subject_order_per_variant, lambd, nvoxels):
    """Group Fisher-information profile in NPCr.

    FI is computed once per subject (aPRF all-sessions joint fit + the
    population's noise covariance, see compute_fisher_information_aprf.py),
    so there's nothing to split by condition here — only by smoothing.
    """
    # FI files exist independently of decode TSVs — collect subjects from disk.
    fi_dir = Path(BIDS_FOLDER) / "derivatives" / "encoding_models" / "aprf"
    if not fi_dir.exists():
        return None

    fig, ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
    fig.suptitle(f"NPCr  ·  Fisher information vs value (n_voxels={nvoxels})",
                 fontsize=10, y=1.02)

    plotted_any = False
    for suffix, smooth_label, color in SMOOTH_VARIANTS:
        subs = subject_order_per_variant.get(suffix, []) or [
            p.name.removeprefix("sub-") for p in sorted(fi_dir.glob("sub-*"))
        ]
        per_sub = []
        used = []
        for sub in subs:
            df = _load_fisher_info(sub, nvoxels, smoothed=bool(suffix))
            if df is None:
                continue
            per_sub.append(df)
            used.append(sub)
        if not per_sub:
            continue
        # All FI grids are the same; stack values across subjects.
        grid = per_sub[0]["value_chf"].to_numpy(np.float32)
        stack = np.stack([d["fisher_information"].to_numpy(np.float32)
                          for d in per_sub], axis=0)
        mean = np.nanmean(stack, axis=0)
        sem = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])

        # Light per-subject traces in the background
        for tr in stack:
            ax.plot(grid, tr, color=color, lw=0.4, alpha=0.25, zorder=1)
        ax.plot(grid, mean, color=color, lw=1.8,
                label=f"{smooth_label}  (n_subj={len(used)})",
                zorder=3)
        ax.fill_between(grid, mean - sem, mean + sem,
                         color=color, alpha=0.22, linewidth=0, zorder=2)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return None
    ax.set_xlabel("Value (CHF)")
    ax.set_ylabel("Fisher information")
    ax.set_yscale("log")               # FI typically spans 3+ orders of magnitude
    ax.set_ylim(bottom=1e-3)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    sns.despine(fig=fig, offset=5, trim=True)
    return fig


def discover_subjects(lambd: float, n_voxels: int) -> list[str]:
    lam_tag = f"_lambda-{lambd}" if lambd != 0.0 else ""
    out = []
    for p in sorted(DERIV.glob("sub-*")):
        fn = (p / "func"
              / f"{p.name}_mask-NPCr_nvoxels-{n_voxels}_noise-full{lam_tag}_pars.tsv")
        if fn.exists() and _has_decoded_trials(fn):
            out.append(p.name.removeprefix("sub-"))
    return out


def run(subjects, lambd, n_voxels, out: Path):
    if subjects is None:
        subjects = discover_subjects(lambd, n_voxels)
    if not subjects:
        raise SystemExit(f"No NPCr decode_value files for n_voxels={n_voxels} "
                         f"under {DERIV}.")

    out.parent.mkdir(parents=True, exist_ok=True)
    group = {s: [] for s, _, _ in SMOOTH_VARIANTS}
    subject_order_per_variant = {s: [] for s, _, _ in SMOOTH_VARIANTS}
    pages = 0

    # First pass: load everything so we can compute a global value range
    loaded = {}
    for sub in subjects:
        for suffix, label, color in SMOOTH_VARIANTS:
            d = load_posteriors(sub, suffix, n_voxels=n_voxels, lambd=lambd)
            if d is None:
                print(f"sub-{sub} {label}: not found — skipping")
                continue
            loaded[(sub, suffix)] = (label, color, d)
            group[suffix].append(d)
            subject_order_per_variant[suffix].append(sub)
    if not any(group.values()):
        raise SystemExit("No data loaded.")

    vmin, vmax = _value_range(group)

    with PdfPages(out) as pdf:
        for sub in subjects:
            datasets = [loaded[(sub, suffix)]
                        for suffix, *_ in SMOOTH_VARIANTS
                        if (sub, suffix) in loaded]
            if not datasets:
                continue
            print(f"sub-{sub}: {[lbl for lbl, _, _ in datasets]}")
            fig = plot_subject_page(sub, datasets, lambd, n_voxels, vmin, vmax)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages += 1

        for fn in (
            plot_group_summary_page,
            plot_decoded_vs_truth_page,
            plot_decoded_vs_truth_by_condition_page,
            lambda g, l, n, vmn, vmx: plot_value_error_vs_orientation_page(
                g, subject_order_per_variant, l, n),
            lambda g, l, n, vmn, vmx: plot_value_error_sd_vs_orientation_page(
                g, subject_order_per_variant, l, n),
            lambda g, l, n, vmn, vmx: plot_map_vs_mean_page(
                g, subject_order_per_variant, l, n, vmn, vmx),
            lambda g, l, n, vmn, vmx: plot_pearson_swarm_page(
                g, subject_order_per_variant, l, n),
            lambda g, l, n, vmn, vmx: plot_fisher_information_page(
                g, subject_order_per_variant, l, n),
        ):
            fig = fn(group, lambd, n_voxels, vmin, vmax)
            if fig is None:
                continue
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages += 1

    print(f"Wrote {out}  ({pages} pages)")


def main():
    global SMOOTH_VARIANTS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+",
                   help="Subject labels (default: discover all on disk)")
    p.add_argument("--nvoxels", type=int, default=100,
                   help="Decoder n_voxels (default 100; pass 0 for all-voxels)")
    p.add_argument("--lambd", type=float, default=0.1,
                   help="Decoder regularisation λ (default 0.1)")
    p.add_argument("--include-smoothed", action="store_true",
                   help="Also include the smoothed BOLD variant (default: unsmoothed only).")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"Output PDF (default {DEFAULT_OUT})")
    args = p.parse_args()
    SMOOTH_VARIANTS = SMOOTH_VARIANTS_ALL if args.include_smoothed else SMOOTH_VARIANTS_DEFAULT
    run(args.subjects, args.lambd, args.nvoxels, Path(args.out))


if __name__ == "__main__":
    main()
