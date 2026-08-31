"""Does decoding accuracy depend on n_voxels? Sweep + plot.

The snakemake pipeline already decodes both `gabor` (orientation) and
`value` (CHF) for n_voxels ∈ {0, 50, 100, 250, 500} (per
``snakemake/config.yaml``). This script loads the existing per-(subject,
mask, nvoxels) decode TSVs and asks: what happens to the within-subject
decoding correlation as voxel count varies?

Two panels per page:

1. **V1 gabor decoding** — y = mean within-run circular correlation
   (Jammalamadaka–Sarma, π-periodic), averaged across runs per subject.
2. **NPCr value decoding** — y = mean within-run Pearson r, averaged
   across runs per subject.

X axis is n_voxels (log-spaced). One line per subject (translucent); a
bold group mean ± SEM overlay summarises the cohort.

Unsmoothed BOLD only by default (smoothing didn't move the needle in
prior analyses); pass ``--include-smoothed`` to add a second curve set.

Usage:
    python -m abstract_values.visualize.check_voxel_count_sweep
    python -m abstract_values.visualize.check_voxel_count_sweep --subjects 08 09
    python -m abstract_values.visualize.check_voxel_count_sweep --lambd 0.0
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from abstract_values.encoding_models.voxel_selection import usable_folds
from abstract_values.utils.data import BIDS_FOLDER

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 4,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "legend.frameon": False, "legend.handlelength": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})
sns.set_context("paper")

DERIV = Path(BIDS_FOLDER) / "derivatives" / "decoding"
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "voxel_count_sweep.pdf"

PERIOD = np.pi   # gabors are π-periodic


def list_nvoxels(decoder: str, subject: str, mask: str,
                 smoothed_suffix: str, lambd: float) -> list[str]:
    """Return the available nvoxels tokens for (decoder, subject, mask).

    Tokens are the raw string after ``_nvoxels-`` in the filename, so they
    include numeric values (e.g. ``"100"``) as well as the
    mixture-criterion labels ``"fdrNN"`` and ``"psigNN"``.
    """
    base = DERIV / decoder / f"sub-{subject}" / "func"
    if not base.exists():
        return []
    lam_tag = f"_lambda-{lambd}" if lambd != 0.0 else ""
    pat = re.compile(
        rf"sub-{subject}_mask-{mask}_nvoxels-(\d+|fdr\d+|psig\d+)"
        rf"_noise-full{re.escape(smoothed_suffix)}"
        rf"{re.escape(lam_tag)}_pars\.tsv$")
    nvs = []
    for p in base.iterdir():
        m = pat.match(p.name)
        if m:
            nvs.append(m.group(1))
    return _sort_nvox(nvs)


def _sort_nvox(tokens: list[str]) -> list[str]:
    """Stable order: numeric ascending first, then fdrNN ascending, then
    psigNN ascending."""
    def key(tok: str):
        if tok.startswith("fdr"):
            return (1, int(tok[3:]))
        if tok.startswith("psig"):
            return (2, int(tok[4:]))
        return (0, int(tok))
    return sorted(set(tokens), key=key)


def _nvox_label(tok: str) -> str:
    """Pretty label for the x-axis tick of a given nvox token."""
    if tok.startswith("fdr"):
        nn = int(tok[3:])
        return f"FDR≤0.{nn:02d}"
    if tok.startswith("psig"):
        nn = int(tok[4:])
        return f"P(sig)≥0.{nn:02d}"
    return tok if tok != "0" else "0\n(all CV)"


def load_pars(decoder: str, subject: str, mask: str, nv: str,
              smoothed_suffix: str, lambd: float) -> pd.DataFrame | None:
    lam_tag = f"_lambda-{lambd}" if lambd != 0.0 else ""
    fn = (DERIV / decoder / f"sub-{subject}" / "func"
          / f"sub-{subject}_mask-{mask}_nvoxels-{nv}_noise-full{smoothed_suffix}"
            f"{lam_tag}_pars.tsv")
    if not fn.exists():
        return None
    df = pd.read_csv(fn, sep="\t", index_col=[0, 1, 2])
    if df.empty:
        # No fold was decodable — treat as missing rather than as a data point.
        return None
    return df


def circular_correlation(true_rad: np.ndarray, pred_rad: np.ndarray,
                         period: float = PERIOD) -> float:
    """Jammalamadaka–Sarma circular correlation for π-periodic angles."""
    if len(true_rad) < 2:
        return float("nan")
    scale = 2 * np.pi / period
    a = (true_rad * scale).astype(np.float64)
    b = (pred_rad * scale).astype(np.float64)
    a_m = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    b_m = np.arctan2(np.sin(b).mean(), np.cos(b).mean())
    sa, sb = np.sin(a - a_m), np.sin(b - b_m)
    num = (sa * sb).sum()
    den = np.sqrt((sa * sa).sum() * (sb * sb).sum())
    return float(num / den) if den > 0 else float("nan")


def posterior_circular_mean(grid: np.ndarray, posteriors: np.ndarray,
                            period: float = PERIOD) -> np.ndarray:
    scale = 2 * np.pi / period
    a = grid.astype(np.float64) * scale
    s = posteriors @ np.sin(a)
    c = posteriors @ np.cos(a)
    return ((np.arctan2(s, c) / scale) % period).astype(np.float32)


def posterior_mean_linear(grid: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
    return (posteriors * grid[None, :]).sum(axis=1).astype(np.float32)


def metric_per_run(df: pd.DataFrame, decoder: str) -> float:
    """Mean within-run metric (circ corr for gabor, Pearson r for value).

    Averages within each (session, run) first, then across runs — the
    'within-run then across-run' aggregation the user prefers.
    """
    if decoder == "gabor":
        truth = df["true_orientation_rad"].to_numpy(np.float32)
        truth_col = "true_orientation_rad"
    else:
        truth = df["true_value_chf"].to_numpy(np.float32)
        truth_col = "true_value_chf"

    grid = np.asarray(df.columns.drop(truth_col), dtype=np.float32)
    posteriors = df.drop(columns=truth_col).to_numpy(np.float32)
    posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)
    if decoder == "gabor":
        decoded = posterior_circular_mean(grid, posteriors)
    else:
        decoded = posterior_mean_linear(grid, posteriors)

    rs = []
    for (s, r), grp_idx in df.groupby(level=["session", "run"]).indices.items():
        if len(grp_idx) < 3:
            continue
        t = truth[grp_idx]
        p = decoded[grp_idx]
        if decoder == "gabor":
            rs.append(circular_correlation(t, p))
        else:
            if t.std() == 0 or p.std() == 0:
                continue
            rs.append(float(np.corrcoef(t, p)[0, 1]))
    return float(np.nanmean(rs)) if rs else float("nan")


def collect_sweep(subjects: list[str], decoder: str, mask: str,
                   smoothed_suffix: str, lambd: float) -> pd.DataFrame:
    rows = []
    for sub in subjects:
        for nv in list_nvoxels(decoder, sub, mask, smoothed_suffix, lambd):
            df = load_pars(decoder, sub, mask, nv, smoothed_suffix, lambd)
            if df is None:
                continue
            m = metric_per_run(df, decoder)
            rows.append({"subject": sub, "nvoxels": nv, "metric": m})
    return pd.DataFrame(rows)


def _aggregate_n_voxels_selected(decoder: str, subject: str, mask: str,
                                  nv: str, smoothed_suffix: str,
                                  lambd: float) -> float | None:
    """Look up the mean per-fold n_voxels_selected from the sidecar meta.

    Used for the FDR / p_signal tokens to give a sense of how many
    voxels actually survived the criterion. Returns ``None`` if the
    meta TSV is missing (older runs) or if no fold was decodable.

    Folds marked undecodable in the sidecar (``status`` column) are excluded:
    averaging their 0 in would understate the count for the folds that did
    select voxels.
    """
    lam_tag = f"_lambda-{lambd}" if lambd != 0.0 else ""
    fn = (DERIV / decoder / f"sub-{subject}" / "func"
          / f"sub-{subject}_mask-{mask}_nvoxels-{nv}_noise-full{smoothed_suffix}"
            f"{lam_tag}_meta.tsv")
    if not fn.exists():
        return None
    try:
        meta = usable_folds(pd.read_csv(fn, sep="\t"))
        if meta.empty:
            return None
        return float(meta["n_voxels_selected"].mean())
    except Exception:
        return None


def _plot_panel(ax, df: pd.DataFrame, decoder: str, smooth_label: str):
    metric_name = "Circular correlation" if decoder == "gabor" else "Pearson r"
    label = f"Pooled across runs · within-subject ({smooth_label.lower()})"
    if df.empty:
        ax.text(0.5, 0.5, f"No data ({decoder}, {smooth_label})",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
        return

    # Categorical x-axis — evenly-spaced tick positions; nvoxels tokens
    # mix numeric counts with the criterion labels (fdrNN, psigNN), so
    # we sort them with the explicit comparator from list_nvoxels.
    grp = df.groupby("nvoxels")["metric"]
    nv_vals = _sort_nvox(list(grp.groups))
    x_pos = {nv: i for i, nv in enumerate(nv_vals)}

    # Per-subject lines (light) — only connect points within the same
    # category (numeric / fdr / psig) so a step between them isn't
    # implied to be a continuous sweep.
    def _cat(tok: str) -> str:
        if tok.startswith("fdr"):
            return "fdr"
        if tok.startswith("psig"):
            return "psig"
        return "num"

    for sub, sub_df in df.groupby("subject"):
        sub_df = sub_df.copy()
        sub_df["_cat"] = sub_df["nvoxels"].map(_cat)
        for _, seg in sub_df.groupby("_cat"):
            seg = seg.sort_values("nvoxels",
                                  key=lambda s: [x_pos[v] for v in s])
            xs = [x_pos[nv] for nv in seg["nvoxels"]]
            ax.plot(xs, seg["metric"], "-o", color="0.75",
                    lw=0.6, ms=3, zorder=1)

    # Group mean ± SEM
    means = np.asarray([grp.get_group(nv).mean() for nv in nv_vals])
    sems  = np.asarray([grp.get_group(nv).std(ddof=1) /
                        np.sqrt(grp.get_group(nv).count())
                        for nv in nv_vals])
    xs = [x_pos[nv] for nv in nv_vals]
    # Connect points by category only (avoid implying a sweep between
    # numeric counts and criterion labels).
    cats = [_cat(nv) for nv in nv_vals]
    for c in ("num", "fdr", "psig"):
        idx = [i for i, cc in enumerate(cats) if cc == c]
        if not idx:
            continue
        ax.plot([xs[i] for i in idx], means[idx], "-o", color="#3B5BA5",
                lw=1.6, ms=5, zorder=3,
                label="Group mean ± SEM" if c == "num" else None)
        ax.fill_between([xs[i] for i in idx],
                        (means - sems)[idx], (means + sems)[idx],
                        color="#3B5BA5", alpha=0.22, linewidth=0)

    # Best n_voxels marker (across all categories)
    if not np.all(np.isnan(means)):
        best = nv_vals[int(np.nanargmax(means))]
        ax.axvline(x_pos[best], color="#C44E52", lw=0.8, ls="--", zorder=2)
        ax.text(x_pos[best], ax.get_ylim()[1], f" Best: {best}",
                color="#C44E52", fontsize=7, va="top", ha="left")

    ax.set_xticks(xs)
    ax.set_xticklabels([_nvox_label(nv) for nv in nv_vals],
                        fontsize=8, rotation=20, ha="right")
    ax.set_xlabel("n_voxels")
    ax.set_ylabel(metric_name)
    ax.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)


def plot_page(subjects: list[str], lambd: float, smooth_label: str,
              smoothed_suffix: str):
    df_g = collect_sweep(subjects, "gabor", "BensonV1", smoothed_suffix, lambd)
    df_v = collect_sweep(subjects, "value", "NPCr",     smoothed_suffix, lambd)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), constrained_layout=True)
    fig.suptitle(f"Decoding accuracy vs n_voxels  ·  λ={lambd}  ·  {smooth_label}",
                 fontsize=10, y=1.02)

    axes[0].set_title("V1 gabor (orientation)", fontsize=9, color="0.2")
    _plot_panel(axes[0], df_g, "gabor", smooth_label)
    axes[1].set_title("NPCr value (CHF)", fontsize=9, color="0.2")
    _plot_panel(axes[1], df_v, "value", smooth_label)

    sns.despine(fig=fig, offset=5, trim=True)
    return fig, df_g, df_v


def discover_subjects() -> list[str]:
    subs = set()
    for d in ("gabor", "value"):
        if not (DERIV / d).exists():
            continue
        for p in (DERIV / d).glob("sub-*"):
            subs.add(p.name.removeprefix("sub-"))
    return sorted(subs)


def run(subjects: list[str] | None, lambd: float, include_smoothed: bool,
        out: Path):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No decode TSVs found.")

    variants = [("", "Unsmoothed")]
    if include_smoothed:
        variants.append(("_smoothed", "Smoothed"))

    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        all_rows = []
        for suffix, label in variants:
            print(f"--- {label} ---")
            fig, df_g, df_v = plot_page(subjects, lambd, label, suffix)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            df_g["decoder"] = "gabor"; df_g["smoothing"] = label
            df_v["decoder"] = "value"; df_v["smoothing"] = label
            all_rows.extend([df_g, df_v])
        # Persist the long-form aggregate as a sidecar TSV for further analysis
        if all_rows:
            agg = pd.concat(all_rows, ignore_index=True)
            tsv = out.with_suffix(".tsv")
            agg.to_csv(tsv, sep="\t", index=False)
            print(f"Sidecar TSV: {tsv}")
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+",
                   help="Subject labels (default: discover all on disk)")
    p.add_argument("--lambd", type=float, default=0.1,
                   help="Decoder regularisation λ (default 0.1)")
    p.add_argument("--include-smoothed", action="store_true")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, args.lambd, args.include_smoothed, Path(args.out))


if __name__ == "__main__":
    main()
