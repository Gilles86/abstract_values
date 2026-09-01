"""Per-subject "report card": where does one subject sit in the cohort?

Answers the question you actually ask when a new subject finishes the
pipeline — *is this one any good?* — by reducing every stage of the pipeline
to one number per subject and showing the whole cohort rank-ordered with the
subject of interest called out.

Metrics, one per pipeline stage:

    mean_fd             head motion (framewise displacement, mm), run mean
    pct_fd_gt_0p5       % of volumes a scrubbing pipeline would discard
    glm_r2_npcr         median GLMsingle type-D R² (%) inside NPCr
    glm_r2_v1           median GLMsingle type-D R² (%) inside BensonV1
    cvr2_win_npcr       % of NPCr voxels where the aPRF beats the null model
    dec_value_npcr_r    value decoding, mean within-run Pearson r, NPCr
    dec_gabor_v1_r      orientation decoding, mean within-run circular r, V1

The cvR² criterion is "beats ``aprf-null.cv`` per voxel", not "> 0" — see
``utils.data.cvr2_signal`` and the project note on the null baseline.

Split along the usual data-size boundary:

    # cluster: walk every subject's derivatives, reduce to one row per subject
    BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-abstractvalue \\
        python -m abstract_values.visualize.pipeline_report_card \\
            --summary-tsv notes/data/pipeline_report_card.tsv

    # local: read that TSV and draw the figure
    python -m abstract_values.visualize.pipeline_report_card \\
        --tsv notes/data/pipeline_report_card.tsv --highlight 29 \\
        --out notes/figures/report_card_sub-29.pdf
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

from abstract_values.utils.data import BIDS_FOLDER, Subject

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

HIGHLIGHT_COLOUR = "#E76F51"
COHORT_COLOUR = "#3B5BA5"
MEDIAN_COLOUR = "#8A8A8A"

DECODE_NOISE = "spherical"
DECODE_NVOXELS = "250"
DECODE_LAMBDA = 0.1
PERIOD = np.pi


# ── aggregation (cluster side) ───────────────────────────────────────────────

def discover_subjects(bids_folder: Path) -> list[str]:
    """Subject labels with a finished GLMsingle, study subjects before pilots."""
    d = bids_folder / "derivatives" / "glmsingle"
    subs = [p.name.removeprefix("sub-") for p in sorted(d.glob("sub-*"))
            if p.is_dir() and (p / "_done.touch").exists()]
    return sorted(subs, key=lambda s: (0 if s[0].isdigit() else 1, s))


def _mask_bool(sub: Subject, roi: str, hemi: str | None) -> np.ndarray | None:
    try:
        m = sub.get_roi_mask(roi, hemi=hemi)
    except FileNotFoundError:
        return None
    return np.asarray(m.get_fdata()) > 0


def _load_vol(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    import nibabel as nib
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    return data[..., 0] if data.ndim == 4 else data


def motion_row(sub: Subject) -> dict:
    """Run-mean FD and the % of volumes above 0.5 mm, pooled over runs."""
    fds = []
    for session in sub.get_sessions():
        for run in sub.get_runs(session):
            fn = (sub.bids_folder / "derivatives" / "fmriprep" /
                  f"sub-{sub.subject_id}" / f"ses-{session}" / "func" /
                  f"sub-{sub.subject_id}_ses-{session}_task-abstractvalue_"
                  f"run-{run}_desc-confounds_timeseries.tsv")
            if not fn.exists():
                continue
            df = pd.read_csv(fn, sep="\t", usecols=["framewise_displacement"])
            # fmriprep leaves volume 1 as NaN (nothing to difference against).
            fd = pd.to_numeric(df["framewise_displacement"],
                               errors="coerce").dropna().to_numpy()
            if fd.size:
                fds.append(fd)
    if not fds:
        return {"mean_fd": np.nan, "pct_fd_gt_0p5": np.nan, "n_runs": 0}
    allfd = np.concatenate(fds)
    return {"mean_fd": float(allfd.mean()),
            "pct_fd_gt_0p5": float(100.0 * (allfd > 0.5).mean()),
            "n_runs": len(fds)}


def glmsingle_row(sub: Subject) -> dict:
    """Median GLMsingle type-D R² (already in percent) inside two ROIs."""
    r2 = _load_vol(sub.bids_folder / "derivatives" / "glmsingle" /
                   f"sub-{sub.subject_id}" / "func" /
                   f"sub-{sub.subject_id}_task-abstractvalue_"
                   f"space-T1w_desc-R2_pe.nii.gz")
    out = {"glm_r2_npcr": np.nan, "glm_r2_v1": np.nan}
    if r2 is None:
        return out
    for key, roi, hemi in [("glm_r2_npcr", "NPCr", None),
                           ("glm_r2_v1", "BensonV1", "LR")]:
        m = _mask_bool(sub, roi, hemi)
        if m is None or m.shape != r2.shape:
            continue
        vals = r2[m]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[key] = float(np.median(vals))
    return out


def cvr2_row(sub: Subject, model: str = "aprf.cv",
             null_model: str = "aprf-null.cv") -> dict:
    """% of NPCr voxels where the aPRF's cvR² beats the null model's, and the
    median margin. Volumetric (T1w) so it stays in register with the ROI mask."""
    def cv(mdl):
        return _load_vol(sub.bids_folder / "derivatives" / "encoding_models" /
                         mdl / f"sub-{sub.subject_id}" / "func" /
                         f"sub-{sub.subject_id}_task-abstractvalue_"
                         f"space-T1w_desc-cvr2_pe.nii.gz")

    out = {"cvr2_win_npcr": np.nan, "cvr2_delta_npcr": np.nan}
    a, n = cv(model), cv(null_model)
    m = _mask_bool(sub, "NPCr", None)
    if a is None or n is None or m is None or a.shape != m.shape:
        return out
    delta = a[m] - n[m]
    delta = delta[np.isfinite(delta)]
    if delta.size:
        out["cvr2_win_npcr"] = float(100.0 * (delta > 0).mean())
        out["cvr2_delta_npcr"] = float(np.median(delta))
    return out


def _circular_correlation(a_rad, b_rad, period=PERIOD) -> float:
    """Jammalamadaka-Sarma correlation for pi-periodic (axial) angles."""
    if len(a_rad) < 2:
        return float("nan")
    scale = 2 * np.pi / period
    a, b = np.asarray(a_rad) * scale, np.asarray(b_rad) * scale
    a_m = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    b_m = np.arctan2(np.sin(b).mean(), np.cos(b).mean())
    sa, sb = np.sin(a - a_m), np.sin(b - b_m)
    den = np.sqrt((sa * sa).sum() * (sb * sb).sum())
    return float((sa * sb).sum() / den) if den > 0 else float("nan")


def _decode_metric(bids_folder: Path, label: str, decoder: str, mask: str,
                   nv: str = DECODE_NVOXELS, noise: str = DECODE_NOISE,
                   lambd: float = DECODE_LAMBDA) -> float:
    """Mean within-run decoding correlation from the decoder's _pars.tsv.

    Circular correlation for orientation, Pearson r for value; averaged within
    each (session, run) first, then across runs.
    """
    lam_tag = f"_lambda-{lambd}" if lambd != 0.0 else ""
    fn = (bids_folder / "derivatives" / "decoding" / decoder / f"sub-{label}" /
          "func" / f"sub-{label}_mask-{mask}_nvoxels-{nv}_noise-{noise}"
                   f"{lam_tag}_pars.tsv")
    if not fn.exists():
        return float("nan")
    df = pd.read_csv(fn, sep="\t", index_col=[0, 1, 2])
    if df.empty:
        return float("nan")

    truth_col = ("true_orientation_rad" if decoder == "gabor"
                 else "true_value_chf")
    if truth_col not in df.columns:
        return float("nan")
    truth = df[truth_col].to_numpy(np.float64)
    grid = np.asarray(df.columns.drop(truth_col), dtype=np.float64)
    post = df.drop(columns=truth_col).to_numpy(np.float64)
    post = post / post.sum(axis=1, keepdims=True)

    if decoder == "gabor":
        scale = 2 * np.pi / PERIOD
        s, c = post @ np.sin(grid * scale), post @ np.cos(grid * scale)
        decoded = (np.arctan2(s, c) / scale) % PERIOD
    else:
        decoded = post @ grid

    rs = []
    for _, idx in df.groupby(level=["session", "run"]).indices.items():
        t, p = truth[idx], decoded[idx]
        if len(idx) < 3:
            continue
        if decoder == "gabor":
            rs.append(_circular_correlation(t, p))
        elif t.std() > 0 and p.std() > 0:
            rs.append(float(np.corrcoef(t, p)[0, 1]))
    return float(np.nanmean(rs)) if rs else float("nan")


def summarise(bids_folder, subjects=None) -> pd.DataFrame:
    bids_folder = Path(bids_folder)
    subjects = subjects or discover_subjects(bids_folder)
    rows = []
    for label in subjects:
        sub = Subject(label, bids_folder=bids_folder)
        row = {"subject": label}
        row.update(motion_row(sub))
        row.update(glmsingle_row(sub))
        row.update(cvr2_row(sub))
        row["dec_value_npcr_r"] = _decode_metric(
            bids_folder, label, "value", "NPCr")
        row["dec_gabor_v1_r"] = _decode_metric(
            bids_folder, label, "gabor", "BensonV1")
        rows.append(row)
        print(f"  sub-{label}: " + "  ".join(
            f"{k}={row[k]:.3g}" for k in
            ("mean_fd", "glm_r2_npcr", "cvr2_win_npcr",
             "dec_value_npcr_r", "dec_gabor_v1_r")
            if isinstance(row.get(k), float)), flush=True)
    return pd.DataFrame(rows)


# ── plotting (local side) ────────────────────────────────────────────────────

# (column, axis label, higher-is-better)
METRICS = [
    ("mean_fd", "Mean FD (mm)", False),
    ("pct_fd_gt_0p5", "Volumes > 0.5 mm (%)", False),
    ("glm_r2_npcr", "GLMsingle R² in NPCr (%)", True),
    ("glm_r2_v1", "GLMsingle R² in V1 (%)", True),
    ("cvr2_win_npcr", "NPCr voxels beating null (%)", True),
    ("cvr2_delta_npcr", "Median cvR² − null in NPCr", True),
    ("dec_value_npcr_r", "Value decoding, NPCr (r)", True),
    ("dec_gabor_v1_r", "Orientation decoding, V1 (circ. r)", True),
]


def _rank_text(values: pd.Series, label: str, higher_is_better: bool) -> str:
    v = values.dropna()
    if label not in v.index:
        return "n/a"
    order = v.rank(ascending=not higher_is_better, method="min")
    return f"rank {int(order[label])}/{len(v)}"


def plot(df: pd.DataFrame, highlight: str | None, out: str):
    """One panel per metric: cohort rank-ordered, highlighted subject called out."""
    df = df.copy()
    df["subject"] = df["subject"].astype(str)
    ncol = 2
    nrow = int(np.ceil(len(METRICS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.0, 1.9 * nrow),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (col, ylabel, higher) in zip(axes, METRICS):
        s = df.set_index("subject")[col].dropna()
        if s.empty:
            ax.set_axis_off()
            continue
        # Rank-order so the highlighted subject's position IS the message.
        s = s.sort_values(ascending=not higher)
        x = np.arange(len(s))
        colours = [HIGHLIGHT_COLOUR if lab == highlight else COHORT_COLOUR
                   for lab in s.index]
        sizes = [34 if lab == highlight else 15 for lab in s.index]
        ax.scatter(x, s.to_numpy(), c=colours, s=sizes, zorder=3,
                   linewidths=0, clip_on=False)
        ax.axhline(float(np.median(s)), color=MEDIAN_COLOUR, lw=0.8,
                   ls=(0, (4, 3)), zorder=1)

        ax.set_xticks(x)
        ax.set_xticklabels(s.index, rotation=90)
        for tick, lab in zip(ax.get_xticklabels(), s.index):
            if lab == highlight:
                tick.set_color(HIGHLIGHT_COLOUR)
                tick.set_fontweight("bold")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.8, len(s) - 0.2)

        if highlight in s.index:
            ax.set_title(
                f"sub-{highlight}: {s[highlight]:.3g}  "
                f"({_rank_text(df.set_index('subject')[col], highlight, higher)}"
                f", {'higher' if higher else 'lower'} is better)",
                loc="left", fontsize=8, color=HIGHLIGHT_COLOUR)
        else:
            ax.set_title(ylabel, loc="left", fontsize=8)

    for ax in axes[len(METRICS):]:
        ax.set_axis_off()

    n = df["subject"].nunique()
    fig.suptitle(
        f"Pipeline report card — sub-{highlight} vs cohort (n = {n})"
        if highlight else f"Pipeline report card (n = {n})",
        fontsize=11, x=0.01, ha="left")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--summary-tsv", default=None,
                   help="aggregate from derivatives and write this TSV (cluster)")
    p.add_argument("--tsv", default=None,
                   help="read this TSV instead of aggregating (local)")
    p.add_argument("--highlight", default=None,
                   help="subject label to call out, e.g. 29")
    p.add_argument("--out", default=None, help="figure path (implies plotting)")
    args = p.parse_args()

    if args.tsv:
        df = pd.read_csv(args.tsv, sep="\t", dtype={"subject": str})
    else:
        df = summarise(args.bids_folder, args.subjects)
        if args.summary_tsv:
            outp = Path(args.summary_tsv)
            outp.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(outp, sep="\t", index=False)
            print(f"Wrote {outp}  ({len(df)} subjects)")

    if args.out:
        plot(df, args.highlight, args.out)


if __name__ == "__main__":
    main()
