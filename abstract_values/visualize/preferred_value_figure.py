"""Preferred value (aPRF mode) across cortex, and its distribution per ROI.

The question this answers: where a voxel's value tuning is real, *which* value
does it prefer — and does that differ between early visual cortex, parietal
cortex, and M1?

M1 is the interesting case. The BDM bid is made by moving a slider, so a bigger
bid is a longer movement. Any voxel whose response scales with bid magnitude
will fit a value model, but as a motor signal rather than a value code. The
signature to look for is preferred values **piling up at one end** of the CHF
range rather than tiling it: a genuine value code should have voxels preferring
values across the range, a movement-magnitude confound should not.

Vertices are included per subject only where the aPRF's cross-validated R2
beats that subject's own aprf-null.cv, which is the project's per-voxel signal
test — a whole-ROI summary without that gate is dominated by voxels with no
value response at all.

Usage
-----
    python -m abstract_values.visualize.preferred_value_figure \\
        --out notes/figures/preferred_value.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize.group_surface_maps import (
    discover_subjects, load_fsaverage)

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
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

FS_LABEL_DIR = ("/data/ds-abstractvalue/derivatives/fmriprep/sourcedata/"
                "freesurfer/fsaverage/label")
N_LH = 163842
N_VERT = 327684
NULL_MODEL = "aprf-null.cv"

# Value range the gabors actually spanned (experiment/README.md); modes pinned
# at the edge of the fitted range are unidentified, not a preference.
VALUE_MIN, VALUE_MAX = 2.0, 42.0

ROI_COLOURS = {"V1": "#3B5BA5", "NPC": "#5D8C3F", "M1": "#C44E52"}


def fs_label(*names):
    """Union of FreeSurfer fsaverage .label files, as an fsaverage vertex mask."""
    m = np.zeros(N_VERT, bool)
    for name in names:
        for hemi, off in (("lh", 0), ("rh", N_LH)):
            m[off + fsio.read_label(f"{FS_LABEL_DIR}/{hemi}.{name}.label")] = True
    return m


def benson_mask(deriv, subject, area=1, eccen=(0.75, 3.75)):
    """Benson area, restricted to the eccentricity band the gabor drives."""
    import nibabel as nib
    va, ec = [], []
    for hemi in ("L", "R"):
        base = (deriv / "neuropythy_atlas" / f"sub-{subject}" /
                f"sub-{subject}_desc-benson14{{}}_space-fsaverage_hemi-{hemi}.func.gii")
        va.append(nib.load(str(base).format("Varea")).darrays[0].data)
        ec.append(nib.load(str(base).format("Eccen")).darrays[0].data)
    varea, eccen_map = np.concatenate(va), np.concatenate(ec)
    return ((np.round(varea).astype(int) == area)
            & (eccen_map >= eccen[0]) & (eccen_map <= eccen[1]))


def collect(deriv, subjects, smoothed=False):
    """Per-subject preferred values inside each ROI, gated on beating the null."""
    import cortex
    rv = cortex.get_roi_verts("fsaverage")
    npc = np.zeros(N_VERT, bool)
    for k in ("NPC_L", "NPC_R"):
        if k in rv:
            npc[rv[k]] = True
    m1 = fs_label("BA4a_exvivo.thresh", "BA4p_exvivo.thresh")

    rows, per_vertex = [], []
    for s in subjects:
        mode = load_fsaverage(deriv, "aprf", s, "mode", smoothed)
        cv = load_fsaverage(deriv, "aprf.cv", s, "cvr2", smoothed)
        null = load_fsaverage(deriv, NULL_MODEL, s, "cvr2", smoothed)
        if mode is None or cv is None or null is None:
            continue
        signal = np.isfinite(cv - null) & ((cv - null) > 0)
        in_range = (mode >= VALUE_MIN) & (mode <= VALUE_MAX)
        good = signal & in_range & np.isfinite(mode)
        per_vertex.append((s, mode, good))

        for name, m in (("V1", benson_mask(deriv, s)), ("NPC", npc), ("M1", m1)):
            g = m & good
            if g.sum() < 20:
                continue
            v = mode[g]
            rows.append(dict(subject=s, roi=name, n=int(g.sum()),
                             median=float(np.median(v)),
                             # Fraction in the top/bottom fifth of the range —
                             # a code that tiles the range should have neither
                             # much above 0.2.
                             frac_high=float(np.mean(v > VALUE_MIN + 0.8 *
                                                     (VALUE_MAX - VALUE_MIN))),
                             frac_low=float(np.mean(v < VALUE_MIN + 0.2 *
                                                    (VALUE_MAX - VALUE_MIN)))))
    return pd.DataFrame(rows), per_vertex


def figure(df, per_vertex, deriv, out_pdf, smoothed=False):
    import cortex
    rv = cortex.get_roi_verts("fsaverage")
    npc = np.zeros(N_VERT, bool)
    for k in ("NPC_L", "NPC_R"):
        if k in rv:
            npc[rv[k]] = True
    rois = {"V1": benson_mask(deriv, per_vertex[0][0]), "NPC": npc,
            "M1": fs_label("BA4a_exvivo.thresh", "BA4p_exvivo.thresh")}

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.6),
                             gridspec_kw=dict(width_ratios=[1.35, 1]),
                             constrained_layout=True)

    # ── a. pooled distribution of preferred value, per ROI ───────────────────
    ax = axes[0]
    bins = np.linspace(VALUE_MIN, VALUE_MAX, 25)
    for name, m in rois.items():
        vals = np.concatenate([mode[m & good] for _, mode, good in per_vertex
                               if (m & good).sum() > 0])
        if not len(vals):
            continue
        h, _ = np.histogram(vals, bins=bins, density=True)
        centres = 0.5 * (bins[:-1] + bins[1:])
        ax.plot(centres, 100 * h * np.diff(bins)[0], color=ROI_COLOURS[name],
                lw=1.4)
        # Direct label at the curve's peak rather than a legend
        i = int(np.argmax(h))
        ax.annotate(name, (centres[i], 100 * h[i] * np.diff(bins)[0]),
                    textcoords="offset points", xytext=(4, 4),
                    color=ROI_COLOURS[name], fontsize=7.5, fontweight="bold")
    ax.axhline(100 / (len(bins) - 1), color="0.7", lw=0.6, ls="--", zorder=0)
    ax.annotate("Uniform", (VALUE_MAX, 100 / (len(bins) - 1)),
                textcoords="offset points", xytext=(-2, 3), ha="right",
                fontsize=6.5, color="0.45")
    ax.set_xlabel("Preferred value (CHF)")
    ax.set_ylabel("Vertices (%)")
    ax.set_xticks([2, 12, 22, 32, 42])

    # ── b. per-subject medians ───────────────────────────────────────────────
    ax = axes[1]
    order = [r for r in ("V1", "NPC", "M1") if r in set(df["roi"])]
    rng = np.random.default_rng(0)
    for i, r in enumerate(order):
        v = df[df["roi"] == r]["median"].to_numpy()
        ax.scatter(i + rng.uniform(-.15, .15, len(v)), v, s=14, alpha=.55,
                   color=ROI_COLOURS[r], linewidths=0, zorder=3)
        ax.hlines(np.median(v), i - .28, i + .28, color=ROI_COLOURS[r],
                  lw=2.2, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Median preferred value (CHF)")
    ax.set_ylim(VALUE_MIN, VALUE_MAX)
    ax.set_yticks([2, 12, 22, 32, 42])
    mid = 0.5 * (VALUE_MIN + VALUE_MAX)
    ax.axhline(mid, color="0.7", lw=0.6, ls="--", zorder=0)
    ax.annotate("Range midpoint", (len(order) - 0.5, mid),
                textcoords="offset points", xytext=(0, 3), ha="right",
                fontsize=6.5, color="0.45")

    for ax, letter in zip(axes, "ab"):
        ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="right")
    sns.despine(fig=fig, offset=4, trim=True)
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(out_pdf).replace(".pdf", ".svg"))
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--out", default="notes/figures/preferred_value.pdf")
    p.add_argument("--tsv", default="notes/data/preferred_value_by_roi.tsv")
    a = p.parse_args()

    deriv = Path(a.bids_folder) / "derivatives"
    subjects = a.subjects or discover_subjects(deriv)
    df, per_vertex = collect(deriv, subjects, a.smoothed)
    if df.empty:
        raise SystemExit("No subjects had mode + cv + null surfaces.")
    print(f"n={df['subject'].nunique()} subjects")
    print(df.groupby("roi")[["n", "median", "frac_low", "frac_high"]]
          .mean().round(3).to_string())
    if a.tsv:
        Path(a.tsv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.tsv, sep="\t", index=False)
    figure(df, per_vertex, deriv, a.out, a.smoothed)


if __name__ == "__main__":
    main()
