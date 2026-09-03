"""Architecture x space x flexibility, decoupled.

Phase 2 and 3 of notes/model_comparison_plan.md.

The `vonmises` vs `aprf` contrast everyone has been reading confounds two
things: an 8-weight orientation basis set against a 4-parameter value bell. So
"orientation beats value" could equally be "a basis set beats a single bell".
This crosses the two factors properly:

                    orientation                value
    linear          vonmises n_basis=2 *       aprf-linear.cv
    one bell        vonmises-prf.cv            aprf.cv
    basis set       vonmises.cv                aprf-weighted.cv

    * not fitted by default; the honest minimal orientation model, since a
      monotonic ramp has no meaning on a circular variable.

**Phase 2** compares those cells per vertex on cvR2, always against the
subject's own `aprf-null.cv`. Two rules that earlier analyses got wrong:

  * cvR2, never full-fit R2. Cells differ in parameter count by 2x or more,
    and full-fit R2 rewards that directly.
  * EQUAL-SIZED POOLS. Comparing best-of-3 value models against one
    orientation model inflated the value side badly enough to invert the V1
    result. One model per cell, or equal-sized max-pools.

**Phase 3** treats flexibility as the result rather than a nuisance level.
Within a session, value is a deterministic monotonic function of orientation,
so any flexible model fits either space equally well — the spaces are only
distinguishable across the cdf/inverse_cdf flip. So the informative quantity is
how much a per-session shift *buys* in each space:

    delta = cvR2(shifted) - cvR2(joint)

A voxel genuinely coding value should gain little from shifting in value space
and a lot from shifting in orientation space; a voxel coding orientation, the
reverse. That asymmetry is a far sharper test than which curve fits better.

Usage
-----
    python -m abstract_values.visualize.factorial_model_comparison \\
        --out notes/figures/factorial_model_comparison.pdf
    python -m abstract_values.visualize.factorial_model_comparison \\
        --rois BensonV1ecc075-375 NPCr --tsv notes/data/factorial.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize.group_surface_maps import (
    discover_subjects, load_fsaverage)

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

NULL_MODEL = "aprf-null.cv"

# (architecture, space) -> cv dir
CELLS = {
    ("one bell", "orientation"): "vonmises-prf.cv",
    ("one bell", "value"):       "aprf.cv",
    ("basis set", "orientation"): "vonmises.cv",
    ("basis set", "value"):       "aprf-weighted.cv",
    ("linear", "value"):          "aprf-linear.cv",
}
ARCHS = ["linear", "one bell", "basis set"]
SPACES = ["orientation", "value"]

# Phase 3: joint -> its per-session-shifted counterpart, per space.
SHIFT_PAIRS = {
    "value":       ("aprf.cv", "aprf-shift.cv"),
    "orientation": ("vonmises.cv", "vonmises-shift.cv"),
}


def roi_mask(deriv, roi):
    """fsaverage vertex mask for an ROI, from the neuropythy atlas or FreeSurfer.

    Benson areas come from the atlas (optionally eccentricity-restricted);
    anything else falls back to pycortex's fsaverage overlay ROIs.
    """
    import cortex
    n = 327684
    if roi.startswith("Benson"):
        # e.g. BensonV1ecc075-375 -> area 1, band 0.75-3.75
        import re
        m = re.match(r"BensonV(\d+)(?:ecc(\d+)-(\d+))?$", roi)
        if not m:
            raise ValueError(f"cannot parse Benson ROI {roi!r}")
        area = int(m.group(1))
        sub0 = discover_subjects(deriv)[0]
        va, ec = [], []
        for hemi in ("L", "R"):
            base = (deriv / "neuropythy_atlas" / f"sub-{sub0}" /
                    f"sub-{sub0}_desc-benson14{{}}_space-fsaverage_hemi-{hemi}.func.gii")
            import nibabel as nib
            va.append(nib.load(str(base).format("Varea")).darrays[0].data)
            ec.append(nib.load(str(base).format("Eccen")).darrays[0].data)
        varea = np.concatenate(va)
        eccen = np.concatenate(ec)
        mask = np.round(varea).astype(int) == area
        if m.group(2):
            lo = float(m.group(2)) / 100.0
            hi = float(m.group(3)) / 100.0
            mask &= (eccen >= lo) & (eccen <= hi)
        return mask
    rv = cortex.get_roi_verts("fsaverage")
    mask = np.zeros(n, bool)
    # The project's volumetric masks encode hemisphere in the desc (NPCr /
    # NPCl); the pycortex fsaverage overlay uses a _L/_R suffix instead. Accept
    # both spellings so a name that works for get_roi_mask() works here too.
    trailing = {"r": "_R", "l": "_L"}.get(roi[-1:])
    if trailing and roi[:-1] + trailing in rv:
        mask[rv[roi[:-1] + trailing]] = True
    else:
        for side in ("_L", "_R"):
            if roi + side in rv:
                mask[rv[roi + side]] = True
        if not mask.any() and roi in rv:
            mask[rv[roi]] = True
    if not mask.any():
        # Silently returning an empty mask makes every downstream statistic
        # NaN, which reads as "no effect" rather than "wrong ROI name".
        raise ValueError(
            f"ROI {roi!r} matched no fsaverage vertices. Known overlay ROIs: "
            f"{sorted(rv)[:12]}... (or a Benson name like BensonV1ecc075-375)")
    return mask


def collect(deriv, subjects, smoothed=False):
    """Long table: one row per (subject, cell, vertex-set summary)."""
    rows = []
    for s in subjects:
        null = load_fsaverage(deriv, NULL_MODEL, s, "cvr2", smoothed)
        if null is None:
            continue
        cell_maps = {}
        for (arch, space), cvdir in CELLS.items():
            m = load_fsaverage(deriv, cvdir, s, "cvr2", smoothed)
            if m is not None:
                cell_maps[(arch, space)] = np.nan_to_num(m, nan=-np.inf)
        if len(cell_maps) < 2:
            continue
        rows.append((s, null, cell_maps))
    return rows


def phase2(deriv, subjects, rois, smoothed=False):
    """Win share per cell, within each ROI, with equal-sized pools."""
    data = collect(deriv, subjects, smoothed)
    out = []
    masks = {r: roi_mask(deriv, r) for r in rois}
    for s, null, cells in data:
        keys = sorted(cells)
        stack = np.vstack([cells[k] for k in keys])
        signal = stack.max(0) > null
        win = np.argmax(stack, 0)
        for r, m in masks.items():
            g = m & signal
            if g.sum() < 20:
                continue
            for i, k in enumerate(keys):
                out.append(dict(subject=s, roi=r, arch=k[0], space=k[1],
                                smoothed=smoothed,
                                win_share=float(np.mean(win[g] == i)),
                                n_vertices=int(g.sum()),
                                median_margin=float(np.median(
                                    (cells[k] - null)[g]))))
    return pd.DataFrame(out)


def phase3(deriv, subjects, rois, smoothed=False):
    """How much does a per-session shift buy, in each space?"""
    masks = {r: roi_mask(deriv, r) for r in rois}
    out = []
    for s in subjects:
        null = load_fsaverage(deriv, NULL_MODEL, s, "cvr2", smoothed)
        if null is None:
            continue
        for space, (joint_dir, shift_dir) in SHIFT_PAIRS.items():
            j = load_fsaverage(deriv, joint_dir, s, "cvr2", smoothed)
            sh = load_fsaverage(deriv, shift_dir, s, "cvr2", smoothed)
            if j is None or sh is None:
                continue
            j, sh = np.nan_to_num(j, nan=-np.inf), np.nan_to_num(sh, nan=-np.inf)
            signal = np.maximum(j, sh) > null
            for r, m in masks.items():
                g = m & signal
                if g.sum() < 20:
                    continue
                out.append(dict(subject=s, roi=r, space=space,
                                smoothed=smoothed, n_vertices=int(g.sum()),
                                delta=float(np.median((sh - j)[g])),
                                frac_shift_wins=float(np.mean(sh[g] > j[g]))))
    return pd.DataFrame(out)


def figure(df2, df3, out_pdf):
    rois = sorted(df2["roi"].unique()) if len(df2) else []
    ncol = max(len(rois), 1)
    fig, axes = plt.subplots(2, ncol, figsize=(4.4 * ncol, 8), squeeze=False)

    for j, r in enumerate(rois):
        ax = axes[0][j]
        sub = df2[df2["roi"] == r]
        piv = (sub.groupby(["arch", "space"])["win_share"].mean()
                  .unstack("space").reindex(index=ARCHS, columns=SPACES))
        im = ax.imshow(100 * piv.values, cmap="magma", vmin=0,
                       vmax=100 * np.nanmax(piv.values) if piv.size else 1)
        ax.set_xticks(range(len(SPACES)), SPACES)
        ax.set_yticks(range(len(ARCHS)), ARCHS)
        for a in range(piv.shape[0]):
            for b in range(piv.shape[1]):
                v = piv.values[a, b]
                if np.isfinite(v):
                    ax.text(b, a, f"{100*v:.0f}%", ha="center", va="center",
                            color="w" if v < 0.6 * np.nanmax(piv.values) else "k",
                            fontsize=10, fontweight="bold")
        n = sub["subject"].nunique()
        ax.set_title(f"{r}\nwin share, n={n}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, label="% of signal vertices won")

        ax = axes[1][j]
        s3 = df3[df3["roi"] == r]
        if len(s3):
            for i, space in enumerate(SPACES):
                v = s3[s3["space"] == space]["delta"].to_numpy()
                if not len(v):
                    continue
                rng = np.random.default_rng(0)
                ax.scatter(i + rng.uniform(-.14, .14, len(v)), v, s=18,
                           alpha=.6, linewidths=0,
                           color="#3B5BA5" if space == "orientation" else "#E76F51")
                ax.hlines(np.nanmean(v), i - .28, i + .28, lw=2.4,
                          color="#3B5BA5" if space == "orientation" else "#E76F51")
            ax.axhline(0, color="#999", lw=.8, ls=(0, (4, 3)))
            ax.set_xticks(range(len(SPACES)), SPACES)
            ax.set_ylabel("Median ΔcvR² (shifted − joint)")
            ax.set_title("What a per-session shift buys", fontsize=9)

    fig.suptitle("Architecture × space, and what flexibility buys "
                 "(cross-validated)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--rois", nargs="+",
                   default=["BensonV1ecc075-375", "BensonV1", "NPC"])
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--out", default="notes/figures/factorial_model_comparison.pdf")
    p.add_argument("--tsv", default=None)
    a = p.parse_args()

    deriv = Path(a.bids_folder) / "derivatives"
    subjects = a.subjects or discover_subjects(deriv)
    print(f"{len(subjects)} subjects; ROIs: {', '.join(a.rois)}")

    df2 = phase2(deriv, subjects, a.rois, a.smoothed)
    df3 = phase3(deriv, subjects, a.rois, a.smoothed)
    if len(df2):
        print("\nPhase 2 — win share (%):")
        print((100 * df2.groupby(["roi", "arch", "space"])["win_share"].mean()
               ).round(1).to_string())
    if len(df3):
        print("\nPhase 3 — median ΔcvR² from allowing a per-session shift:")
        print(df3.groupby(["roi", "space"])["delta"].mean().round(5).to_string())
    if a.tsv:
        Path(a.tsv).parent.mkdir(parents=True, exist_ok=True)
        df2.to_csv(a.tsv, sep="\t", index=False)
        df3.to_csv(str(a.tsv).replace(".tsv", "_flex.tsv"), sep="\t", index=False)
        print(f"\nWrote {a.tsv}")
    figure(df2, df3, a.out)


if __name__ == "__main__":
    main()
