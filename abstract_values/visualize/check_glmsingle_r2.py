"""QA: thresholded GLMsingle R² maps in axial mosaic, one page per (subject, smoothing).

Looks at the GLMsingle R² (TYPE-D fracridge) for each subject, thresholds at
a configurable R² value (default 5%; note GLMsingle stores R² in percent,
not fraction — values range 0–100), and writes an axial-mosaic PDF — one
page per available (subject, smoothing variant). Useful as a quick eyeball
on overall SNR across subjects and to flag subjects with weirdly low or
high R² coverage.

Inputs are read from local sync:
    derivatives/glmsingle{_smoothed}/sub-XX/func/sub-XX_task-abstractvalue_space-T1w_desc-R2_pe.nii.gz
    derivatives/fmriprep/sub-XX/ses-1/anat/sub-XX_ses-1_desc-preproc_T1w.nii.gz

Output:
    derivatives/qa/glmsingle_r2.pdf

Usage:
    python -m abstract_values.visualize.check_glmsingle_r2
    python -m abstract_values.visualize.check_glmsingle_r2 --threshold 2
    python -m abstract_values.visualize.check_glmsingle_r2 --subjects 07 08
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import image, plotting

from abstract_values.utils.data import BIDS_FOLDER

DERIV = Path(BIDS_FOLDER) / "derivatives"
DEFAULT_OUT = DERIV / "qa" / "glmsingle_r2.pdf"
SMOOTH_VARIANTS = [("glmsingle", ""), ("glmsingle_smoothed", "_smoothed")]


def find_t1w(subject: str) -> Path | None:
    for cand in sorted((DERIV / "fmriprep" / subject).glob("ses-*/anat/*desc-preproc_T1w.nii.gz")):
        return cand
    return None


def find_r2(subject: str, deriv_dir: str) -> Path | None:
    p = DERIV / deriv_dir / subject / "func" / f"{subject}_task-abstractvalue_space-T1w_desc-R2_pe.nii.gz"
    return p if p.exists() else None


def find_mask(subject: str, desc: str, hemi: str | None = None) -> Path | None:
    base = DERIV / "masks" / subject / "anat"
    if hemi:
        p = base / f"{subject}_space-T1w_hemi-{hemi}_desc-{desc}_mask.nii.gz"
    else:
        p = base / f"{subject}_space-T1w_desc-{desc}_mask.nii.gz"
    return p if p.exists() else None


def _as_3d(path: Path):
    """Load a mask NIfTI and squeeze a trailing singleton if present (some
    masks are stored as 4D with a singleton time axis)."""
    img = image.load_img(str(path))
    if img.ndim == 4:
        img = image.index_img(img, 0)
    return img


# (mask_img, color) — subtle outlines added on top of each row.
def collect_overlays(subject: str):
    overlays = []
    v1 = find_mask(subject, "BensonV1", hemi="LR")
    if v1 is not None:
        overlays.append((_as_3d(v1), "#7ec8ff"))   # pale blue for V1
    npcr = find_mask(subject, "NPCr")
    if npcr is not None:
        overlays.append((_as_3d(npcr), "#a6e8a6")) # pale green for NPCr
    return overlays


def plot_one(t1w: Path, r2: Path, overlays,
             title: str, threshold: float, vmax: float):
    # 3 rows on one figure: axial (z), coronal (y), sagittal (x). Pass each
    # row's region to plot_stat_map as a (left, bottom, width, height) axes
    # rect — nilearn carves its own internal cuts layout inside that region.
    # Using SubFigures here doesn't work: plot_stat_map internally calls
    # `plt.figure(figure, ...)` which doesn't accept a SubFigure, so each
    # row would end up on a fresh hidden figure.
    fig = plt.figure(figsize=(14, 13))
    rows = [
        ("z", (0.02, 0.68, 0.96, 0.27)),   # axial   — top
        ("y", (0.02, 0.36, 0.96, 0.27)),   # coronal — middle
        ("x", (0.02, 0.04, 0.96, 0.27)),   # sagittal — bottom
    ]
    for i, (mode, axes_rect) in enumerate(rows):
        display = plotting.plot_stat_map(
            str(r2),
            bg_img=str(t1w),
            display_mode=mode,
            cut_coords=8,
            threshold=threshold,
            vmin=threshold,
            vmax=vmax,
            cmap="hot",
            colorbar=(i == 0),                # one colorbar at the top is enough
            title=title if i == 0 else None,
            figure=fig,
            axes=axes_rect,
            dim=-0.5,                         # brighten T1w background
            draw_cross=False,
        )
        for mask_img, color in overlays:
            display.add_contours(
                mask_img,
                levels=[0.5],
                colors=[color],
                linewidths=0.25,
            )
    return fig, None


def run(subjects: list[str] | None, threshold: float, vmax: float, out: Path,
        show_rois: bool = True):
    sub_dirs = sorted(
        p for p in (DERIV / "fmriprep").glob("sub-*") if p.is_dir()
    )
    if subjects:
        wanted = {f"sub-{s.lstrip('sub-').lstrip('-')}" for s in subjects}
        sub_dirs = [p for p in sub_dirs if p.name in wanted]

    if not sub_dirs:
        print(f"No subjects found under {DERIV/'fmriprep'}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    pages = 0
    with PdfPages(out) as pdf:
        for sub_dir in sub_dirs:
            sub = sub_dir.name
            t1w = find_t1w(sub)
            if t1w is None:
                print(f"{sub}: no T1w — skipping")
                continue

            for deriv_dir, smooth_suffix in SMOOTH_VARIANTS:
                r2 = find_r2(sub, deriv_dir)
                if r2 is None:
                    continue
                overlays = collect_overlays(sub) if show_rois else []
                roi_note = "  (V1=blue, NPCr=green)" if overlays else ""
                title = f"{sub}  R²{smooth_suffix}  threshold ≥ {threshold:.1f}%{roi_note}"
                print(f"  rendering {title}")
                fig, _ = plot_one(t1w, r2, overlays, title, threshold, vmax)
                pdf.savefig(fig, bbox_inches="tight", dpi=200)
                plt.close(fig)
                pages += 1
    print(f"Wrote {out}  ({pages} pages)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold", type=float, default=10.0,
                   help="R² threshold in PERCENT (default 10 = 10%% variance explained). "
                        "GLMsingle stores R² as percent (0-100), not fraction.")
    p.add_argument("--vmax", type=float, default=30.0,
                   help="Colorbar max in PERCENT (default 30 = 30%% variance explained)")
    p.add_argument("--subjects", nargs="+",
                   help="Restrict to these subject labels (e.g. 07 08 09)")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"Output PDF path (default {DEFAULT_OUT})")
    p.add_argument("--no-rois", action="store_true",
                   help="Drop the V1 + NPCr outline overlays entirely")
    args = p.parse_args()
    run(args.subjects, args.threshold, args.vmax, Path(args.out),
        show_rois=not args.no_rois)


if __name__ == "__main__":
    main()
