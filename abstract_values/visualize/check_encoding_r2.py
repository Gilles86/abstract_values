"""QA: encoding-model R² brain map — orange-only ramp.

For each (subject, encoding model, smoothing variant) on disk, render a
3-row mosaic (axial / coronal / sagittal) of the encoding R² NIfTI on
the T1w background. Single hue (orange) so the map reads as
"how strongly does this voxel agree with the encoding model?" with no
distraction from a diverging colorbar.

Models scanned by default: aprf, aprf-weighted, vonmises,
aprf-session-shift. Pass ``--models`` to restrict.

Usage
-----
    python -m abstract_values.visualize.check_encoding_r2
    python -m abstract_values.visualize.check_encoding_r2 --models aprf
    python -m abstract_values.visualize.check_encoding_r2 --subjects 03 04 \\
        --threshold 0.03 --vmax 0.4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from nilearn import image, plotting

from abstract_values.utils.data import BIDS_FOLDER

DERIV = Path(BIDS_FOLDER) / "derivatives"
DEFAULT_OUT = DERIV / "qa" / "encoding_r2.pdf"
DEFAULT_MODELS = ("aprf", "aprf-weighted", "vonmises", "aprf-session-shift")
SMOOTH_SUFFIXES = ("", "_smoothed")


def _orange_cmap():
    """Pure-orange ramp from low-alpha pastel to deep saturated orange.

    Avoids the white→yellow→red palette traps that confuse "value is
    low" with "no signal here". Single hue, transparent low, opaque high.
    """
    return LinearSegmentedColormap.from_list(
        "raw_orange",
        [(1.0, 0.95, 0.86), (1.0, 0.78, 0.45), (0.94, 0.49, 0.13), (0.55, 0.21, 0.05)])


def find_t1w(subject: str) -> Path | None:
    for cand in sorted((DERIV / "fmriprep" / subject)
                       .glob("ses-*/anat/*desc-preproc_T1w.nii.gz")):
        return cand
    return None


def find_r2(subject: str, model: str, smooth: str) -> Path | None:
    p = (DERIV / "encoding_models" / model / subject / "func"
         / f"{subject}_task-abstractvalue_space-T1w_desc-r2{smooth}_pe.nii.gz")
    return p if p.exists() else None


def find_mask(subject: str, desc: str, hemi: str | None = None) -> Path | None:
    base = DERIV / "masks" / subject / "anat"
    fn = (base / f"{subject}_space-T1w_hemi-{hemi}_desc-{desc}_mask.nii.gz"
          if hemi else
          base / f"{subject}_space-T1w_desc-{desc}_mask.nii.gz")
    return fn if fn.exists() else None


def _as_3d(p: Path):
    img = image.load_img(str(p))
    if img.ndim == 4:
        img = image.index_img(img, 0)
    return img


def collect_overlays(subject: str):
    overlays = []
    v1 = find_mask(subject, "BensonV1", hemi="LR")
    if v1 is not None:
        overlays.append((_as_3d(v1), "#7ec8ff"))
    npcr = find_mask(subject, "NPCr")
    if npcr is not None:
        overlays.append((_as_3d(npcr), "#a6e8a6"))
    return overlays


def plot_one(t1w: Path, r2: Path, overlays,
             title: str, threshold: float, vmax: float):
    fig = plt.figure(figsize=(14, 13))
    rows = [
        ("z", (0.02, 0.68, 0.96, 0.27)),
        ("y", (0.02, 0.36, 0.96, 0.27)),
        ("x", (0.02, 0.04, 0.96, 0.27)),
    ]
    cmap = _orange_cmap()
    for i, (mode, axes_rect) in enumerate(rows):
        display = plotting.plot_stat_map(
            str(r2), bg_img=str(t1w), display_mode=mode,
            cut_coords=8, threshold=threshold,
            vmin=threshold, vmax=vmax,
            cmap=cmap, symmetric_cbar=False,
            colorbar=(i == 0),
            title=title if i == 0 else None,
            figure=fig, axes=axes_rect, dim=-0.5, draw_cross=False,
        )
        for mask_img, color in overlays:
            display.add_contours(mask_img, levels=[0.5],
                                 colors=[color], linewidths=0.25)
    return fig


def run(subjects, models, threshold, vmax, out, show_rois=True):
    sub_dirs = sorted(p for p in (DERIV / "fmriprep").glob("sub-*") if p.is_dir())
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
            for model in models:
                for smooth in SMOOTH_SUFFIXES:
                    r2 = find_r2(sub, model, smooth)
                    if r2 is None:
                        continue
                    overlays = collect_overlays(sub) if show_rois else []
                    roi_note = "  (V1=blue, NPCr=green)" if overlays else ""
                    title = (f"{sub}  {model}  R²{smooth}  "
                             f"threshold ≥ {threshold:.2f}{roi_note}")
                    print(f"  rendering {title}")
                    fig = plot_one(t1w, r2, overlays, title, threshold, vmax)
                    pdf.savefig(fig, bbox_inches="tight", dpi=200)
                    plt.close(fig)
                    pages += 1
    print(f"Wrote {out}  ({pages} pages)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold", type=float, default=0.05,
                   help="R² floor for the colormap (fraction, default 0.05)")
    p.add_argument("--vmax", type=float, default=0.35,
                   help="R² upper bound for the colormap (default 0.35)")
    p.add_argument("--subjects", nargs="+",
                   help="Subject labels (default: all under derivatives/fmriprep)")
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                   help="Encoding-model subdir names under derivatives/encoding_models/")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--no-rois", action="store_true",
                   help="Skip V1/NPCr outline overlays")
    args = p.parse_args()
    run(args.subjects, args.models, args.threshold, args.vmax,
        Path(args.out), show_rois=not args.no_rois)


if __name__ == "__main__":
    main()
