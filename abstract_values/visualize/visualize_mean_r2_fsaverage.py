"""Group-mean R² (aPRF + vonmises) on fsaverage cortex via pycortex.

Loads the fsaverage-space R² `.func.gii` files produced by
`sample_r2_to_surface.py`, averages across subjects per model, and
displays the means in a pycortex webgl viewer on the `fsaverage` subject.

Per model, alpha-masks vertices by the **group mean R²** itself with a
smooth Gaussian-CDF transition centred on `--r2-thr`, so weak-signal
vertices fade into the curvature background.

Usage:
    python -m abstract_values.visualize.visualize_mean_r2_fsaverage
    python -m abstract_values.visualize.visualize_mean_r2_fsaverage --subjects 07 08 09 10
    python -m abstract_values.visualize.visualize_mean_r2_fsaverage --r2-thr 5 --r2-sigma 2
    python -m abstract_values.visualize.visualize_mean_r2_fsaverage --models aprf vonmises aprf_session_shift

Note: GLMsingle and the encoding models in this project store R² in
PERCENT (0–100), not fraction. `--r2-thr` defaults follow that scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cortex
import numpy as np
from scipy.stats import norm

from abstract_values.surface.sampling import load_surface_data
from abstract_values.utils.data import BIDS_FOLDER

DEFAULT_MODELS = ["aprf", "vonmises"]
PYCORTEX_FSAVG_SUBJECT = "fsaverage"


def fsaverage_r2_path(subject: str, model: str, hemi: str,
                      bids_folder: Path) -> Path:
    return (Path(bids_folder) / "derivatives" / "encoding_models" / model
            / f"sub-{subject}" / "func"
            / f"sub-{subject}_task-abstractvalue_hemi-{hemi}_space-fsaverage_desc-r2_pe.func.gii")


def load_bilateral(subject: str, model: str, bids_folder: Path) -> np.ndarray | None:
    """Return L+R fsaverage R² concatenated as a (n_vertices,) array, or None if missing."""
    arrays = []
    for hemi in ("L", "R"):
        p = fsaverage_r2_path(subject, model, hemi, bids_folder)
        if not p.exists():
            return None
        arrays.append(load_surface_data(p))
    return np.concatenate(arrays)


def discover_subjects(bids_folder: Path, model: str = "aprf") -> list[str]:
    """Subjects with an fsaverage R² file for `model`."""
    base = Path(bids_folder) / "derivatives" / "encoding_models" / model
    out = []
    for p in sorted(base.glob("sub-*")):
        if (p / "func" / f"{p.name}_task-abstractvalue_hemi-L_space-fsaverage_desc-r2_pe.func.gii").exists():
            out.append(p.name.removeprefix("sub-"))
    return out


def soft_alpha(values: np.ndarray, thr: float, sigma: float) -> np.ndarray:
    """Gaussian-CDF centred on thr (smooth alpha) — same convention as
    visualize_subject_model.py."""
    return norm.cdf(values, loc=thr, scale=sigma).astype(np.float32)


def main(subjects: list[str], models: list[str], bids_folder: Path,
         r2_thr: float, r2_sigma: float) -> None:
    ds: dict[str, cortex.Vertex] = {}
    for model in models:
        per_sub = []
        used: list[str] = []
        for sub in subjects:
            arr = load_bilateral(sub, model, bids_folder)
            if arr is None:
                print(f"sub-{sub} {model}: fsaverage R² not found — skipping")
                continue
            per_sub.append(arr)
            used.append(sub)
        if not per_sub:
            print(f"{model}: no subjects with fsaverage R² — skipping")
            continue

        stack = np.stack(per_sub, axis=0)        # (n_subjects, n_vertices)
        mean_r2 = np.nanmean(stack, axis=0)

        pos = mean_r2[mean_r2 > 0]
        vmax = float(np.nanpercentile(pos, 99.5)) if pos.size else max(r2_thr * 2, 10.0)

        alpha = soft_alpha(mean_r2, r2_thr, r2_sigma)
        v = cortex.Vertex(np.nan_to_num(mean_r2).astype(np.float32),
                          PYCORTEX_FSAVG_SUBJECT,
                          vmin=r2_thr, vmax=vmax, cmap="hot")
        label = f"mean_{model}_r2  (n={len(used)})"
        ds[label] = v.blend_curvature(alpha)
        print(f"{model}: n={len(used)} subjects "
              f"[{', '.join(used)}], R² range [{mean_r2.min():.2f}, {mean_r2.max():.2f}]%, "
              f"colorbar [{r2_thr:.1f}, {vmax:.1f}]")

    if not ds:
        raise SystemExit("Nothing to show — run sample_r2_to_surface.py first.")

    print(f"\nLaunching pycortex viewer with {len(ds)} dataset(s)...")
    cortex.webgl.show(ds)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+",
                   help="Subject labels (default: discover from disk)")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help=f"Models to average (default: {' '.join(DEFAULT_MODELS)})")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--r2-thr", type=float, default=2.0,
                   help="R² alpha-masking threshold in PERCENT (default 2)")
    p.add_argument("--r2-sigma", type=float, default=1.0,
                   help="Gaussian-CDF transition width in PERCENT (default 1)")
    args = p.parse_args()

    subjects = (args.subjects
                if args.subjects is not None
                else discover_subjects(Path(args.bids_folder), model=args.models[0]))
    if not subjects:
        raise SystemExit("No subjects found — pass --subjects or run "
                         "sample_r2_to_surface.py first.")

    main(subjects, args.models, Path(args.bids_folder),
         args.r2_thr, args.r2_sigma)
