"""Group-mean R² / cvR² on fsaverage cortex via pycortex.

Loads the fsaverage-space (cv)R² `.func.gii` files produced by
`sample_r2_to_surface.py`, averages across subjects per model, and
displays the means in a pycortex webgl viewer on the `fsaverage` subject.

Per model, alpha-masks vertices by the **group mean R²** itself with a
smooth Gaussian-CDF transition centred on `--r2-thr`, so weak-signal
vertices fade into the curvature background.

Use `--desc cvr2` to visualise the cross-validated R² maps (from the
`*.cv` model dirs aggregated by `aggregate_cvr2.py`); the default
`--desc r2` is the full-fit R² from the non-CV dirs.

Usage:
    python -m abstract_values.visualize.visualize_mean_r2_fsaverage
    python -m abstract_values.visualize.visualize_mean_r2_fsaverage --subjects 07 08 09 10
    python -m abstract_values.visualize.visualize_mean_r2_fsaverage --r2-thr 5 --r2-sigma 2
    python -m abstract_values.visualize.visualize_mean_r2_fsaverage --models aprf vonmises aprf-session-shift
    python -m abstract_values.visualize.visualize_mean_r2_fsaverage --desc cvr2 --models aprf.cv vonmises.cv

Note: the abstract_values encoding models (fit_aprf.py, fit_vonmises.py)
write R² as a **fraction (0–1)** — `get_rsq()` output. That's a different
convention from GLMsingle (which uses percent 0–100). Thresholds here are
on the fraction scale. R² ≥ 0.05 is a reasonable starting point.
"""

from __future__ import annotations

# This module is meant to run from the `pycortex2` conda env, which is kept
# minimal (pycortex + numpy + nibabel + scipy) to avoid conflicts with
# the heavy `abstract_values` env (nilearn / TF / etc.). Keep imports
# accordingly — don't pull in nilearn-flavoured helpers from
# `abstract_values.surface.sampling`; tiny utilities are inlined below.

import argparse
from pathlib import Path

import cortex
import nibabel as nib
import numpy as np
from scipy.stats import norm

# BIDS_FOLDER is just a Path constant — abstract_values.utils.data imports
# pandas + numpy but no nilearn, so it's safe in pycortex2.
from abstract_values.utils.data import BIDS_FOLDER

# Full-fit defaults — the encoder dirs whose `desc-r2` maps anchor downstream
# visualisations. The CV counterparts (aprf.cv, vonmises.cv, ...) are
# available too; switch by passing `--desc cvr2 --models aprf.cv vonmises.cv ...`.
DEFAULT_MODELS = ["aprf", "vonmises", "aprf-weighted", "aprf-gauss"]
DEFAULT_CVR2_MODELS = ["aprf.cv", "vonmises.cv", "aprf-weighted.cv", "aprf-gauss.cv"]
PYCORTEX_FSAVG_SUBJECT = "fsaverage"


def fsaverage_r2_path(subject: str, model: str, hemi: str,
                      bids_folder: Path, desc: str = "r2") -> Path:
    return (Path(bids_folder) / "derivatives" / "encoding_models" / model
            / f"sub-{subject}" / "func"
            / f"sub-{subject}_task-abstractvalue_hemi-{hemi}_space-fsaverage_desc-{desc}_pe.func.gii")


def _load_gifti(path: Path) -> np.ndarray:
    """Load a `.func.gii` first darray as float32. Inlined to avoid pulling
    in nilearn (which would force pycortex2 env to install nilearn just to
    load surface files)."""
    return nib.load(str(path)).darrays[0].data.astype(np.float32)


def load_bilateral(subject: str, model: str, bids_folder: Path,
                   desc: str = "r2") -> np.ndarray | None:
    """Return L+R fsaverage (cv)R² concatenated as a (n_vertices,) array, or None if missing."""
    arrays = []
    for hemi in ("L", "R"):
        p = fsaverage_r2_path(subject, model, hemi, bids_folder, desc=desc)
        if not p.exists():
            return None
        arrays.append(_load_gifti(p))
    return np.concatenate(arrays)


def discover_subjects(bids_folder: Path, model: str = "aprf",
                      desc: str = "r2") -> list[str]:
    """Subjects with an fsaverage (cv)R² file for `model`."""
    base = Path(bids_folder) / "derivatives" / "encoding_models" / model
    out = []
    for p in sorted(base.glob("sub-*")):
        fn = f"{p.name}_task-abstractvalue_hemi-L_space-fsaverage_desc-{desc}_pe.func.gii"
        if (p / "func" / fn).exists():
            out.append(p.name.removeprefix("sub-"))
    return out


def soft_alpha(values: np.ndarray, thr: float, sigma: float) -> np.ndarray:
    """Gaussian-CDF centred on thr (smooth alpha) — same convention as
    visualize_subject_model.py."""
    return norm.cdf(values, loc=thr, scale=sigma).astype(np.float32)


def main(subjects: list[str], models: list[str], bids_folder: Path,
         r2_thr: float, r2_sigma: float, desc: str = "r2") -> None:
    ds: dict[str, cortex.Vertex] = {}
    for model in models:
        per_sub = []
        used: list[str] = []
        for sub in subjects:
            arr = load_bilateral(sub, model, bids_folder, desc=desc)
            if arr is None:
                print(f"sub-{sub} {model}: fsaverage {desc} not found — skipping")
                continue
            per_sub.append(arr)
            used.append(sub)
        if not per_sub:
            print(f"{model}: no subjects with fsaverage {desc} — skipping")
            continue

        stack = np.stack(per_sub, axis=0)        # (n_subjects, n_vertices)
        mean_r2 = np.nanmean(stack, axis=0)

        # vmax: 99.5th percentile of positive R², but never below the
        # threshold (matplotlib's Normalize errors out if vmin >= vmax,
        # which can happen with small n where the percentile sits below
        # the threshold).
        pos = mean_r2[mean_r2 > 0]
        vmax = float(np.nanpercentile(pos, 99.5)) if pos.size else r2_thr * 2
        vmax = max(vmax, r2_thr * 2)

        alpha = soft_alpha(mean_r2, r2_thr, r2_sigma)
        v = cortex.Vertex(np.nan_to_num(mean_r2).astype(np.float32),
                          PYCORTEX_FSAVG_SUBJECT,
                          vmin=r2_thr, vmax=vmax, cmap="hot")
        label = f"mean_{model}_{desc}  (n={len(used)})"
        ds[label] = v.blend_curvature(alpha)
        print(f"{model}: n={len(used)} subjects "
              f"[{', '.join(used)}], {desc} range [{mean_r2.min():.2f}, {mean_r2.max():.2f}], "
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
    p.add_argument("--models", nargs="+", default=None,
                   help=f"Models to average. Default depends on --desc: "
                        f"r2 -> {' '.join(DEFAULT_MODELS)}; "
                        f"cvr2 -> {' '.join(DEFAULT_CVR2_MODELS)}.")
    p.add_argument("--desc", default="r2", choices=["r2", "cvr2"],
                   help="Which fsaverage desc-entity to load: "
                        "r2 (full-fit, default) or cvr2 (cross-validated).")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--r2-thr", type=float, default=0.05,
                   help="R² alpha-masking threshold (FRACTION 0–1; "
                        "default 0.05 = 5%% variance explained)")
    p.add_argument("--r2-sigma", type=float, default=0.02,
                   help="Gaussian-CDF transition width on the fraction scale "
                        "(default 0.02)")
    args = p.parse_args()

    if args.models is None:
        args.models = DEFAULT_CVR2_MODELS if args.desc == "cvr2" else DEFAULT_MODELS

    subjects = (args.subjects
                if args.subjects is not None
                else discover_subjects(Path(args.bids_folder),
                                       model=args.models[0],
                                       desc=args.desc))
    if not subjects:
        raise SystemExit("No subjects found — pass --subjects or run "
                         "sample_r2_to_surface.py first.")

    main(subjects, args.models, Path(args.bids_folder),
         args.r2_thr, args.r2_sigma, desc=args.desc)
