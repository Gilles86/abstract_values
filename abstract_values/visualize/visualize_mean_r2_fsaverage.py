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
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm

# Register a pure-orange ramp with pycortex once at import time. Pycortex
# resolves colormap names against the matplotlib registry, so plt.register
# is enough — no separate cortex.utils calls needed.
def _register_r2_cmap(name: str = "r2_warm") -> str:
    """Perceptually-uniform warm ramp for R² on the fsaverage surface.

    Builds a ``magma``-derived palette truncated to skip the near-black low
    end (which read as muddy against curvature) and the near-white high end
    (which read as washed out). Result: deep purple → red → orange → yellow,
    perceptually uniform, prints / projects well, plays nicely with
    `blend_curvature`. Falls back gracefully on older matplotlib registries.
    """
    import numpy as _np
    import matplotlib.pyplot as _plt
    try:
        _plt.get_cmap(name)
        return name
    except Exception:
        pass
    base = _plt.get_cmap("magma")
    colors = base(_np.linspace(0.22, 0.92, 256))
    cmap = LinearSegmentedColormap.from_list(name, colors)
    try:
        _plt.colormaps.register(cmap, name=name)
    except AttributeError:                       # matplotlib < 3.9 fallback
        import matplotlib as _mpl
        _mpl.cm.register_cmap(name=name, cmap=cmap)
    return name


R2_CMAP = _register_r2_cmap()
ORANGE_CMAP = R2_CMAP                            # backward-compat alias

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


def _fdr_threshold_from_mixture(fit: dict, alpha: float,
                                  n_grid: int = 4000) -> float:
    """R² threshold at which the 2-component logit-Gaussian mixture's
    tail-FDR is ≤ α. Mirrors :func:`braincoder.utils.stats.r2_fdr_threshold`
    but inlined so this script runs in the pycortex2 env without
    pulling in braincoder.
    """
    z = np.linspace(fit['noise_mu'] - 5 * fit['noise_sigma'],
                    fit['signal_mu'] + 8 * fit['signal_sigma'],
                    n_grid)
    sf_n = 1.0 - norm.cdf(z, fit['noise_mu'],  fit['noise_sigma'])
    sf_s = 1.0 - norm.cdf(z, fit['signal_mu'], fit['signal_sigma'])
    denom = fit['noise_weight'] * sf_n + fit['signal_weight'] * sf_s
    fdr = np.where(denom > 1e-12,
                   fit['noise_weight'] * sf_n / np.maximum(denom, 1e-12),
                   1.0)
    hits = np.where(fdr <= alpha)[0]
    if len(hits) == 0:
        return float("inf")
    z_thr = z[hits[0]]
    return float(1.0 / (1.0 + np.exp(-z_thr)))      # inverse logit → R²


def _fit_r2_mixture_inline(r2: np.ndarray, n_init: int = 8,
                            max_iter: int = 500, seed: int = 0) -> dict | None:
    """Fit a 2-component Gaussian mixture on logit(R²). Returns the same
    dict shape as the per-subject p_signal.json (``noise_mu``, ``noise_sigma``,
    ``signal_mu``, ``signal_sigma``, ``noise_weight``, ``signal_weight``)
    suitable to feed straight into :func:`_fdr_threshold_from_mixture`.

    Mirrors ``braincoder.utils.stats.fit_r2_mixture`` but uses sklearn's
    GaussianMixture so it runs in pycortex2 (no braincoder dependency).
    Filters to finite R² in (0, 1) before logit.
    """
    from sklearn.mixture import GaussianMixture
    r2 = np.asarray(r2, dtype=np.float64).ravel()
    r2 = r2[np.isfinite(r2)]
    r2 = r2[(r2 > 1e-6) & (r2 < 1 - 1e-6)]
    if len(r2) < 1000:
        return None      # not enough vertices to trust the fit
    z = np.log(r2 / (1.0 - r2)).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, n_init=n_init,
                          max_iter=max_iter, random_state=seed)
    gmm.fit(z)
    means = gmm.means_.ravel()
    sigmas = np.sqrt(gmm.covariances_.ravel())
    weights = gmm.weights_
    # noise = lower-mean component; signal = higher-mean component
    n_idx = int(np.argmin(means))
    s_idx = 1 - n_idx
    return {
        "noise_mu":      float(means[n_idx]),
        "noise_sigma":   float(sigmas[n_idx]),
        "noise_weight":  float(weights[n_idx]),
        "signal_mu":     float(means[s_idx]),
        "signal_sigma":  float(sigmas[s_idx]),
        "signal_weight": float(weights[s_idx]),
        "n_vertices":    int(len(r2)),
    }


def _per_subject_fdr_threshold(subject: str, model: str, bids_folder: Path,
                                alpha: float, smoothed: bool):
    """Look up the per-subject whole-brain FDR-α R² threshold from the
    cached p_signal.json written by
    ``abstract_values.encoding_models.compute_r2_mixture``.

    Returns the threshold as a fraction (0–1), or ``None`` if the
    mixture file is missing or flagged degenerate. The .cv encoder dirs
    share the mixture with the non-CV encoder (the mixture is fit on
    cv-R²), so ``aprf.cv`` falls back to ``aprf`` when its own dir lacks
    a p_signal.json.

    No braincoder / abstract_values imports — works in the pycortex2
    env directly.
    """
    base_model = model.split(".")[0]
    p = (Path(bids_folder) / "derivatives" / "encoding_models"
         / base_model / f"sub-{subject}"
         / f"sub-{subject}_desc-p_signal.json")
    if not p.exists():
        return None
    import json
    info = json.loads(p.read_text())
    brain = info.get("BRAIN", info)        # backward-compat
    if brain.get("degenerate"):
        return None
    # Pre-tabulated keys: fdr_05, fdr_01, etc.
    fdr_key = f"fdr_{int(round(alpha * 100)):02d}"
    if fdr_key in brain:
        return float(brain[fdr_key])
    # Otherwise re-derive from the cached mixture parameters.
    needed = ("noise_mu", "noise_sigma", "signal_mu", "signal_sigma",
              "noise_weight", "signal_weight")
    if not all(k in brain for k in needed):
        return None
    thr = _fdr_threshold_from_mixture(brain, alpha)
    return None if not np.isfinite(thr) else float(thr)


def main(subjects: list[str], models: list[str], bids_folder: Path,
         r2_thr: float, r2_sigma: float, desc: str = "r2",
         smoothing: tuple[str, ...] = ("", "_smoothed"),
         fdr_alpha: float | None = None,
         fdr_mode: str = "group") -> None:
    """Build one pycortex dataset per (model, smoothing) combination present
    on disk. By default both unsmoothed and smoothed variants are shown side
    by side; the smoothed variant is loaded from `desc-<desc>_smoothed`.

    ``fdr_mode``:
      - ``"group"`` (default): fit a 2-component logit-Gaussian mixture on
        the **group-mean R²** vertex distribution and use the resulting
        FDR-α threshold for the (single) alpha mask. The right object for
        what's actually being displayed.
      - ``"per_subject"``: legacy behaviour — look up each subject's own
        whole-brain mixture threshold from the cached p_signal.json, apply
        per-subject alpha masks, then average. Subjects with a degenerate /
        missing mixture fall back to ``r2_thr``.
    """
    ds: dict[str, cortex.Vertex] = {}
    for model in models:
        for smooth in smoothing:
            smoothed_bool = (smooth == "_smoothed")
            full_desc = f"{desc}{smooth}"
            per_sub = []          # raw fsaverage R² stacks (one row per subject)
            per_sub_thr = []      # per-subject FDR threshold (None if missing)
            used: list[str] = []
            for sub in subjects:
                arr = load_bilateral(sub, model, bids_folder, desc=full_desc)
                if arr is None:
                    continue
                per_sub.append(arr)
                used.append(sub)
                if fdr_alpha is not None and fdr_mode == "per_subject":
                    thr = _per_subject_fdr_threshold(
                        sub, model, bids_folder, fdr_alpha, smoothed_bool)
                    per_sub_thr.append(thr if thr is not None else r2_thr)
                else:
                    per_sub_thr.append(r2_thr)
            if not per_sub:
                print(f"{model} {full_desc}: no subjects with fsaverage data — skipping")
                continue

            stack = np.stack(per_sub, axis=0)        # (n_subjects, n_vertices)
            mean_r2 = np.nanmean(stack, axis=0)

            thr_note = ""
            if fdr_alpha is None:
                alpha = soft_alpha(mean_r2, r2_thr, r2_sigma)
                cohort_thr = r2_thr
            elif fdr_mode == "per_subject":
                # Each subject's threshold applied to its own map first.
                alpha_per_sub = [soft_alpha(arr, thr, r2_sigma)
                                  for arr, thr in zip(per_sub, per_sub_thr)]
                alpha = np.nanmean(alpha_per_sub, axis=0)
                cohort_thr = float(np.mean(per_sub_thr))
                spread = (f"{min(per_sub_thr):.3f}–{max(per_sub_thr):.3f}"
                          if per_sub_thr else "n/a")
                thr_note = f"  FDR α={fdr_alpha:.2f} per-subj ∈ [{spread}]"
            else:                                # group-mean mixture
                fit = _fit_r2_mixture_inline(mean_r2)
                if fit is None:
                    print(f"  warning: group mixture fit failed "
                          f"(n_vertices={int(np.isfinite(mean_r2).sum())}); "
                          f"falling back to fixed --r2-thr")
                    cohort_thr = r2_thr
                    thr_note = "  (mixture fit failed → r2_thr fallback)"
                else:
                    cohort_thr = _fdr_threshold_from_mixture(fit, fdr_alpha)
                    if not np.isfinite(cohort_thr):
                        print(f"  warning: no R² achieves FDR α={fdr_alpha}; "
                              f"falling back to fixed --r2-thr")
                        cohort_thr = r2_thr
                        thr_note = "  (FDR α unreachable → r2_thr fallback)"
                    else:
                        thr_note = (f"  FDR α={fdr_alpha:.2f} on group R² → "
                                    f"thr={cohort_thr:.3f}  "
                                    f"(noise μ={1/(1+np.exp(-fit['noise_mu'])):.3f}, "
                                    f"signal μ={1/(1+np.exp(-fit['signal_mu'])):.3f}, "
                                    f"w_signal={fit['signal_weight']:.2f})")
                alpha = soft_alpha(mean_r2, cohort_thr, r2_sigma)

            # vmax: 99.5th percentile of positive R², but never below the
            # threshold (matplotlib's Normalize errors out if vmin >= vmax).
            pos = mean_r2[mean_r2 > 0]
            vmax = float(np.nanpercentile(pos, 99.5)) if pos.size else cohort_thr * 2
            vmax = max(vmax, cohort_thr * 2)

            v = cortex.Vertex(np.nan_to_num(mean_r2).astype(np.float32),
                              PYCORTEX_FSAVG_SUBJECT,
                              vmin=cohort_thr, vmax=vmax, cmap=R2_CMAP)
            label = f"mean_{model}_{full_desc}  (n={len(used)}){thr_note}"
            ds[label] = v.blend_curvature(alpha)
            print(f"{model} {full_desc}: n={len(used)} "
                  f"[{', '.join(used)}], range [{mean_r2.min():.3f}, {mean_r2.max():.3f}], "
                  f"colorbar [{cohort_thr:.3f}, {vmax:.3f}]{thr_note}")

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
    p.add_argument("--r2-sigma", type=float, default=0.005,
                   help="Gaussian-CDF transition width on the fraction scale "
                        "(default 0.005 — quarter of the old 0.02, gives a "
                        "much steeper transition into the colored region; "
                        "vertices with R² ~thr ± 0.01 sweep from ≈0 to ≈1 "
                        "alpha. Pass a larger value for a softer fade.)")
    p.add_argument("--fdr-alpha", type=float, default=0.01,
                   help="FDR-α target on the R² mixture for the alpha mask. "
                        "Default 0.01. Smoothed maps need a stricter α than "
                        "unsmoothed because smoothing inflates the signal "
                        "mixture component — α=0.05 can leave the threshold "
                        "below the noise mode (all vertices survive). Pass "
                        "--fdr-alpha 0 to disable (use the fixed --r2-thr "
                        "cohort-wide).")
    p.add_argument("--fdr-mode", default="group",
                   choices=["group", "per_subject"],
                   help="'group' (default): fit the 2-component R² mixture "
                        "on the group-mean fsaverage R² (the actual map "
                        "being displayed). 'per_subject': legacy path that "
                        "looks up each subject's cached whole-brain "
                        "p_signal.json and averages per-subject alpha masks.")
    p.add_argument("--smoothing", nargs="+", default=["", "_smoothed"],
                   choices=["", "_smoothed"],
                   help="Which BOLD-smoothing variants to include "
                        "(default: both unsmoothed and smoothed). "
                        "Pass `--smoothing ''` for unsmoothed only or "
                        "`--smoothing _smoothed` for smoothed only.")
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

    # --fdr-alpha 0 (or negative) is the explicit "disable" sentinel.
    fdr_alpha = args.fdr_alpha if (args.fdr_alpha and args.fdr_alpha > 0) else None
    main(subjects, args.models, Path(args.bids_folder),
         args.r2_thr, args.r2_sigma, desc=args.desc,
         smoothing=tuple(args.smoothing),
         fdr_alpha=fdr_alpha, fdr_mode=args.fdr_mode)
