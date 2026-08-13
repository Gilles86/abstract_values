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
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, Normalize
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
from abstract_values.utils.data import BIDS_FOLDER, cvr2_prevalence, DEFAULT_NULL_MODEL

# Full-fit defaults — the encoder dirs whose `desc-r2` maps anchor downstream
# visualisations. The CV counterparts (aprf.cv, vonmises.cv, ...) are
# available too; switch by passing `--desc cvr2 --models aprf.cv vonmises.cv ...`.
DEFAULT_MODELS = ["aprf", "vonmises", "aprf-weighted", "aprf-gauss"]
DEFAULT_CVR2_MODELS = ["aprf.cv", "vonmises.cv", "aprf-weighted.cv", "aprf-gauss.cv"]
PYCORTEX_FSAVG_SUBJECT = "fsaverage"


def _roi_contour(fig, roi_names, *, subject=PYCORTEX_FSAVG_SUBJECT,
                 color="cyan", linewidth=1.3):
    """Draw an ROI boundary on top of an existing quickflat figure, without
    going through pycortex's own ``with_rois=True`` (which shells out to
    Inkscape to rasterize the ROI SVG — not installed here, and broken in
    the current Homebrew cask). Instead: pull the ROI's vertex indices,
    rasterize a 0/1 membership map to the *same* flatmap projection via
    ``make_flatmap_image`` (pure numpy/pycortex, no Inkscape), and contour
    it on the figure's existing flatmap axis — same pixel grid, so no
    alignment step is needed.

    ``fsaverage``'s pycortex ROI database already has NPC_L/NPC_R (and
    NPC1-3, NF1-2, NTO, NINS) from prior numerical-cognition work — the
    same NPC naming this project's volumetric masks use.
    """
    roi_verts = cortex.get_roi_verts(subject, roi=list(roi_names))
    # fsaverage's full bilateral vertex count (163842/hemi); using the
    # canonical count rather than max-referenced-index avoids silently
    # truncating the flatmap projection if an ROI happens to miss the
    # last few vertices.
    n_verts = cortex.db.get_surfinfo(subject).data.shape[0]
    mask = np.zeros(n_verts, dtype=np.float32)
    for idx in roi_verts.values():
        mask[idx] = 1.0
    mask = np.zeros(n_verts, dtype=np.float32)
    for idx in roi_verts.values():
        mask[idx] = 1.0
    vtx_mask = cortex.Vertex(mask, subject, vmin=0, vmax=1)
    im, extents = cortex.quickflat.make_flatmap_image(vtx_mask, height=1024)
    ax = fig.axes[0]
    ax.contour(np.nan_to_num(im), levels=[0.5], extent=extents,
              colors=color, linewidths=linewidth, origin="upper")


def _save_png(vtx, out, *, label, cmap, vmin, vmax, cbar_label="R²",
              roi_overlay=None):
    """Render the flatmap to a static PNG with its own colorbar (pycortex's
    ``with_colorbar=True`` is meaningless here — ``blend_curvature`` returns
    plain RGB with no scalar cmap/vmin/vmax) and the dataset label — which
    already carries n= — burned into the image itself as a title, so the
    subject count survives outside the interactive-only webgl viewer.

    ``roi_overlay``: optional list of pycortex ROI names (e.g. ``["NPC_R"]``)
    to outline — see :func:`_roi_contour`. A cyan outline locating rIPS/NPCr
    is far more informative than chasing a threshold that makes it "light
    up": aPRF whole-cortex R² is dominated by V1's stimulus-driven response,
    so NPCr rarely wins the colorbar regardless of threshold.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = cortex.quickflat.make_figure(vtx, with_curvature=True,
                                       with_colorbar=False, with_rois=False,
                                       with_labels=False)
    if roi_overlay:
        _roi_contour(fig, roi_overlay)
    fig.suptitle(label, fontsize=8.5, color="0.15", y=0.99)
    cax = fig.add_axes([0.36, 0.06, 0.28, 0.020])
    cb = ColorbarBase(cax, cmap=plt.get_cmap(cmap),
                      norm=Normalize(vmin=vmin, vmax=vmax),
                      orientation="horizontal")
    cb.set_label(cbar_label, fontsize=9, labelpad=3)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=8, width=0.5, length=2.5)
    fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved flatmap PNG → {out}")


def _static_png_path(static_png, n_combos, model, full_desc):
    """A single explicit --static-png path is used as-is only when there's
    exactly one (model, smoothing) combination being rendered; otherwise
    --static-png is treated as a directory and each combo gets its own
    auto-named file inside it."""
    p = Path(static_png)
    if n_combos == 1:
        return p
    p.mkdir(parents=True, exist_ok=True)
    return p / f"mean_{model}_{full_desc}_fsaverage.png"


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


def _empirical_null_threshold(r2: np.ndarray, alpha: float
                                ) -> tuple[float, float, float]:
    """Efron-style empirical-null threshold. Fits ONE Gaussian to the
    bulk of logit(R²) via the median and IQR (robust to the signal
    upper tail), then takes the (1−α) quantile of that null as the
    threshold. Returns ``(thr_r2, mu_logit, sigma_logit)``.

    Why this and not a 2-component mixture: averaging R² across
    subjects compresses the per-subject noise/signal bimodality into
    a near-unimodal distribution, so the 2-component GMM either lumps
    everything into one mode (`signal_weight≈1`) or splits the body
    arbitrarily — see :func:`_mixture_is_degenerate`. A single robust
    Gaussian on the bulk is the cleaner model: it doesn't pretend
    there are two modes, and the IQR-based scale is insensitive to
    the upper signal tail.

    Why this and not just a percentile of R²: a percentile always
    marks exactly α fraction of vertices regardless of whether the
    map contains signal. The EN threshold is pegged to the *noise
    scale*, so the fraction that actually survives carries
    information — `frac_surviving / α` is an enrichment ratio. A
    well-tuned map has frac>>α (real signal in the tail); a
    noise-only map has frac≈α."""
    r2 = np.asarray(r2, dtype=np.float64).ravel()
    r2 = r2[np.isfinite(r2)]
    r2 = r2[(r2 > 1e-6) & (r2 < 1 - 1e-6)]
    if len(r2) < 1000:
        return float("nan"), float("nan"), float("nan")
    z = np.log(r2 / (1.0 - r2))
    q25, q50, q75 = np.percentile(z, [25, 50, 75])
    mu0 = q50
    sigma0 = (q75 - q25) / 1.349                  # IQR → σ of a Gaussian
    z_thr = norm.ppf(1.0 - alpha, loc=mu0, scale=sigma0)
    return float(1.0 / (1.0 + np.exp(-z_thr))), float(mu0), float(sigma0)


def _mixture_is_degenerate(fit: dict,
                             max_signal_weight: float = 0.85,
                             min_noise_weight: float = 0.05,
                             min_logit_separation: float = 0.5,
                             ) -> tuple[bool, str]:
    """Detect a 2-component R² mixture that didn't actually find a
    noise/signal split. Three failure modes seen empirically on
    abstract_values fsaverage R² maps:

      - **One component dominates** (`signal_weight > 0.85` or
        equivalently `noise_weight < 0.05`): the GMM collapsed onto a
        single mode and called it "signal". The "noise" component is
        fitting either a thin tail of the same distribution or a
        handful of extreme vertices — the resulting FDR threshold then
        either falls to ≈0 (everything passes) or jumps to ∞ (nothing
        passes), depending on which side of the dominant mode the
        spurious component landed.
      - **Modes too close to separate** (logit-space separation <0.5,
        ≈ R² difference of <0.01 around R²=0.03): the components
        overlap so heavily that the noise tail outweighs the signal
        tail at every threshold, also collapsing FDR to ≈0.

    Returning `(True, reason)` lets the caller fall back to a
    percentile or a fixed threshold and surface a clear warning.
    Thresholds picked from the observed-good fits on this project's
    maps (noise_weight ~0.8, separation ~1.5 logit units between
    noise mode R²~0.01 and signal mode R²~0.04)."""
    if fit['signal_weight'] > max_signal_weight:
        return True, (f"signal weight {fit['signal_weight']:.2f} "
                      f"> {max_signal_weight} — mixture lumped almost all "
                      f"vertices into the signal component")
    if fit['noise_weight'] < min_noise_weight:
        return True, (f"noise weight {fit['noise_weight']:.2f} "
                      f"< {min_noise_weight} — noise component collapsed")
    sep = fit['signal_mu'] - fit['noise_mu']
    if sep < min_logit_separation:
        return True, (f"mode separation {sep:.2f} (logit) "
                      f"< {min_logit_separation} — components overlap")
    return False, ""


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


def _prevalence_cmap(name: str = "prevalence_warm") -> str:
    """Sequential ramp for the 0–1 prevalence (proportion-of-subjects) map.
    Reuses the same magma-derived warm palette as the R² map so the two
    visualisations read consistently."""
    return _register_r2_cmap(name)


PREVALENCE_CMAP = _prevalence_cmap()


def main_prevalence(subjects: list[str], models: list[str], bids_folder: Path,
                    cv_thr: float, min_prevalence: float,
                    baseline_model: str | None = DEFAULT_NULL_MODEL,
                    smoothing: tuple[str, ...] = ("", "_smoothed"),
                    sigma: float = 0.08,
                    static_png: str | None = None) -> None:
    """Cross-model-comparable consistency map: per vertex, the *proportion of
    subjects* where the model's cross-validated R² beats the per-vertex null.

    Rationale
    ---------
    The per-voxel signal test is ``cvR² > cvR²_null`` (the null-null model
    predicts the training-set mean), NOT ``cvR² > 0``: a silent voxel scores a
    slightly negative cvR² on held-out data (train/test mean mismatch), so the
    proper baseline is negative and ``> 0`` discards real-but-modest voxels
    (empirically ~11× fewer survive). cvR² is parameter-count-fair, so the same
    criterion is comparable across models. Binarising per subject before pooling
    discards the very-negative noise-voxel magnitudes (cvR² reaches ≈ −0.5) that
    would otherwise dominate a cross-subject average.

    The actual per-subject comparison lives in
    :func:`abstract_values.utils.data.cvr2_prevalence` so other scripts can
    reuse it. Here we just colour the resulting proportion (0–1) and fade alpha
    in at ``min_prevalence``. Because the base rate of "wins" is small, a
    binomial prevalence test is hyper-sensitive; we threshold on a *fixed
    prevalence* and print the binomial p at that count as a sanity check only.
    """
    from scipy.stats import binom

    ds: dict[str, cortex.Vertex] = {}
    for model in models:
        for smooth in smoothing:
            smoothed = (smooth == "_smoothed")
            res = cvr2_prevalence(subjects, model, baseline_model=baseline_model,
                                  cv_thr=cv_thr, smoothed=smoothed,
                                  bids_folder=bids_folder)
            if res is None:
                print(f"{model}{smooth}: no subjects with fsaverage cvr2 — skipping")
                continue

            prop, n, p0 = res['prop'], res['n'], res['p0']
            k_thr = int(np.ceil(min_prevalence * n))
            p_binom = float(binom.sf(k_thr - 1, n, p0)) if n else float("nan")
            alpha = soft_alpha(prop, min_prevalence, sigma)
            frac_surv = float((prop >= min_prevalence).mean()) * 100
            ref = baseline_model if baseline_model else f"{cv_thr:g}"

            v = cortex.Vertex(np.nan_to_num(prop).astype(np.float32),
                              PYCORTEX_FSAVG_SUBJECT,
                              vmin=min_prevalence, vmax=1.0, cmap=PREVALENCE_CMAP)
            label = (f"prev_{model}{smooth}  (n={n}, cvR² > {ref}; "
                     f"≥{min_prevalence:.0%}={k_thr}/{n} subj)")
            vtx = v.blend_curvature(alpha)
            ds[label] = vtx
            print(f"{model}{smooth}: n={n} [{', '.join(res['subjects'])}], "
                  f"baseline={ref}, base rate p0={p0:.4f}, "
                  f"max prevalence={prop.max():.2f}, "
                  f"surviving ≥{min_prevalence:.0%}: {frac_surv:.2f}% of vertices "
                  f"(binomial P(X≥{k_thr}|n={n},p0)={p_binom:.1e})")
            if static_png:
                n_combos = len(models) * len(smoothing)
                out_path = _static_png_path(static_png, n_combos, model,
                                            f"cvr2{smooth}")
                _save_png(vtx, out_path, label=label, cmap=PREVALENCE_CMAP,
                          vmin=min_prevalence, vmax=1.0, cbar_label="Prevalence")

    if not ds:
        raise SystemExit("Nothing to show — need fsaverage cvr2 files "
                         "(run aggregate_cvr2.py + sample_r2_to_surface.py --desc cvr2).")

    if not static_png:
        print(f"\nLaunching pycortex viewer with {len(ds)} dataset(s)...")
        cortex.webgl.show(ds)


def main(subjects: list[str], models: list[str], bids_folder: Path,
         r2_thr: float, r2_sigma: float, desc: str = "r2",
         smoothing: tuple[str, ...] = ("", "_smoothed"),
         fdr_alpha: float | None = None,
         fdr_mode: str = "empirical_null",
         static_png: str | None = None,
         roi_overlay: list[str] | None = None) -> None:
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
            elif fdr_mode == "empirical_null":
                thr, mu0, sigma0 = _empirical_null_threshold(mean_r2, fdr_alpha)
                cohort_thr = max(thr, r2_thr)
                alpha = soft_alpha(mean_r2, cohort_thr, r2_sigma)
                frac_surv = float((mean_r2 > cohort_thr).mean()) * 100
                thr_note = (f"  EN-null α={fdr_alpha:.3f} → thr={cohort_thr:.3f}  "
                            f"(null μ={1/(1+np.exp(-mu0)):.3f}, "
                            f"σ_logit={sigma0:.2f}; surviving {frac_surv:.2f}% "
                            f"≈ {frac_surv/(fdr_alpha*100):.1f}× chance)")
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
                # Percentile fallback: top fdr_alpha fraction of vertices.
                # Tracks the alpha intent (α=0.05 → top 5%) and prevents
                # the alpha mask from collapsing to "everything" or
                # "nothing" when the mixture is unreliable. Bounded below
                # by --r2-thr so it never goes below the explicit floor.
                pct_thr = float(np.nanpercentile(
                    mean_r2, 100.0 * (1.0 - fdr_alpha)))
                pct_thr = max(pct_thr, r2_thr)
                if fit is None:
                    print(f"  warning: group mixture fit failed "
                          f"(n_vertices={int(np.isfinite(mean_r2).sum())}); "
                          f"falling back to top-{fdr_alpha*100:.1f}% "
                          f"percentile (thr={pct_thr:.3f})")
                    cohort_thr = pct_thr
                    thr_note = (f"  mixture fit failed → top "
                                f"{fdr_alpha*100:.1f}% percentile fallback "
                                f"(thr={cohort_thr:.3f})")
                else:
                    degen, reason = _mixture_is_degenerate(fit)
                    if degen:
                        print(f"  warning: group mixture degenerate "
                              f"({reason}); falling back to top-"
                              f"{fdr_alpha*100:.1f}% percentile "
                              f"(thr={pct_thr:.3f})")
                        cohort_thr = pct_thr
                        thr_note = (f"  mixture degenerate ({reason}) → "
                                    f"top {fdr_alpha*100:.1f}% percentile "
                                    f"(thr={cohort_thr:.3f})")
                    else:
                        cohort_thr = _fdr_threshold_from_mixture(fit, fdr_alpha)
                        if not np.isfinite(cohort_thr) or cohort_thr <= 0:
                            print(f"  warning: FDR α={fdr_alpha} unreachable "
                                  f"(thr={cohort_thr}); falling back to top-"
                                  f"{fdr_alpha*100:.1f}% percentile "
                                  f"(thr={pct_thr:.3f})")
                            cohort_thr = pct_thr
                            thr_note = (f"  FDR α unreachable → top "
                                        f"{fdr_alpha*100:.1f}% percentile "
                                        f"(thr={cohort_thr:.3f})")
                        else:
                            thr_note = (f"  FDR α={fdr_alpha:.3f} on group R² → "
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
            vtx = v.blend_curvature(alpha)
            ds[label] = vtx
            print(f"{model} {full_desc}: n={len(used)} "
                  f"[{', '.join(used)}], range [{mean_r2.min():.3f}, {mean_r2.max():.3f}], "
                  f"colorbar [{cohort_thr:.3f}, {vmax:.3f}]{thr_note}")
            if static_png:
                n_combos = len(models) * len(smoothing)
                out_path = _static_png_path(static_png, n_combos, model, full_desc)
                _save_png(vtx, out_path, label=label,
                          cmap=R2_CMAP, vmin=cohort_thr, vmax=vmax,
                          roi_overlay=roi_overlay)

    if not ds:
        raise SystemExit("Nothing to show — run sample_r2_to_surface.py first.")

    if not static_png:
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
    p.add_argument("--fdr-mode", default="empirical_null",
                   choices=["empirical_null", "group", "per_subject"],
                   help="'empirical_null' (default): fit ONE robust "
                        "Gaussian to the bulk of logit(group-mean R²) via "
                        "median/IQR, threshold at the (1−α) tail of that "
                        "null. Efron-style local FDR; never degenerate. "
                        "'group': legacy 2-component mixture on group-mean "
                        "R² with percentile fallback when degenerate (it "
                        "usually is — the cross-subject mean compresses "
                        "the per-subject bimodality). 'per_subject': look "
                        "up each subject's cached whole-brain p_signal.json "
                        "and average per-subject alpha masks.")
    p.add_argument("--smoothing", nargs="+", default=["", "_smoothed"],
                   choices=["", "_smoothed"],
                   help="Which BOLD-smoothing variants to include "
                        "(default: both unsmoothed and smoothed). "
                        "Pass `--smoothing ''` for unsmoothed only or "
                        "`--smoothing _smoothed` for smoothed only.")
    p.add_argument("--agg", default="mean", choices=["mean", "prevalence"],
                   help="How to combine subjects. 'mean' (default): "
                        "group-mean R² map with an FDR/empirical-null alpha "
                        "mask. 'prevalence': cross-model-comparable "
                        "consistency map — per vertex, the proportion of "
                        "subjects with cvR² > --cv-thr. Forces --desc cvr2 "
                        "(cvR²>0 is the parameter-count-fair, zero-anchored "
                        "criterion); binarising per subject discards the "
                        "very-negative noise voxels that corrupt a mean.")
    p.add_argument("--cv-thr", type=float, default=0.0,
                   help="Per-subject cvR² positivity threshold for "
                        "--agg prevalence (default 0.0 = held-out fit beats "
                        "the mean).")
    p.add_argument("--min-prevalence", type=float, default=0.5,
                   help="Display/alpha threshold for --agg prevalence: "
                        "fraction of subjects that must be positive "
                        "(default 0.5 = majority).")
    p.add_argument("--baseline-model", default=DEFAULT_NULL_MODEL,
                   help="Per-vertex cvR² null reference for --agg prevalence "
                        f"(default {DEFAULT_NULL_MODEL!r}, the 'predict "
                        "training mean' model). A voxel is a 'win' where the "
                        "model's cvR² exceeds this model's cvR² at the same "
                        "vertex. Pass 'none' to compare against the scalar "
                        "--cv-thr (i.e. cvR² > 0) instead.")
    p.add_argument("--static-png", default=None,
                   help="If set, save static flatmap PNG(s) (n= burned into "
                        "the title) instead of launching the webgl viewer. "
                        "A single file path when exactly one (model, "
                        "smoothing) combination is requested; otherwise "
                        "treated as an output directory.")
    p.add_argument("--roi-overlay", nargs="*", default=None,
                   help="pycortex ROI name(s) to outline on --static-png "
                        "output (e.g. --roi-overlay NPC_R NPC_L). aPRF "
                        "whole-cortex R² is dominated by V1, so NPCr/IPS "
                        "rarely 'lights up' regardless of threshold — an "
                        "outline locates it independent of color. No "
                        "argument (bare --roi-overlay) draws none; omitting "
                        "the flag entirely also draws none.")
    args = p.parse_args()

    if args.agg == "prevalence" and args.desc != "cvr2":
        print("note: --agg prevalence requires cross-validated R²; "
              "forcing --desc cvr2")
        args.desc = "cvr2"

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

    if args.agg == "prevalence":
        baseline = (None if str(args.baseline_model).lower() == "none"
                    else args.baseline_model)
        main_prevalence(subjects, args.models, Path(args.bids_folder),
                        cv_thr=args.cv_thr, min_prevalence=args.min_prevalence,
                        baseline_model=baseline, smoothing=tuple(args.smoothing),
                        static_png=args.static_png)
    else:
        # --fdr-alpha 0 (or negative) is the explicit "disable" sentinel.
        fdr_alpha = args.fdr_alpha if (args.fdr_alpha and args.fdr_alpha > 0) else None
        main(subjects, args.models, Path(args.bids_folder),
             args.r2_thr, args.r2_sigma, desc=args.desc,
             smoothing=tuple(args.smoothing),
             fdr_alpha=fdr_alpha, fdr_mode=args.fdr_mode,
             static_png=args.static_png, roi_overlay=args.roi_overlay)
