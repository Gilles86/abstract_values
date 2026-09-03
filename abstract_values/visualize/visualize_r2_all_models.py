#!/usr/bin/env python3
"""Combined group-mean R² / cvR² pycortex viewer across all encoding models.

One persistent webgl viewer with a dataset per (model family, r2/cvr2)
combination, so they can be flipped between live in the pycortex mixer
instead of one static PNG or one webgl session per model.

The families are the architecture x space grid of
``notes/model_comparison_plan.md``: linear / one bell / basis set, crossed
with value (CHF) and orientation (deg) space, all with tuning fixed across
sessions. That C = fixed row is the identifying comparison — once tuning may
shift per session, each space can simply re-fit and the space contrast
collapses — so the two shifted families are listed last and read as
flexibility, not as a space effect. There is no linear-orientation cell:
orientation is circular, so a monotonic ramp has no honest analogue (the
minimal model would be a single circular harmonic, k=2 von Mises basis).

Reuses `visualize_mean_r2_fsaverage.py`'s own per-metric logic rather than
reimplementing it — monkeypatching `cortex.webgl.show` to collect each
call's dataset instead of launching a viewer, then launching once at the
end with every dataset combined. See the pycortex skill's
`persistent_webshow.py` template for the pattern this follows
(daemon-thread-dies / hostname-vs-localhost gotchas).

Full-fit R² and cvR² are NOT interchangeable through the same thresholding
code (see the pycortex skill's "don't feed cvR²-shaped data through
R²-shaped auto-threshold code" section): cvR² is legitimately negative for
weak voxels, so `main()`'s empirical-null-on-the-group-mean alpha mask
degenerates to NaN once cvR² dominates the negative range. So this script
routes on desc:

  - r2   -> `vr2.main()`            group-mean + empirical-null alpha mask
  - cvr2 -> `vr2.main_prevalence()`  (default, `--cvr2-agg prevalence`)
            per vertex, proportion of subjects where cvR² beats the
            per-vertex null-model cvR² — the pycortex skill's own
            recommendation for cvR² (binarising per subject before pooling
            avoids the very-negative noise vertices dominating a raw mean).
            `--cvr2-agg ttest` switches to `vr2.main_ttest()` (paired,
            BH-FDR-corrected) instead — more rigorous but, at the default
            whole-cortex correction, conservative enough to render blank
            on this dataset's current signal levels.

Usage
-----
    conda activate pycortex2
    python -m abstract_values.visualize.visualize_r2_all_models
    python -m abstract_values.visualize.visualize_r2_all_models --smoothing "" _smoothed
    python -m abstract_values.visualize.visualize_r2_all_models \
        --families value-bell ori-bell
    python -m abstract_values.visualize.visualize_r2_all_models --subjects 07 08 09 10
"""

from __future__ import annotations

# Kept import-light (pycortex2 env) — see visualize_mean_r2_fsaverage.py's
# own note; only pull in that module and cortex itself.

import argparse
import subprocess
import time
from pathlib import Path

import cortex

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize import visualize_mean_r2_fsaverage as vr2

DEFAULT_NULL_MODEL = vr2.DEFAULT_NULL_MODEL

# (family label, full-fit model dir, cross-validated model dir). A family
# whose full-fit R2 was never sampled to the surface (the weighted bases, the
# single von Mises) simply skips that half — build_datasets() reports it.
MODEL_FAMILIES = [
    ("value-linear", "aprf-linear", "aprf-linear.cv"),
    ("value-bell", "aprf", "aprf.cv"),
    ("value-basis", "aprf-weighted", "aprf-weighted.cv"),
    ("ori-bell", "vonmises-prf", "vonmises-prf.cv"),
    ("ori-basis", "vonmises", "vonmises.cv"),
    # flexibility (C = free per session), not part of the space contrast
    ("value-bell-shift", "aprf-fwhm-shift", "aprf-fwhm-shift.cv"),
    ("ori-basis-shift", "vonmises-shift", "vonmises-shift.cv"),
]
FAMILY_NAMES = [f[0] for f in MODEL_FAMILIES]


def build_datasets(bids_folder: Path, families: list[tuple[str, str, str]],
                   subjects: list[str] | None, r2_thr: float, r2_sigma: float,
                   fdr_alpha: float | None, fdr_mode: str,
                   cvr2_agg: str, cv_thr: float, min_prevalence: float,
                   fdr_q: float, baseline_model: str,
                   smoothing: tuple[str, ...]) -> dict[str, cortex.Vertex]:
    """Run vr2.main() (r2) + vr2.main_prevalence()/vr2.main_ttest() (cvr2)
    once per family, capturing each call's dataset instead of letting it
    launch its own viewer."""
    merged: dict[str, cortex.Vertex] = {}
    captured: dict[str, cortex.Vertex] = {}

    def fake_show(data, **kwargs):
        captured.update(data)

    real_show = cortex.webgl.show
    cortex.webgl.show = fake_show
    try:
        for label, r2_model, cv_model in families:
            r2_subs = subjects if subjects is not None else vr2.discover_subjects(
                bids_folder, model=r2_model, desc="r2")
            if not r2_subs:
                print(f"{label} r2 ({r2_model}): no subjects on disk — skipping")
            else:
                captured.clear()
                try:
                    vr2.main(r2_subs, [r2_model], bids_folder, r2_thr, r2_sigma,
                            desc="r2", smoothing=smoothing,
                            fdr_alpha=fdr_alpha, fdr_mode=fdr_mode)
                except SystemExit as e:
                    print(f"{label} r2 ({r2_model}): {e}")
                else:
                    for key, vtx in captured.items():
                        merged[f"{label}.{key}"] = vtx

            cv_subs = subjects if subjects is not None else vr2.discover_subjects(
                bids_folder, model=cv_model, desc="cvr2")
            if not cv_subs:
                print(f"{label} cvr2 ({cv_model}): no subjects on disk — skipping")
                continue
            captured.clear()
            try:
                if cvr2_agg == "prevalence":
                    vr2.main_prevalence(cv_subs, [cv_model], bids_folder,
                                        cv_thr=cv_thr, min_prevalence=min_prevalence,
                                        baseline_model=baseline_model,
                                        smoothing=smoothing)
                else:
                    vr2.main_ttest(cv_subs, [cv_model], bids_folder,
                                   baseline_model=baseline_model, fdr_q=fdr_q,
                                   smoothing=smoothing)
            except SystemExit as e:
                print(f"{label} cvr2 ({cv_model}): {e}")
                continue
            for key, vtx in captured.items():
                merged[f"{label}.{key}"] = vtx
    finally:
        cortex.webgl.show = real_show

    return merged


def main(bids_folder: Path = BIDS_FOLDER, subjects: list[str] | None = None,
        families: list[tuple[str, str, str]] | None = None,
        r2_thr: float = 0.05, r2_sigma: float = 0.005,
        fdr_alpha: float | None = 0.01, fdr_mode: str = "empirical_null",
        cvr2_agg: str = "prevalence", cv_thr: float = 0.0,
        min_prevalence: float = 0.5, fdr_q: float = 0.05,
        baseline_model: str = DEFAULT_NULL_MODEL,
        smoothing: tuple[str, ...] = ("",)) -> None:
    bids_folder = Path(bids_folder)
    families = families or MODEL_FAMILIES

    ds = build_datasets(bids_folder, families, subjects, r2_thr, r2_sigma,
                        fdr_alpha, fdr_mode, cvr2_agg, cv_thr, min_prevalence,
                        fdr_q, baseline_model, smoothing)
    if not ds:
        raise SystemExit("Nothing to show — check model dirs / subjects.")

    print(f"\nLaunching pycortex viewer with {len(ds)} dataset(s):")
    for key in ds:
        print(f"  {key}")

    # cortex.webgl.show()'s server thread is daemon=True and dies the instant
    # this script's main thread returns; open_browser=True also builds the
    # URL from the machine hostname, which usually fails to resolve in a
    # browser. Keep the process alive ourselves and open a plain localhost
    # URL instead (see the pycortex skill).
    server = cortex.webgl.show(ds, open_browser=False, autoclose=False)
    url = f"http://localhost:{server.port}/mixer.html"
    print(f"\n=== WEBSHOW READY ===\nOpen this URL:  {url}\n======================\n",
          flush=True)
    subprocess.run(["open", url])
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Subject labels (default: discover per model/desc from disk)")
    p.add_argument("--families", nargs="+", default=FAMILY_NAMES,
                   choices=FAMILY_NAMES,
                   help=f"Model families to include (default: all — {' '.join(FAMILY_NAMES)})")
    p.add_argument("--r2-thr", type=float, default=0.05,
                   help="R²/cvR² alpha-masking floor (fraction 0-1; default 0.05)")
    p.add_argument("--r2-sigma", type=float, default=0.005,
                   help="Gaussian-CDF alpha transition width (default 0.005)")
    p.add_argument("--fdr-alpha", type=float, default=0.01,
                   help="[r2 datasets] Empirical-null FDR-alpha for the alpha "
                        "mask threshold (default 0.01). Pass 0 to disable and "
                        "use --r2-thr as a fixed cohort-wide threshold instead.")
    p.add_argument("--fdr-mode", default="empirical_null",
                   choices=["empirical_null", "group", "per_subject"],
                   help="[r2 datasets] see visualize_mean_r2_fsaverage.py --fdr-mode.")
    p.add_argument("--cvr2-agg", default="prevalence",
                   choices=["prevalence", "ttest"],
                   help="[cvr2 datasets] 'prevalence' (default): per-vertex "
                        "proportion of subjects with cvR² > --baseline-model "
                        "cvR², alpha-masked at --min-prevalence. 'ttest': "
                        "paired t-test of (model cvR² − baseline cvR²), "
                        "whole-cortex BH-FDR at --fdr-q — more rigorous but "
                        "prone to rendering blank at this dataset's current "
                        "signal levels.")
    p.add_argument("--cv-thr", type=float, default=0.0,
                   help="[cvr2 prevalence] per-subject cvR² positivity "
                        "threshold (default 0.0).")
    p.add_argument("--min-prevalence", type=float, default=0.5,
                   help="[cvr2 prevalence] alpha-mask threshold: fraction of "
                        "subjects that must be positive (default 0.5).")
    p.add_argument("--fdr-q", type=float, default=0.05,
                   help="[cvr2 ttest] Benjamini-Hochberg FDR level for the "
                        "paired t-test vs --baseline-model (default 0.05).")
    p.add_argument("--baseline-model", default=DEFAULT_NULL_MODEL,
                   help="[cvr2 datasets] null-model cv dir each family's cvR² "
                        f"is compared against (default {DEFAULT_NULL_MODEL!r}).")
    p.add_argument("--smoothing", nargs="+", default=[""],
                   choices=["", "_smoothed"],
                   help="BOLD-smoothing variant(s) to include (default: "
                        "unsmoothed only, to keep the dataset list short — "
                        "pass `--smoothing '' _smoothed` for both).")
    args = p.parse_args()

    families = [f for f in MODEL_FAMILIES if f[0] in args.families]
    fdr_alpha = args.fdr_alpha if (args.fdr_alpha and args.fdr_alpha > 0) else None

    main(Path(args.bids_folder), subjects=args.subjects, families=families,
        r2_thr=args.r2_thr, r2_sigma=args.r2_sigma,
        fdr_alpha=fdr_alpha, fdr_mode=args.fdr_mode,
        cvr2_agg=args.cvr2_agg, cv_thr=args.cv_thr,
        min_prevalence=args.min_prevalence,
        fdr_q=args.fdr_q, baseline_model=args.baseline_model,
        smoothing=tuple(args.smoothing))
