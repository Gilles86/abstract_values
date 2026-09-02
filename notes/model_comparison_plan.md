# Decoupling architecture, space and flexibility

Plan for a model comparison that separates three things currently confounded in
the `vonmises` vs `aprf` contrast.

## The three factors

| Factor | Levels |
|---|---|
| **A. Architecture** | linear (untuned) · one bell · basis set of bells (free weights) |
| **B. Space** | orientation (deg, π-periodic) · value (CHF) |
| **C. Flexibility** | tuning fixed across sessions · tuning free per session (cdf vs inverse_cdf) |

## What already exists

Most of the grid is fitted. Marking A × B for the joint (C = fixed) case:

| | orientation | value |
|---|---|---|
| linear | **missing** (see below) | `aprf-linear` (`LinearValuePRF`) |
| one bell | **MISSING — the real gap** | `aprf` (log-Gauss), `aprf-gauss` (symmetric) |
| basis set | `vonmises` (8 × von Mises, free weights) | `aprf-weighted` (N log-Gauss, free weights) |

Flexibility (C) exists for: `aprf-session-shift` (mode), `aprf-fwhm-shift`,
`aprf-fully-shifted`, `aprf-gauss-session-shift`, and `vonmises --session-shift`
(per-session basis weights → `vonmises-shift.cv`).

Hyperparameters are already exposed: `fit_vonmises_model.py --n-basis --kappa`,
`fit_aprf_weighted.py --n-basis --fwhm --basis {loggauss,gaussian}`. And
`sweep_v1_k_kappa.py` already sweeps n_basis × kappa in V1 with leave-one-run-out
CV.

**So the current `vonmises` vs `aprf` comparison confounds A and B**: it pits an
8-weight orientation basis against a 4-parameter value bell. That is exactly why
the V1 result looked strange.

## The one model to build

**Single von Mises PRF in orientation space** — free `mu`, `kappa`, `amplitude`,
`baseline` per voxel. `braincoder.models.AxialVonMisesPRF` already exposes
precisely those four `parameter_labels` and inherits from `GaussianPRF`, so it
drops straight into the `fit_aprf.py` grid-search + Adam pattern. This is the
architectural twin of `aprf` and completes the A × B grid.

A linear/untuned orientation model has no honest analogue of
`LinearValuePRF`'s monotonic ramp (orientation is circular). The right minimal
model is a **single circular harmonic** (sin/cos regression = `n_basis=2`), which
`fit_vonmises_model.py` can already produce.

## The design problem worth being explicit about

**B and C are not independent, and that is the whole experiment.**

Within a session, value is a deterministic monotonic function of orientation, so
*any* sufficiently flexible model fits equally well in either space. A bell in
value space is a warped bell in orientation space. The spaces are only
distinguishable **across** the mapping flip:

- a voxel truly tuned to **orientation** keeps its orientation tuning across
  sessions, so its *value* tuning inverts;
- a voxel truly tuned to **value** keeps its value tuning, so its *orientation*
  tuning inverts.

Consequences for the design:

1. The identifying contrast is **joint (C = fixed) fits in each space**. That is
   the test of "which variable is this voxel actually stable in".
2. Once C = free-per-session is allowed in both spaces, the space contrast
   largely **collapses** — each model can just re-fit per session. A fully
   shifted model in value space and one in orientation space are close to the
   same model. Do not read a space effect off shifted fits.
3. So C is best treated not as a nuisance level to average over but as its own
   result: *how much does allowing a shift buy in each space?* A voxel coding
   value should gain little from shifting in value space and a lot from shifting
   in orientation space.

## Phases

**Phase 1 — fill the grid (the actual gap).**
Write `fit_vonmises_prf.py` + `_cv.py` mirroring `fit_aprf.py`, using
`AxialVonMisesPRF` with all four parameters free. Register in `model_specs.py`,
add to the Snakefile alongside the other fits and to `cvr2_dirs` for surface
sampling. Then the 3 × 2 architecture × space grid is complete at C = fixed.

**Phase 2 — the clean comparison.**
Six models, all joint, compared per vertex on **cvR²** against `aprf-null.cv`.
Report as a 3 × 2 heat map of win share plus per-ROI (V1, NPC, M1) breakdowns.
Guardrails learned the hard way today:
- cvR², never full-fit R² — parameter counts differ by 2× or more across cells.
- **Equal-sized pools.** Best-of-3 vs best-of-1 inflated the value side badly
  earlier; compare one model per cell, or take the max within equal-sized sets.
- Prevalence is a display threshold, not a test.

**Phase 3 — flexibility as a result.**
For each (A, B) cell, fit the shifted variant and report ΔcvR² (shifted −
joint). The prediction above is directly testable: value-coding cortex should
show a large Δ in orientation space and a small one in value space, and vice
versa. This is the panel that actually answers the scientific question.

**Phase 4 — basis hyperparameters.**
Extend `sweep_v1_k_kappa.py` to (a) run in NPC as well as V1, and (b) have a
value-space twin sweeping `n_basis` × `fwhm` for `aprf-weighted`. Two things to
watch:
- **n_basis and dispersion interact.** Too narrow for the spacing and the basis
  does not tile the stimulus space; too broad and every basis function is nearly
  collinear, so the weights are unidentified and cvR² collapses. Sweep jointly,
  never one at a time.
- **Match the units across spaces.** N von Mises over 0–π and N log-Gaussians
  over 2–42 CHF are only comparable if the dispersion is expressed as a fraction
  of inter-basis spacing (`fit_aprf_weighted` already defaults `fwhm` to 2×
  spacing; give the von Mises side the same convention rather than a raw kappa).

Pick k and dispersion by cvR² *before* Phase 2, or the architecture comparison
inherits an arbitrary hyperparameter choice.

## Cost

Phase 1 is one new model class plus two fit scripts — the model already exists in
braincoder, so this is mostly wiring. Phases 2–3 are analysis over fits that
mostly exist. Phase 4 is the expensive one: a k × dispersion grid × 29 subjects,
though `sweep_v1_k_kappa.py` shows the pattern and restricting to ROIs keeps it
affordable.
