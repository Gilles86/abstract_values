# Attentional Field model for value-coding pRFs — design sketch

**Status:** design only. Not yet implemented. Decision after reviewing the
efficient-coding Q-Q overlay in `notes/figures/shifted_preferred_value.pdf`.

## Motivation

The session-shift fits (`aprf-session-shift`) let each voxel have a
different preferred value (`mode_1`, `mode_2`) per session, while
amplitude/baseline/fwhm stay shared. Two interpretations of any observed
mode shift:

1. **True reorganization** — the voxel's underlying tuning has remapped
   to match the new value distribution (efficient coding).
2. **Attentional gain** — the voxel's tuning is unchanged, but a
   multiplicative attentional field, centered on a "currently attended"
   value, weights the response such that the effective tuning peak
   appears shifted.

These two predict the same first-order moment shift but differ in
higher-order structure (selectivity, gain, tail behaviour). Fitting an
AF model gives us a principled way to compete the two accounts.

## Model

Per voxel `v`, per trial with value `V`:

```
r_v(V) = baseline_v + amplitude_v · pRF_v(V; mode_v, fwhm_v) · g(V; V_att, σ_att)
g(V; V_att, σ_att) = exp(−(V − V_att)² / (2 σ_att²))
```

- `pRF_v` is the log-Gaussian tuning curve (same as `LogGaussianPRF` in
  braincoder, parameterised by mode/fwhm).
- `g(V)` is a Gaussian attentional gain on the *value axis* (not the
  retinotopic axis — there is no retinotopic competition in this task,
  the gabor is foveal each trial).
- `V_att`, `σ_att` are **shared across all voxels** within a (subject,
  condition).

### Parameter inventory per subject

| Parameter        | Shape       | Notes |
|---               |---          |---    |
| mode_v           | (n_vox,)    | shared across the two conditions |
| fwhm_v           | (n_vox,)    | shared across the two conditions |
| amplitude_v      | (n_vox,)    | shared across the two conditions |
| baseline_v       | (n_vox,)    | shared across the two conditions |
| V_att^(cond)     | (2,)        | one per condition (CDF / InvCDF) |
| σ_att^(cond)     | (2,)        | one per condition |

So in addition to the 4 per-voxel params, only **4 global params per
subject** are added. The model has roughly the same degrees of freedom
as the session-shift model (which adds 1 extra `mode` per voxel × 2
sessions = 2 modes/voxel) but localises the shift signal into a single
attentional centroid rather than spreading it across all voxels.

## Identifiability concerns

1. **`mode_v` vs `V_att` for an isolated voxel.** With only one
   condition, `mode_v + V_att` are not separately identifiable — only
   their sum/product matters. Solved by sharing `V_att` across voxels:
   the global centroid is pinned by the *population* of `mode_v`'s.
2. **`fwhm_v` vs `σ_att`.** A wider pRF and a wider attentional field
   both broaden the effective tuning. We mitigate by also sharing
   `σ_att` across voxels, and by fitting both conditions jointly so the
   relative-to-unattended baseline is informative.
3. **Per-voxel V_att.** Avoid — equivalent to the session-shift model
   plus a Gaussian envelope, with no extra constraint.

## Falsification grid

Three candidate models on the same (subject × session × voxel) data:

| Model            | Per-voxel free | Global per cond | Story it tells |
|---               |---             |---              |---             |
| standard (`aprf`) | `mode, fwhm, amp, baseline` | — | Single tuning, no condition effect |
| session-shift    | `mode_1, mode_2, fwhm, amp, baseline` | — | Per-voxel remapping |
| AF (this sketch) | `mode, fwhm, amp, baseline` | `V_att, σ_att` (×2) | Global attentional gain |

Compare on:
- **Held-out cv-R²** (leave-one-run-out, with fold matching the
  existing cv pipeline) — model selection by predictive accuracy.
- **Population shift pattern**: AF predicts a *coherent* gain centered
  on `V_att` (same for every voxel), so per-voxel shifts should *all*
  point toward `V_att`. Session-shift predicts *idiosyncratic*
  per-voxel shifts. The Q-Q line in
  `shifted_preferred_value.pdf` shows what efficient-coding-remap
  looks like (divergent S through y=x). AF predicts a different shape
  (convergent — voxels pulled toward a single point).

## Implementation cost

If we decide to fit:

1. **New braincoder model class** (~50 LoC) wrapping `LogGaussianPRF`
   with a multiplicative gain on the value axis. Use the existing
   `model_stimulus_amplitude=True` plumbing (see `SessionShiftedLogGaussianPRF`
   in `abstract_values/encoding_models/models.py` for the recipe).
2. **New fit script** `abstract_values/encoding_models/fit_aprf_af.py`
   — same structure as `fit_aprf.py` but with global-parameter handling
   (2 global × 2 conditions = 4 extra params at the subject level,
   optimised jointly with per-voxel params using a 2-stage approach:
   grid-search global, then alternate per-voxel/global until convergence).
3. **SLURM job** + Snakemake rule (parallel to `fit_aprf_session_shift`).
4. **Decoding pass** (`decode_value.py --model af`) only needed if the
   AF model becomes the published encoder; not on the critical path
   for the identifiability falsification.

Estimated effort: ~1 day of code + ~6 h of cluster time for fitting
across the cohort. **Decision: defer until we've reacted to the
efficient-coding Q-Q overlay.**

## Open question

The attentional gain assumes that participants are attending to a
specific value range. Plausible candidates for `V_att`:
- Mean reservation price (from BDM auction)
- Currently-attended decision threshold (e.g., expected payout)
- Just the value of the *previous* trial (sequential effects)

If `V_att` differs systematically between CDF and InvCDF conditions in
behaviorally interpretable directions, we have a story. If it's just an
arbitrary best-fit value with no relation to behavior, the model is
empirically useful but theoretically thin.
