# Orientation-tuning vs value-tuning disambiguation

**Status:** design only. Triggered by the discovery that the
"efficient-coding histogram-matching" Q-Q line and the "stable
orientation-tuning" null land at *identical* loci in the
`(mode_cdf, mode_invcdf)` hexbin (see
`notes/figures/shifted_preferred_value.pdf` page 2 — yellow dots sit
exactly on the teal Q-Q line).

## The confound

Both candidate accounts of the per-condition mode shifts predict the
same per-voxel cloud in value-axis fit-space:

1. **Orientation-tuning.** Voxels are tuned to gabor orientation;
   they don't reorganize between conditions. The session-shift PRF on
   the *value* axis only appears to shift because the orientation→CHF
   mapping flips between conditions.
2. **Value-tuning + efficient-coding remap.** Voxels truly encode
   value; tuning relocates to histogram-match the new stimulus
   distribution.

Because the orientation→CHF mapping is monotonically increasing in
*both* conditions and shares the same orientation grid, sorting by
orientation = sorting by either condition's value. Rank-matched
predictions therefore coincide.

## The orthogonal test: fit on the orientation axis, per session

If a voxel is truly orientation-tuned, its preferred *orientation*
should be invariant across sessions. If it is value-tuned with
remapping, its preferred *orientation* should shift between sessions
to keep the preferred value stable.

Concretely, for a voxel that prefers value V* under value-tuning:

```
θ_pref^CDF    = V_cdf⁻¹(V*)
θ_pref^InvCDF = V_invcdf⁻¹(V*)
Δθ_pred       = θ_pref^InvCDF − θ_pref^CDF      (function of V*)
```

Under orientation-tuning, the prediction is simply `Δθ = 0` for every
voxel.

Because the two mappings differ in their derivatives w.r.t. orientation
(CDF is steeper in the middle, InvCDF is steeper at the tails), `Δθ`
under value-tuning is a *known non-zero function of V* (or
equivalently, of θ_pref^CDF). For a voxel whose CHF preference sits
near the mean of the distribution, predicted `Δθ` is small; for a
voxel preferring an extreme value, `Δθ` is large and signed.

## What's needed to run it

### Data
- Existing single-trial gabor betas (per-session, both sessions; cv
  variants not strictly required).
- ROI masks: NPCr (test) + BensonV1 (control — should be flat at
  `Δθ ≈ 0`).

### Code

The light path (no new model class):

1. **Modify `fit_vonmises_model.py`** to accept a `--per-session` flag.
   When set, it fits the existing `AxialVonMisesPRF` basis-weights
   independently for each session (closed-form lstsq is cheap; the
   model already supports this — just call `WeightFitter` on
   per-session subsets of the trial data). Output one weight set per
   session under
   `derivatives/encoding_models/vonmises_per_session/sub-XX/ses-N/func/`.
2. **Derive preferred orientation per voxel per session**: take a fine
   orientation grid (180 points in [0, π)), evaluate
   `basis_predictions @ weights`, return `argmax` per voxel. (Wrap as
   `vonmises_preferred_orientation.py` in `visualize/`.)
3. **Plot** `(θ_pref^CDF, θ_pref^InvCDF)` as a 2D hexbin per ROI, with
   the y=x diagonal and the predicted-shift curve under value-tuning
   (computed once from the (orientation, value) lookup table; see
   `_orientation_value_pairs()` in `shifted_preferred_value.py`).

The heavy path (new model class — more principled but more work):

- Implement `SessionShiftedAxialVonMisesPRF` (analog to
  `SessionShiftedLogGaussianPRF` in
  `abstract_values/encoding_models/models.py`) with `mu_1`, `mu_2`
  free per voxel, `kappa` shared. Fit via `WeightFitter` with a
  session-dependent basis matrix, save the per-session preferred
  orientations directly.

The light path is much faster to validate (an afternoon's work; no
new SLURM jobs needed since per-session weight fitting is closed-form
and runs locally in seconds per voxel).

## Predicted outcomes

| ROI  | Orientation-tuning prediction | Value-tuning prediction |
|---   |---                            |---                      |
| V1   | `Δθ ≈ 0` for almost all voxels | (V1 isn't expected to be value-tuned, so this is the control prediction either way) |
| NPCr | `Δθ ≈ 0` (with noise)         | `Δθ` is signed and structured by `θ_pref^CDF` — follows the inverse-Q-Q curve |

Two simple summary statistics:

1. **Mean |Δθ|** within each ROI. If NPCr's mean |Δθ| is reliably
   larger than V1's (signed test across subjects), that's evidence
   for value-tuning + remapping.
2. **Correlation of observed Δθ with the value-tuning prediction** as
   a function of `θ_pref^CDF`. Significant positive correlation in
   NPCr but not V1 falsifies the orientation-tuning account.

## Out of scope (for now)

- The AF (attentional gain) account — see
  [`attentional_field_design.md`](attentional_field_design.md). Both
  this test and the AF model address the same underlying question
  ("are the observed shifts real?"). The orientation-vs-value test
  is cheaper and more diagnostic for the *kind* of stimulus the voxel
  represents; the AF model adds a third candidate mechanism (gain
  modulation around an attended value) that this test cannot rule out
  on its own.
- Cross-decoding (train value decoder on CDF, test on InvCDF). Also
  diagnostic but coupled to the noise model + decoder hyperparameters;
  the encoding-side test is cleaner.
