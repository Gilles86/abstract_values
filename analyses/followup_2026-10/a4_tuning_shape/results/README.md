# A4: tuning-function shape near the cardinals

Status: complete (29 Oct 2026). n = 36 MRI participants for V1–V3 and 35 for IPS: sub-36 has
no Wang IPS masks yet. Unsmoothed GLMsingle gabor betas. Every model is fitted to the same
voxels (the permnull ROI union) and evaluated on the same leave-one-run-out folds (16 per
participant).

## What was run

Code is in `/Users/gdehol/git/abstract_values/analyses/followup_2026-10/a4_tuning_shape/`:

- `fit_tuning_shape.py` and `fit_tuning_shape.sh` run on the cluster, one array task per
  participant (job 7839319, 36/36 COMPLETED).
- `analyze_tuning_shape.py` runs locally and makes the tables and figures.

The new model classes are in `abstract_values/encoding_models/models.py`. No existing spec
was changed.

All models live in orientation space.

| Tag | Model | Free per voxel |
|---|---|---|
| `null` | Training-set mean (= `aprf-null.cv`) | – |
| `vm` | Single axial von Mises PRF (= `vonmises-prf`: grid, then Adam, 1000 iterations) | μ, κ, amplitude, baseline |
| `gvm` | **(a) Exponential-power von Mises** `exp(κ − κ·d^s)/(πI₀(κ))`, d = 1 − cos 2(θ−μ). s = 1 is the von Mises; s < 1 gives a pointed peak; s > 1 gives a flatter top with lighter tails. Warm-started from `vm` with s = 1, so it nests `vm` exactly. | + shape s |
| `warp` | Von Mises on a cardinal-warped axis F_a(θ) = θ + (a/4) sin 4θ, with a free per voxel. a > 0 makes tuning narrower at 0° and 90°. Warm-started from `vm` with a = 0. | + warp a |
| `basis_a{a}` | **(b) 8-channel von Mises basis** (κ = 2, ridge α = 10). At a = 0 this is the deployed uniform `vonmises.cv` model. The channels are spaced evenly on F_a, so for a > 0 they are denser and narrower at the cardinals; a < 0 is the oblique-dense control. a ∈ {−0.5, 0, 0.25, 0.5, 0.75}. | 8 weights |

Measures:

- **cvR² minus null.** Mean over all ROI voxels, the same convention as the talk figure
  `talks/2026-10_update_ruff_bedi/make_prf_model_comparison_fig.py`.
- **Binned held-out error.** Out-of-sample SSE per orientation bin (90° ± 15°, 0° ± 15°,
  other). It is reported as the fraction of the null model's held-out error removed. It is
  computed on "signal" voxels: those where `vm` or the uniform basis beats the null. The
  voxel set is the same for every model.
- **Decoding** (the decode_gabor machinery):
  - per fold and ROI, the top 100 voxels by mean training R² of `vm` and the uniform basis,
    the same voxels for every decoder
  - Student-t spherical noise model (ResidualFitter)
  - flat-prior posterior on a 0.5° grid
- **Decoding metrics:**
  - |error| of the circular posterior mean
  - posterior circular SD
  - log density at the true orientation, a proper score that combines accuracy and
    calibration
  - 50/80/95% HPD coverage, summarised as `cov_err`, the mean |coverage − nominal|
- **Model-implied A3 index.** Computed exactly as in A3 `compute_rdms.model_rdm`:
  - D_ij = mean over voxels of ((f(i) − f(j))/σ)²
  - the voxels are ROI signal voxels
  - fits use all data
  - local ratio = d90/d(other 20 adjacent pairs)
  - CI = (d90 − dadj)/dall

Stats are paired across participants: the mean difference with a 95% t-CI and the count of
participants above 0. All p-values are in `a4_contrasts.tsv`.

## Numbers

### (a) Sharper or lighter-tailed single PRF: no

The fitted shape leans towards **flat-topped, lighter-tailed** curves, not pointed ones:

- median log₂ s per participant is +0.36 ± 0.04 (SEM) in V1 and +0.23 to +0.27 in IPS
- 37% of V1 voxels have s < 1

The extra parameter does not generalise. ΔcvR² for `gvm` − `vm`:

| ROI | ΔcvR² (pp) [95% CI] | Participants better |
|---|---|---|
| V1 | −0.168 [−0.190, −0.146] | 1/36 |
| IPS1–2 | −0.091 [−0.116, −0.067] | 3/35 |

Every ROI is negative. The per-voxel warp model behaves the same way:

- fitted a > 0 in about 57–61% of voxels, median a = 0.17 in V1
- ΔcvR² V1 −0.085 [−0.110, −0.061] pp, 3/36 better

### (b) Cardinal-dense basis: small but consistent gain

cvR² rises monotonically with the warp a in every ROI. The oblique-dense control is worse.

| ROI | a = 0.5 − uniform (pp) | Better | a = −0.5 − uniform (pp) |
|---|---|---|---|
| V1 | +0.080 [0.041, 0.120] | 30/36 | −0.090 [−0.130, −0.049] |
| V2 | +0.087 [0.062, 0.112] | 33/36 | −0.103 |
| V3 | +0.085 [0.060, 0.110] | 35/36 | −0.099 |
| IPS0 | +0.181 [0.131, 0.230] | 33/35 | −0.187 |
| IPS1–2 | +0.287 [0.188, 0.387] | 34/35 | −0.289 |
| IPS3–5 | +0.172 [0.106, 0.238] | 29/35 | −0.195 |

The gain has not saturated at a = 0.75 (IPS1–2 +0.40 pp). For scale, it is about 10–20% of
what the uniform basis already gains over the single von Mises (+1.0 to +1.6 pp).

Where the gain sits:

- In **IPS** it is concentrated near 90°. Error removed near 90° minus elsewhere, a = 0.5:
  IPS1–2 +0.79 [0.45, 1.12] pp, IPS0 +0.44 [0.20, 0.68], IPS3–5 +0.39 [0.12, 0.65].
- In **V1–V3** it is not: V1 +0.09 [−0.05, 0.23], V3 −0.07 [−0.14, −0.01].

### Decoding near 90°

Overall, decoding is weak and badly calibrated for every model (spherical noise, top 100
voxels):

- mean |error| is 26–28° in V1–V3 and about 38° in IPS (chance is 45°)
- 80% HPD coverage is only 0.24–0.34

Cardinal basis (a = 0.5) against the uniform basis, for 90° ± 15° trials:

- **|error| is lower:**
  - V1 −0.86° [−1.17, −0.55] (31/36 lower)
  - V2 −0.82°
  - IPS1–2 −1.21° [−2.04, −0.39]
- **The log score is worse:** V1 −0.17 [−0.27, −0.06], IPS1–2 −0.28 [−0.43, −0.13].
- **Coverage error is slightly larger** (+0.005 to +0.018).
- So the posterior is pulled towards 90°: more accurate there, but more confidently wrong when
  it misses.
- Over all orientations the change is about 0.1°.

The oblique basis is the mirror image: larger error near 90° (V1 +0.79°) and a better log
score.

`gvm` and `warp` do not improve 90° error (V1 −0.06° and −0.27°, CIs include 0). They have a
worse log score everywhere (V1 `warp` −0.11 [−0.17, −0.04]).

### Does it change A3's 90° index?

**Only by construction, and it does not close the gap.**

The model-implied local ratio (ratio of group means, participant bootstrap,
`a4_index_group_ratio.tsv`):

| Model | V1 | IPS1–2 |
|---|---|---|
| `vm` | 0.67 [0.49, 0.89] | 0.95 [0.64, 1.37] |
| Uniform basis | 0.75 [0.67, 0.85] | 0.45 [0.40, 0.50] |
| Cardinal basis, a = 0.5 | 1.46 [1.31, 1.63] | 0.98 [0.89, 1.08] |
| Cardinal basis, a = 0.75 | 1.80 [1.62, 2.02] | 1.30 [1.19, 1.43] |
| `gvm` | 0.73 | 1.28 |

The `vm` value of 0.67 reproduces A3's 0.69.

- Every model stays far below the crossnobis local ratio of about 6 that A3 measures (V1 6.0,
  IPS1–2 7.0).
- The cardinal basis raises the CI index by +0.044 to +0.051 in **every** ROI (36/36 or 35/35
  participants). V1 moves exactly as much as IPS.
- That shift is a property of the basis geometry: denser channels at 90° give larger local
  distances whatever the data. It is therefore not evidence for a categorical code, and it
  leaves the IPS-vs-V1 comparison unchanged.
- A3's empirical crossnobis index does not depend on any model, so A3's conclusions stand.

## Figures (in this folder)

- `fig_a4_cvr2.pdf`:
  - a: ΔcvR² of the shape families against `vm`
  - b: ΔcvR² against the basis warp a, per ROI
  - c: binned held-out error in V1
- `fig_a4_params.pdf`: fitted shape s and warp a per ROI.
- `fig_a4_decoding.pdf`: |error| and 80% HPD coverage by orientation bin, for V1 and IPS1–2.
- `fig_a4_index.pdf`: model-implied CI index per model and ROI.

Error bars are 95% t-CIs across participants. The dots in fig_a4_cvr2 a and fig_a4_params are
individual participants.

## Plain reading

Making each voxel's tuning curve sharper, lighter-tailed or cardinal-warped does not
generalise: the extra per-voxel parameter overfits in every ROI.

What does help, a little, is a population basis with more and narrower channels near 0° and
90°. That gain is consistent across participants and is largest in IPS, where it concentrates
on 90°. It also lowers decoding error at 90° by about 1° but makes those posteriors
overconfident.

So the data favour "more resolution near the cardinals at the population level" over
"differently shaped tuning curves". That fits efficient coding of a cardinal-heavy prior. But
the effect is an order of magnitude smaller than the basis-vs-single-PRF difference.

None of the encoding models comes close to reproducing the roughly 6× crossnobis jump at 90°
that A3 finds. The better model raises the model-implied index equally in V1 and IPS, and it
does so by construction. The question of whether 90° is built in IPS or inherited from V1
therefore still rests on A3's model-free distances.

## Flags

- **The cardinal-basis gain may not be about tuning density.** It could reflect any
  orientation-specific response structure at the cardinals that the basis can express. One
  example is the lower mean betas at 90° that A3 reports. Treat it as "more model flexibility
  at the cardinals helps", not as proof of denser neural channels.
- **Fixed basis settings.** The basis keeps 8 channels and κ = 2. The warp sweep was not
  tuned beyond a = 0.75, and the gain had not peaked there.
- **Small, coarse warp effect.** a is one population-level warp; there is no per-participant
  fit of a.
- **Decoding is overconfident for every model.** Coverage is far below nominal because the
  noise model is spherical (the project default). The calibration comparisons between models
  are relative only.
- **Selection is shared but not neutral.** Signal and decoding voxels are chosen with `vm`
  and the uniform basis. The set is the same for every model, but it is not neutral towards
  the basis family.
- **The per-bin and decoding-near-90° contrasts are exploratory.** There are many ROI ×
  measure cells and no multiplicity correction.
- **sub-36 is missing from IPS** because it has no Wang masks.

## Outputs

- `/data/ds-abstractvalue/derivatives/followup_2026-10/a4_tuning_shape/`, mirrored at
  `/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/followup_2026-10/a4_tuning_shape/`.
  Each `sub-*/` folder holds `_voxels.tsv.gz`, `_decoding.tsv.gz` and `_index.tsv`. The
  summary TSVs and figures sit at the top level.
- The same TSVs, figures and this README are in
  `/Users/gdehol/git/abstract_values/analyses/followup_2026-10/a4_tuning_shape/results/`.
