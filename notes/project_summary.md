---
title: "Abstract values — project summary"
subtitle: "Where the fMRI and modelling work stands (35 participants)"
date: 2026-10-25
geometry: margin=1.8cm
mainfont: Helvetica
fontsize: 10pt
---

<!-- Build:
pandoc notes/project_summary.md --resource-path=notes --pdf-engine=xelatex \
    -o notes/figures/project_summary.pdf
Status tags: [E] established (survived controls), [T] tentative, [X] invalidated. -->

**In one paragraph.** People learn to map gabor orientation to a value in CHF and bid for it (BDM).
They do this under two mappings with different value distributions:

- Misaligned (repo `cdf`), dense near 12 and 32 CHF.
- Aligned (repo `inverse_cdf`), dense near 22 CHF.

Behaviour replicates Bedi et al.'s preregistered sign reversal of the valuation bias. IPS and V1
both carry reliable tuning. V1 codes orientation on a circle, and parietal cortex lays it out as an
ordered line. A cross-validated fit of the coding axis, now validated end to end by simulation,
puts IPS tuning at the quantile or orientation-rank end (λ̂ ≈ 0.75–0.95) rather than at fixed CHF.
The design cannot separate "quantile of value" from "orientation rank". The strongest
value-specific result is the V1-to-IPS connective-field coupling that follows the current mapping.
Individual differences (per-person λ, the perceptual vs value noise split) are not measurable
reliably.

# 1. Design and behaviour

![](figures/report_fig1_behaviour.pdf)

**Design**

- 23 orientations (7.5–172.5°) are mapped to 2–42 CHF. Two sessions, one mapping each, with the
  order counterbalanced by participant ID.
- 8 runs × 23 trials per session, so 368 trials per participant. n = 35 with MRI.
- ROIs: Benson V1–hV4 and V3a/b, and Wang IPS0, IPS1–2 and IPS3–5.
- Every QC incident was fixed rather than excluded. sub-28 is the worst mover and stays in.

**Behaviour and cognitive models**

- **[E] Sign reversal replicated.** The valuation-bias slope in 65–115° reverses between mappings:
  divergence +0.089, against +0.083 in the paper.
- **[E] Bid noise follows the mapping.** Bids are noisier under the Aligned mapping (SD 1.69 vs
  1.24 CHF) and collapse at 22 CHF.
- **[E] Categorical model wins.** The categorical efficient-coding model (Bedi et al., hard
  22 CHF / 90° gate) wins LOO (ΔELPD ≈ 270). Residuals are heavy-tailed, and modelling motor tails
  wins.
- **[E, negative] Noise split not identified.** The split of noise into perceptual vs value noise
  is not identified per participant, whether by model, model-free slope test or decoding.

# 2. What the parietal code looks like

![](figures/fig_circularity.pdf)

- **[E] Selection rule.** A voxel counts as carrying signal when its cvR² beats the null model's,
  not when cvR² > 0.
- **[E] Both regions are tuned, the type is undecided.** V1 and IPS carry reliable tuning (split-half
  r ≈ .2 within and across sessions). Mean-response model comparison cannot tell value from
  orientation tuning: each wins in about 45% of voxels in both regions.
- **[T] Circle vs line.** V1 is circular (a wrapped code fits better), and IPS1–2 is an ordered
  line (unwrapped fits better), on all voxels, smoothed, n = 35. Crossnobis RDMs show the same.
  Under permutation-null selection very few IPS voxels survive, so these contrasts are underpowered
  there.
- **[X] Retracted early claims.** The caudal-IPS "value cluster" on winner maps did not survive
  smoothing. The early session-shift "24/24" win came from biased voxel selection; under neutral
  selection, no-shift wins in 22 of 24.
- **[T, negative] vmPFC/OFC.** Low tSNR; no detectable value code.

# 3. Value code or quantile code? (λ)

![](figures/fig_lambda_summary_profile.pdf)

- **[E] Method validated.** The cvR² λ-profile fits a Gaussian coding model on the mix axis
  (λ = 0 fixed CHF, λ = 1 quantile) directly to the betas, leave-one-run-out, with one λ per
  participant and ROI. On betas simulated from each voxel's real fit with a known λ, it gives
  about 0 for λ = 0 at all noise levels and 0.3–0.7 for λ = 1. The bias is conservative.
- **[T] Result.** Observed group peaks: IPS0 0.75, IPS1–2 0.95, IPS3–5 0.95, V3a/b 0.85,
  hV4 0.70. All are beyond the simulated quantile code.
- **[X] Hierarchical λ from fitted preferred values.** At realistic noise it gives λ ≈ 1.8 for
  either truth, because noisy preferred values bunch mid-range. Details in
  `lambda_executive_summary.pdf`.
- **Confound.** Orientations are uniform and both mappings are monotone, so the quantile equals
  orientation rank. λ ≈ 1 is therefore also "IPS follows orientation rank". The fix is in the
  design: non-uniform orientation sampling or a non-monotone mapping.
- **[T] Cross-condition decoding.** Transfer degrades but does not invert (r = .19; MAE 12.8 vs
  8.5 CHF).
- **[X] Expected uncertainty vs value density.** The original r = −.61 was an artefact of the full
  noise covariance. It is −.17 with spherical noise.

# 4. Mapping-specific routing: V1-to-IPS connective fields

![](figures/fig_connectopy.pdf)

- **[T, strongest] Coupling follows the mapping.** IPS couples to the V1 orientation channel that
  carries the voxel's preferred value under the mapping currently in force. Everything shared
  across sessions cancels out of this score.
  - IPS0–5: +0.0133 ± 0.0031, p = .0001, n = 35, growing along IPS.
  - Replicates in NPC, which overlaps IPS.
  - Gain controls are only partial (map-level agreement p = .14 after gain regression). The reverse
    direction is null, with unknown power.
- **[T] Two components.** A stable orientation-rank component (t = 6.4) and a weaker
  mapping-moving one, seen on high-drive trials only.
- **[T] Individual differences.** Coupling strength correlates with value precision (ρ = .35,
  p = .04).

# 5. Decoding and brain–behaviour

![](figures/decoded_predicts_bid.pdf)

- **[E] Decoded value predicts the bid beyond the true value.** V1 +0.085 (p = 4e-6), NPC +0.047
  (p = .0005). The next-trial control is null.
- **[T] Tuned cortex and precision.** More tuned cortex goes with more precise bids (r = −.71). This
  is probably a general data-quality factor.

![](figures/fig_gradient.pdf)

- **[T] A gradient.** The V1 orientation code relates to perceptual precision, the IPS value code to
  value precision. It is fragile: the slope has P = .14 and disappears in the shrink + motor-tail
  fits.
- **[X/negative] Individual differences.** Per-person λ (both methods) is unrelated to behavioural
  λ and to σ_rep. Trial-wise decoded SD does not predict bid errors.

# 6. Open problems and next steps

1. **A neural_priors-style direct fit.** Fit the log-Gaussian aPRF on the mix axis in braincoder,
   with λ (and a width scale) shared across a participant's voxels and fitted by gradient descent.
   Compare fixed λ = 0, fixed λ = 1 and shared λ by cvR², and recover it on the same simulations.
   This would make the λ result consistent with the aPRF results.
2. **Connective-field coupling:** an anatomical control region, power for the reverse direction,
   and a tighter gain control.
3. **Quantile vs orientation:** state the confound in the paper and propose the design fix. No
   analysis of this dataset can break it.
4. **Cognitive model:** finish the prior-shape sweep (10 fits). Run recovery for the
   neural-covariate effects.

*The full one-page poster of the project:* `notes/figures/fig_summary.pdf`. *λ details:*
`notes/figures/lambda_executive_summary.pdf`.
