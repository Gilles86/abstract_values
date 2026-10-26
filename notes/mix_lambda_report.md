---
title: "Value code or quantile code? Fitting the code directly to the voxel data"
subtitle: "Shared mixing parameters, neural_priors style, with recovery on simulated betas"
date: 2026-10-26
geometry: margin=1.8cm
mainfont: Helvetica
fontsize: 10pt
---

<!-- Build: pandoc notes/mix_lambda_report.md --resource-path=notes --pdf-engine=xelatex \
     -o notes/figures/mix_lambda_report.pdf -->

**Bottom line.** Fitted directly to the betas and compared on held-out runs, tuning in every
ROI, IPS included, is best described by a **quantile code**. The preferred value keeps its quantile
when the mapping changes, *and* the tuning width follows the local density of values. Each half
of that helps on its own (position +1.3 to +1.6, width +0.9 to +1.1 × 10⁻³ cvR² in IPS), and both
together help most (+1.6 to +2.5).

On betas simulated with a known code at realistic noise, this model comparison picks the right
code: value 61%, position-only 79%, width-only 65%, quantile 86% of participants × ROIs. In IPS
the observed contrasts closely resemble the simulated quantile code. The
**estimated** mixing parameters are a different story: at realistic noise they do not recover, so
the conclusion rests on the comparison between fixed models.

The same pattern holds in **V1** (quantile code +2.4 × 10⁻³). Here the quantile is orientation
rank, so this is the signature of orientation tuning, and IPS cannot be told apart from it.

# 1. The models

**The mix axis.** x_λ(v) = (1 − λ) v + λ q(v), with q(v) = 2 + 40 F(v), where F is the value's
quantile within the session's value distribution.

- λ = 0 is value in CHF.
- λ = 1 is the quantile, which in this design equals orientation rank.

**Single λ** (`MixLambdaLogGaussianPRF`). A log-Gaussian on the mix axis, so position and width
move together.

**Position and width separately** (`MixPositionWidthGaussianPRF`, Gaussian tuning):

- **λ_μ** sets where the curve sits: the preferred position lives on x_{λ_μ}.
- **λ_σ** sets the axis on which the curve has a fixed width. At λ_σ = 0 the width is the same in
  CHF everywhere; at λ_σ = 1 it is the same in quantile units, so the curve is narrow where values
  are dense.

**How they are fitted.** Each voxel gets its own position, width, amplitude and baseline; the
mixing parameter(s) are shared by all voxels of a participant's ROI. Fitting is grid search
followed by gradient descent. The comparison uses leave-one-run-out cvR² (8 folds; run k of both
sessions), on run-demeaned betas of all ROI voxels, with no selection. The fixed models are:

- λ in {0, .5, 1} for the single-λ model;
- the corners (λ_μ, λ_σ) in {(0,0), (1,0), (0,1), (1,1), (.5,.5)} for the position/width model;
- plus the shared, freely estimated version of each.

![](figures/fig_quantile_model_comparison.pdf)

# 2. Position and width separately (figure)

- **a.** What each corner predicts for one voxel.
- **b. Observed**, cvR² relative to the value code (0,0), mean ± s.e.m. over 35 participants.

  | ROI | Position (1,0) | Width (0,1) | Both (1,1) |
  |---|---|---|---|
  | IPS0 | +1.38 (p = .0002) | +0.95 (p = .004) | +1.59 (p = .0005) |
  | IPS1–2 | +1.30 (p = .01) | +0.91 (p = .03) | +2.46 (p = .0001) |
  | IPS3–5 | +1.58 (p < .001) | +1.14 (p = .03) | +2.43 (p = .0005) |

  - The full code beats each half in IPS1–2 (vs position +1.16, p = .005; vs width +1.55,
    p = .007) and IPS3–5 (vs width +1.30, p = .007).
  - V1 shows the same ranking (both +2.36, p < 10⁻⁵), as do V3a/b and hV4.
- **c.** The share of participants whose best code is each of the four.
- **d–e. Recovery** at noise ×2, which matches the observed cvR² level. Each simulation reproduces
  each voxel's real Misaligned preferred value, width and signal level.
  - Every simulated code wins its own corner: value 61%, position-only 79%, width-only 65%,
    quantile 86%.
  - Observed IPS contrasts (position +1.4, width +1.0, both +2.2) match the simulated quantile
    code (+1.5, +1.1, +3.3) much better than any other simulation. The simulated value code gives
    all three negative.
  - The noise ×1 simulations are still running and are not shown.

# 3. Single λ (log-Gaussian on the mix axis)

| cvR² vs λ = 0 (× 10⁻³) | V1 | hV4 | V3a/b | IPS0 | IPS1–2 | IPS3–5 |
|---|---|---|---|---|---|---|
| Observed, λ = 1 | +2.61 | +1.36 | +2.49 | +1.70 | +2.50 | +2.72 |
| p (paired t, n = 35) | 2e-6 | 7e-4 | 2e-8 | 7e-4 | 3e-5 | 1e-5 |
| Simulated λ = 0, noise ×2 (IPS1–2) | | | | | −2.98 | |

The fixed λ = 1 model wins in every ROI, and λ = 0.5 lies in between. Simulations from a value
code give the opposite sign.

# 4. What does not work: estimating the mixing parameter

- **Single λ.** The shared estimate, fitted by gradient descent, recovers a value code at low
  noise (true 0 gives 0.04). At noise ×3 it drifts to its middle starting value whatever the
  truth (true 0 gives 0.44–0.49, true 1 gives 0.47–0.53). The simulated λ = 1 there moves only
  the position, so this model cannot reach it, and it comes out at about 0.5 even at noise ×1.
  The shared fit's out-of-sample fit is no better than the best fixed model.
- **Position/width model.** The estimates do better at noise ×2: λ_μ is 0.73 for the
  position-only code and 0.05 for the value code; λ_σ is 1.11 for the width-only code and 0.09 for
  position-only. The observed medians are λ_μ = 0.51 and λ_σ = 0.83. But per-fold SDs are about
  0.2–0.3, and the shared fit never beats the best fixed corner on held-out data.
- **So:** report the comparison between fixed models, as neural_priors did, not λ̂.
- **Consistent with earlier results.** This matches the earlier failure of λ from fitted preferred
  values (the hierarchical approach) and the conservative but valid cvR² profile
  (`lambda_executive_summary.pdf`).

# 5. Interpretation and caveats

- **The parietal code is not anchored to CHF.** When the mapping changes, IPS tuning keeps its
  orientation rank (= quantile) in both position and width, exactly like V1. These data cannot
  separate "efficient coding of value" from "orientation coding", because the quantile *is*
  orientation rank here. Separating them needs a design change: non-uniform orientation sampling or
  a non-monotone mapping.
- **Absolute fit is low.** cvR² is negative for all models in IPS once every ROI voxel is
  included; the comparison is relative.
- **The noise in the simulations** is independent Gaussian at 1–2× each voxel's residual SD. Real
  noise is structured over runs and sessions.
- **Pending.** A finer estimate of (λ_μ, λ_σ) per participant and ROI is running: a cvR²
  surface over a 0.1-step grid, followed by gradient descent from its peak.

*Code:* `abstract_values/encoding_models/{models, fit_mix_lambda, simulate_lambda_pipeline}.py`,
`visualize/figure_mix_lambda.py`. *Data:* `notes/data/mix_lambda_{mix,width}.tsv`.
