---
title: "Does parietal value tuning follow the value distribution?"
subtitle: "Hierarchical λ from session-shift aPRF preferred values — executive summary"
date: 2026-10-25
geometry: margin=1.8cm
mainfont: Helvetica
fontsize: 10pt
---

<!-- Build:
pandoc notes/lambda_executive_summary.md --resource-path=notes --pdf-engine=xelatex \
    -o notes/figures/lambda_executive_summary.pdf
Figures: python -m abstract_values.visualize.figure_lambda_summary -->

**Bottom line.** At the group level, IPS seemed to lean towards a quantile ("efficient") code: the
group mean λ in IPS is 0.93 [−0.25, 1.90]. But an end-to-end simulation shows that **at the noise
level of our data, this estimator returns λ ≈ 1.8 whether the true code is value (λ = 0) or quantile
(λ = 1)**. The preferred-value route therefore can't decide between the codes. Per-participant λ is
fairly stable across odd and even runs (ρ = .65), but it is unrelated to behaviour or to the
cognitive model's value noise.

# 1. Question and approach

![](figures/fig_lambda_summary_concept.pdf)

- Each participant saw two value-to-orientation mappings in separate sessions: Misaligned (cdf) and
  Aligned (inverse cdf). Each mapping gives a different value distribution (**a**).
- If a voxel codes value (λ = 0), its preferred value is the same in both sessions. If it codes the
  quantile (λ = 1), its preferred value moves to the same quantile of the new distribution
  (**b**, purple: flat near 22 CHF).
- λ is estimated per participant from the preferred value of each IPS voxel in each session
  (session-shift aPRF), with a hierarchical model across participants. Voxels are selected by the
  session-shift aPRF's cvR² over the null model, a gate that doesn't favour either code (**c**).

# 2. Group result

![](figures/fig_lambda_summary_group.pdf)

- **a.** Group μ is about 1 in IPS, higher in IPS0 (1.69) and V3a/b (1.45), and about 0 in hV4 (0.05).
- **b.** The estimate barely depends on the voxel gate or on adding the offset, but the intervals are
  wide.
- **c.** Participants spread from −0.7 to 1.8 (τ = 1.44).

# 3. Model checks

![](figures/fig_lambda_summary_checks.pdf)

- **a–c.** Without a per-participant session offset, the model misses each participant's median
  session shift: only 8 of 35 fall in the 95% predictive band. Adding the offset fixes this (28 of 35),
  and the estimated offsets track the observed shifts (ρ = .94).
- **d–f.** On preferred values simulated directly from the model, λ is recovered well: r = .89 per
  participant, a group μ of 0 gives 0.03 and 1 gives 1.02. The offset is recovered at r = .98 and
  does not trade off against λ (r = .09).

# 4. The decisive check: through the whole fitting pipeline

![](figures/fig_lambda_summary_pipeline.pdf)

- **Simulation.** Each voxel keeps its real tuning width, amplitude and baseline and its real
  Misaligned preferred value. Its Aligned preferred value is set by a known λ. We simulate betas and
  refit them exactly as the real data.
- **Low noise:** this works. With Gaussian noise at each voxel's residual SD, λ = 0 gives
  μ = 0.00 and λ = 1 gives 1.00 (**e**). But the simulated preferred values then scatter far less than
  the real ones (2.5 vs 9.5 CHF; **a–b**).
- **Realistic noise:** it breaks. At three times that noise the scatter matches the data
  (9.8 CHF; **c–d**), and **λ = 0 is recovered as μ = 1.82 and λ = 1 as 1.84**. Half the runs gives
  the same upward bias, which explains why the observed odd and even halves both land near 1.7.
- **Why.** Noisy preferred values bunch in the middle of the value range, and that looks like the
  quantile code's plateau near 22 CHF.

# 5. Individual differences

![](figures/fig_lambda_summary_individual.pdf)

- **a.** Per-participant λ from odd runs and from even runs agrees (ρ = .65, IPS). It doesn't track
  signal strength (ρ = .07) or how much the preferred values scatter (ρ = .11). What it *does*
  measure is unclear, given section 4.
- **b–c.** λ is unrelated to behavioural λ (ρ = .02) and to the cognitive model's value noise
  σ_rep (ρ = −.13).

# 6. What this means and what next

- Don't report the hierarchical λ from preferred values as evidence for quantile coding. At our
  noise level its group estimate isn't informative, and the half-run "reliability" comes from a
  regime where the estimator is biased.
- The same simulated datasets are a ready-made test bed. Next, run the cross-validated cvR²-profile
  λ̂ (prediction-based, so it may not suffer this bias) on them. If it recovers λ = 0 vs 1 at noise ×3,
  that becomes the primary analysis.
- The simulations use independent Gaussian noise. Real noise is structured over runs and sessions,
  and part of the observed scatter may be real. The noise ×3 condition matches the scatter, not
  necessarily the mechanism behind it.

*Code:* `abstract_values/encoding_models/{hierarchical_lambda, simulate_lambda_pipeline}.py`,
`fit_aprf.py --runs`, `visualize/{collect_session_modes, figure_hierlambda_ppc, figure_lambda_summary}.py`.
*Data:* `notes/data/hierlambda_default/`, `notes/data/hierlambda_offset_recovery.tsv`.
