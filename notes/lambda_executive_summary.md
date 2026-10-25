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

**Bottom line.** Two ways to estimate λ were put through the same end-to-end simulation (real
voxel fits, known λ, refit).

- **Hierarchical λ from fitted preferred values fails.** It works post hoc on the per-voxel fits. At
  realistic noise it returns λ ≈ 1.8 whether the truth is 0 or 1, so its IPS estimate
  (0.93 [−0.25, 1.90]) is uninformative (sections 2–5).
- **The cvR² λ-profile works.** It fits the coding model directly to the betas, with one λ per
  participant and ROI, cross-validated over runs. It separates the codes at every noise level: λ = 0
  stays at about 0, and λ = 1 is recovered as 0.4–0.7, so its bias is conservative, towards the value
  code. The observed IPS profile peaks at 0.75–0.95, beyond the simulated quantile code (section 6).
- **Caveat.** Here the quantile is orientation rank, so "λ ≈ 1" means tuning that follows orientation
  rank rather than CHF.
- **Individual differences.** Per participant, the profile λ̂ is too noisy, and the hierarchical λ is
  unrelated to behaviour.

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

# 4. End-to-end check of the preferred-value route

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

# 6. Fitting the model directly to the voxel data: the cvR² λ-profile

![](figures/fig_lambda_summary_profile.pdf)

- **Method.** The same coding model is fitted to the betas on a fine λ grid: a Gaussian on the mix
  axis x_λ(v), shape chosen per voxel on the training runs, leave-one-run-out, all ROI voxels. λ̂ is
  the peak of the participant's ROI-mean cvR² profile. This is the analogue of the neural_priors
  approach, where the condition effect is one parameter shared across a participant's voxels and is
  judged by cross-validated fit, never read off per-voxel estimates.
- **Tested on the same simulated betas as section 4.** λ = 0 gives a group profile peak at 0–0.1
  (noise ×1–2) and −0.15 to −0.45 (×3); per-participant medians are near 0. λ = 1 gives peaks at
  0.55–0.70 (×1) and 0.30–0.70 (×3). Noise lowers the estimate rather than raising it.
- **Which noise level is realistic.** The observed ROI-mean cvR² (−0.7 to −2.8 × 10⁻³, all
  voxels) lies between the ×1 and ×2 simulations. That differs from the ×3 that matched the scatter
  of preferred values.
- **Observed** peaks: IPS0 0.75, IPS1–2 0.95, IPS3–5 0.95 (V3a/b 0.85, hV4 0.70). These are above
  every λ = 1 simulation, and the codes are well separated at this noise level.
- **Why the simulated λ = 1 lands below 1.** Only the preferred value follows the quantile there;
  the tuning width stays fixed in CHF, so it is not a pure quantile code.

# 7. What this means and what next

- **Lead with the cvR² λ-profile:** a direct, validated fit. The group result is that IPS tuning
  follows the value's quantile (equivalently, orientation rank) at least as much as a simulated
  quantile code does.
- **Drop the hierarchical λ from preferred values** as an estimator. It can stay as an
  illustration of what goes wrong.
- **Next: a closer neural_priors analogue.** Fit the log-Gaussian aPRF in braincoder on the mix
  axis, with λ (and optionally a width scale) shared across a participant's voxels and fitted by
  gradient descent. Then compare fixed λ = 0, fixed λ = 1 and shared λ by cvR², and recover it on
  these simulations.
- **The confound remains.** With uniform orientations and monotone mappings, a quantile code
  cannot be told apart from orientation coding.

*Code:* `abstract_values/encoding_models/{hierarchical_lambda, simulate_lambda_pipeline, fit_lambda_profile}.py`,
`fit_aprf.py --runs`, `visualize/{collect_session_modes, figure_hierlambda_ppc, figure_lambda_summary}.py`.
*Data:* `notes/data/hierlambda_default/`, `notes/data/hierlambda_offset_recovery.tsv`,
`notes/data/lambda_profile_sim_{group,subjects}.tsv`.
