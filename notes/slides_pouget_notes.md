# Slides for Alexandre Pouget (22 Oct 2026): speaker notes

Deck: `/Users/gdehol/git/abstract_values/notes/figures/slides_pouget.pdf` (+ `slides_pouget_pNN.png`),
built by `abstract_values/visualize/slides_pouget.py` from Bedi et al. (2026) Figs 1–4
(`notes/papers/bedi_et_al2026.pdf`), `fig_summary.pdf`, `fig_connectopy.pdf`,
`sim_v1_ips_mapping.pdf`, `fig_model_comparison.pdf`, `fig_coupling_components.pdf`,
`fig_gradient.pdf` and `fig_wrap_behavior.pdf`.

Naming bridge: the paper's **Misaligned** mapping is the repo's **cdf**; **Aligned** is **inverse cdf**.
The paper used a between-subject design with three mappings (Uniform, Aligned, Misaligned); the fMRI
study is within-subject with Aligned and Misaligned only, one per session.

## Part 1: the question and the behavioural logic (Bedi et al. 2026)

1. **Title.** Where does efficient coding act when value is constructed from perception? The coupling
   maps are the hook for the second half.
2. **Value is constructed.** The same perceptual feature has different value under different goals
   (banana: eat vs store). Value = G(percept). Changing G reshapes the value prior with the perceptual
   prior fixed: p(v) = p(s)|ds/dv|, so values are dense where G is shallow. Both stages are noisy and
   resource-limited; three architectures: efficient coding in perception, in valuation, or both.
3. **Task.** Learn an orientation-to-CHF mapping (observation, then feedback), then bid on low-contrast
   (2%) Gabors without feedback. BDM auction: bidding the true subjective value is optimal, so the bid
   is a trial-by-trial readout of the internal value estimate.
4. **The trick.** Orientations are sampled uniformly in every condition; the long-term natural prior
   over-represents cardinals. Three mappings change only the value prior: Uniform (linear), Aligned
   (dense values at cardinals, agreeing with the perceptual prior), Misaligned (dense at obliques).
5. **Predictions.** Efficient perception: the bias keeps its sign across mappings (only Jacobian
   effects). Efficient valuation: Aligned and Misaligned give opposite biases, Uniform none. Multi-stage:
   reversal plus asymmetry (amplified when the priors agree, damped when they conflict) and a residual
   bias under Uniform.
6. **Result.** n = 119, preregistered. The bias around 90° reverses between Aligned and Misaligned
   (slope divergence t = 27) and is asymmetric, with persistent bias under Uniform: efficient coding
   at both stages. Perception stays anchored to the long-term prior; value adapts to the context.
   Bridge: so where is the value code in the brain, and how does it follow a mapping that changes?

## Part 2: the fMRI study

7. **fMRI design.** 35 participants, two sessions, one mapping each (cdf = Misaligned, values bunch at
   12 and 32 CHF; inverse cdf = Aligned, at 2, 22, 42 CHF). Single-trial GLMsingle betas, 23
   orientations × 8 runs per session.
8. **Behaviour replicates.** Bias and spread swing with the mapping. Two-stage model plus a category
   gate at 90° / 22 CHF (ΔELPD = 270). Open: per participant, the perceptual vs value noise split is
   weakly identified from bids alone.
9. **Hypotheses.** A: parietal cortex just carries orientation, value is read out downstream. B: a
   parietal value code with fixed labels, fed by re-routed V1 input (the toy model shown). C: parietal
   neurons re-allocate their tuning to the session's value prior (efficient code in the tuning).
10. **Encoding models.** Orientation-tuned fits dominate early visual cortex, value-tuned fits light up
    parietal cortex; preferred value maps are smooth along IPS.
11. **Geometry: IPS codes value, not orientation.** Crossnobis RDMs: V1 is a circle (7.5° and 172.5°
    are neighbours), IPS an ordered line (they are the far ends). An orientation code has no reason for
    an edge at 0°/180°; the value axis does (42 CHF sits next to 2 CHF). The IPS axis is cut where the
    task puts it, not where the stimulus does. The geometry over orientations is the same in both
    sessions, which is exactly what a quantile (efficient) value code predicts, since quantile = rank.
12. **The catch (theory).** Uniform orientations and monotone mappings make the value quantile equal to
    the orientation rank in both sessions. Any uniform orientation code has Fisher information about
    value ∝ p(v)². So efficient-coding signatures alone cannot tell A from B. What does separate them is
    wrap-around: a sensory orientation code is a circle, a value code is a line.
13. **Model comparison** (`fig_model_comparison.pdf`). cvR² over the null, all ROI voxels, n = 35, models
    matched in parameter count. Circle vs line flips along the hierarchy: V1–3 better as a circle
    (+0.11 to +0.22 pp), IPS1–2 as a line (−0.17 pp, p = .003); IPS vs V1–3 t(34) = −4.7, p < .001.
    Preferred values follow the session's value distribution (quantile) better than fixed CHF in
    V1–V3, IPS1–2 (+0.32, p = .01) and IPS3–5 (+0.31, p = .03): an efficient value code in the tuning
    itself (C). A free preferred value per session overfits (−0.6 pp). Fair, model-neutral voxel
    selection (retinonumeral permutation FDR) is running.
14. **What B predicts.** Re-routing alone gives an efficient value code (FI ∝ p(v)^1.6–1.9, ideal 2;
    IPS stays bell-shaped in value). Its unique prediction: the V1 orientation an IPS unit listens to
    moves with the mapping, while its preferred value stays.
15. **Coupling maps.** Residual trial-to-trial coupling (stimulus means removed) between parietal
    voxels, binned by preferred value, and V1 orientation channels. The ridge follows the mapping in
    force. Session-invariant coupling cancels in the score, so A predicts zero.
16. **Statistics.** IPS0–5 +0.013 ± 0.003, p = .0001 with neighbouring trials removed (n = 35); IPS0
    n.s., IPS3–5 +0.020. The mapping can be read out from a session's coupling map in ~60% of sessions.
17. **Robustness.** NPC1–3 (n = 35): +0.012, p = .0002 at ±1 lag; flat across NPC1–3 (NPC overlaps the
    Wang IPS maps, so not independent). Full value-basis tuning curve as template: a bit weaker but
    significant in every ROI at ±1 lag, including IPS0 (+0.011, p = .009). Orientation-winning voxels
    also route (IPS0–5 +0.010, p = .025).
18. **Two components** (`fig_coupling_components.pdf`). Both sessions' coupling profiles fitted at once
    with a stable template (the voxel's quantile/orientation-rank tuning) and one that moves with the
    mapping. Stable part a = +0.037, t(34) = 6.4; it survives per-trial gain removal (+0.035, t = 5.8)
    and is there on low-drive trials (+0.034, t = 3.9), so it is not shared gain. Moving part b: +0.016,
    p = .02 on strongly driven trials, n.s. over all trials (+0.006) and on low-drive trials; it tracks
    the original score across people (r = .43). Neither component relates to a voxel's tuning
    preference. Key point: with uniform orientations and a monotone map, fixed wiring from orientation
    rank already yields the efficient (quantile) value code, so the stable link is what C needs.
19. **Brain and model stages** (`fig_gradient.pdf`). Code strength vs the cognitive model's precision
    per stage (partial r, sequential fit, n = 35): V1 orientation code with perceptual precision +0.40;
    IPS0/IPS1–2/IPS3–5 value code with value precision +0.39/+0.32/+0.31. Inside the cognitive model as
    covariates, a stronger IPS value code means less value noise in both fit variants (+1.85 [0.66,
    3.07] and +1.73 [0.25, 3.09] on the logit-share scale). The V1 half depends on how the fit splits
    the noise (the split is weakly identified per person).
20. **Code shape and bid precision** (`fig_wrap_behavior.pdf`). Wrap index (circle minus line fit) vs
    the model's total bid noise: precise bidders have a circular code in V1–3 (ρ ≈ −0.4) and a
    line-like code in IPS0–2 (ρ ≈ +0.4 to +0.5); the sign flips between V3 and V3a/IPS0. Behaviour
    already says the readout doesn't wrap (1° is never bid as 42 CHF). Who routes more has less value
    noise (IPS0–5 ±1 lag, ρ = −0.38, p = .02). Uncorrected, n = 35; the structured sign flip is the
    point.
21. **Summary and questions.** V1 codes orientation on a circle; IPS codes value on a line that tracks
    the value distribution, and its strength and shape predict value-stage precision; V1 to IPS is a
    stable orientation-rank link plus routing that follows the mapping on strongly driven trials.
    Questions: quantile value = orientation rank by design, so which experiment separates an efficient
    value code from a relabelled sensory axis (non-uniform orientation sampling, non-monotone mapping;
    the bauer models handle both, except the categorical gate for a non-monotone map)? Re-routing vs a
    fixed population with a flexible readout? Should routing show up in information-limiting noise
    correlations?
