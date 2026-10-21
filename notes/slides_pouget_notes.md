# Slides for Alexandre Pouget (22 Oct 2026): speaker notes

Deck: `/Users/gdehol/git/abstract_values/notes/figures/slides_pouget.pdf` (+ `slides_pouget_pNN.png`),
built by `abstract_values/visualize/slides_pouget.py` from Bedi et al. (2026) Figs 1–4
(`notes/papers/bedi_et_al2026.pdf`), `fig_summary.pdf`, `fig_connectopy.pdf` and
`sim_v1_ips_mapping.pdf`.

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
11. **Geometry.** Crossnobis RDMs: V1 is a circle (7.5° and 172.5° are neighbours), parietal cortex an
    ordered line (they are the far ends). The parietal geometry does not re-scale with the mapping:
    against C.
12. **The catch (theory).** Uniform orientations and monotone mappings make the value quantile equal to
    the orientation rank in both sessions. Any uniform orientation code has Fisher information about
    value ∝ p(v)². So efficient-coding signatures cannot tell A from B, and the stimulus model comparison
    barely helps (orientation wins ~55–60% of vertices in V1 and in IPS alike).
13. **What B predicts.** Re-routing alone gives an efficient value code (FI ∝ p(v)^1.6–1.9, ideal 2;
    IPS stays bell-shaped in value). Its unique prediction: the V1 orientation an IPS unit listens to
    moves with the mapping, while its preferred value stays.
14. **Coupling maps.** Residual trial-to-trial coupling (stimulus means removed) between parietal
    voxels, binned by preferred value, and V1 orientation channels. The ridge follows the mapping in
    force. Session-invariant coupling cancels in the score, so A predicts zero.
15. **Statistics.** IPS0–5 +0.013 ± 0.003, p = .0001 with neighbouring trials removed (n = 35); IPS0
    n.s., IPS3–5 +0.020. The mapping can be read out from a session's coupling map in ~60% of sessions.
    Gain confound: gain regression leaves +0.010 (p = .02), but map-level agreement drops (p = .14).
16. **Robustness.** NPC1–3 (n = 35): +0.012, p = .0002 at ±1 lag; NPC1 .0135, NPC2 .0116, NPC3 .0113,
    flat rather than graded as along IPS (NPC overlaps the Wang IPS maps, so not independent). Full
    value-basis tuning curve as template: a bit weaker but significant in every ROI at ±1 lag, including
    IPS0 (+0.011, p = .009). Orientation-winning voxels also route (IPS0–5 +0.010, p = .025).
17. **Individual differences.** Participants whose parietal coupling follows the mapping more closely
    have less value noise (ρ = .35, p = .04; r = .48); perceptual precision is unrelated.
18. **Summary and questions.** Fixed codes, flexible routing. Then: testing value vs orientation coding
    when efficient-coding signatures coincide; re-routing vs fixed population + flexible readout;
    decoded uncertainty to split perceptual from value noise; noise correlations; single-peak vs
    basis tuning labels for the routing test.
