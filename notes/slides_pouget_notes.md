# Slides for Alexandre Pouget (22 Oct 2026): speaker notes

Deck: `/Users/gdehol/git/abstract_values/notes/figures/slides_pouget.pdf` (+ `slides_pouget_pNN.png`),
built by `abstract_values/visualize/slides_pouget.py` from `fig_summary.pdf`, `fig_connectopy.pdf`
and `sim_v1_ips_mapping.pdf`.

1. **Title.** The one-line story: the codes are fixed (V1 orientation, parietal value), and what
   changes with the mapping is how parietal cortex reads V1. The coupling maps are the hook.
2. **Design.** 35 participants, 23 orientations shown equally often, BDM bids in CHF (truth-telling,
   so the bid is the value). Two mappings, one per fMRI session. Both rise with orientation; they
   differ in where values bunch up (cdf: 12 and 32 CHF; inverse cdf: 2, 22, 42 CHF).
3. **Behaviour.** Bedi et al. (2026): two-stage model, efficient coding of orientation with a
   cardinal prior, then efficient coding in value space (σ_rep), plus a category gate at 90° / 22
   CHF (ΔELPD = 270). Open problem: per participant, the split between perceptual and value noise is
   weakly identified (not by the model, the model-free slope test, or V1/IPS decoding).
4. **Encoding models.** Orientation-tuned (von Mises) fits win in early visual cortex, value-tuned
   (aPRF) fits light up parietal cortex. Preferred-value maps are smooth along IPS.
5. **Geometry.** Crossnobis RDMs: in V1 the end orientations (7.5°, 172.5°) are neighbours (a circle);
   in parietal cortex they are the far ends (an ordered line). The parietal geometry does not re-scale
   when the mapping changes, so no evidence that parietal cortex re-allocates resolution in value.
6. **Theory, the confound.** Orientations are uniform and both mappings monotone, so the value quantile
   equals the orientation rank in both sessions. Any uniform orientation code gives Fisher information
   about value ∝ p(v)². Efficient-coding signatures, and quantile-coding models, therefore cannot tell a
   value code from an orientation code in this design. Stimulus model comparison barely helps either:
   the orientation model wins about 55–60% of vertices in V1 and in IPS alike.
7. **Toy model.** IPS units with fixed value labels; only the V1→IPS weights follow the mapping. This
   alone gives FI ∝ p(v)^1.6–1.7 (1.8–1.9 with normalised weights), with IPS still bell-shaped in value.
   Its distinguishing prediction: the V1 orientation an IPS unit listens to moves with the mapping.
8. **Coupling maps.** Residual (stimulus-removed) trial-to-trial coupling between parietal voxels,
   binned by preferred value, and V1 orientation channels. The ridge follows the mapping in force.
   Session-invariant coupling cancels in the score by construction, so a fixed orientation code
   predicts zero.
9. **Coupling statistics.** IPS0–5: +0.013 ± 0.003, p = .0001 with the neighbouring trials removed
   (n = 35); grows from IPS0 (n.s.) to IPS3–5 (+0.020). The mapping in force can be read out from a
   session's coupling map in about 60% of sessions. Gain confound: gain regression leaves +0.010
   (p = .02), but map-level agreement is no longer significant (p = .14).
10. **Individual differences.** Participants whose parietal coupling follows the mapping more closely
    have less value noise in the model (ρ = .35, p = .04; r = .48). Perceptual precision is unrelated.
11. **Questions.** How to separate value from orientation coding when efficient-coding signatures are
    identical by design; flexible routing vs a fixed population with a flexible readout; whether
    decoded posterior uncertainty can split perceptual from value noise; information-limiting
    correlations; and which tuning label (single-peak vs basis) the routing test should use.
