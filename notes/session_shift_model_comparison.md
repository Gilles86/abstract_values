# Does the session-shift model actually beat no-shift?

**Status:** open again — the "resolved, 20/20 subjects favor shift" answer
below turned out to itself be a selection-bias artifact. The best current
answer, under a selection criterion that doesn't presuppose which model
wins, is **standard (no-shift) wins**. See "Second reversal" below before
trusting anything upstream that cites the 20/20 result.

**Figure:** `notes/figures/model_comparison_selection_bias.pdf` (source
data: `notes/data/model_comparison_selection_bias.tsv`,
`abstract_values/visualize/model_comparison_selection_bias.py`) — shows
both selection criteria side by side; the conclusion visibly flips.

## The question

`shifted_preferred_value.pdf` was built around the premise that NPCr
voxels' preferred CHF value (`mode`) reorganizes between the `cdf` and
`inverse_cdf` conditions. Several indirect tests of that premise gave
conflicting signals:

- Per-voxel cross-condition correlation (`mode_cdf` vs `mode_invcdf`,
  criterion-B/null-gated voxels): weak but real — r≈0.15 (Pearson) /
  0.18 (Spearman), Fisher-z one-sample t-test **p<0.001**, sign
  consistent in 16-17/20 subjects (binomial p=0.003-0.012).
- Per-subject *group-level* shift direction (median
  `mode_invcdf - mode_cdf`): **not significant** (Wilcoxon p=0.26,
  t-test p=0.28) — roughly a 60/40 sign split across subjects.
- Session-order (subject-parity) as an alternative explanation for the
  shift: also not significant (Mann-Whitney p=0.46).

None of that directly asks the right question: does letting the mode
differ by condition actually **explain more held-out variance** than
assuming a stable mode? That's a nested model comparison, and the models
are already fit and cross-validated on disk:

- `derivatives/encoding_models/aprf.cv/` — **standard** (single shared
  mode/fwhm/amp/baseline pooled across both conditions).
- `derivatives/encoding_models/aprf-shift.cv/` — **session-shift**
  (`mode_1`, `mode_2` free; fwhm/amp/baseline shared).
- `derivatives/encoding_models/aprf-fully-shifted.cv/` — **fully-shifted**
  (all 4 params free per session).

All leave-one-run-out cvR², same procedure.

## First gotcha: forgetting to filter to signal voxels at all

`compare_voxel_selection.py`'s `_collect_all()` returns **every** NPCr
voxel per subject (e.g. 834 for sub-03), with `sel_A`/`sel_B` as boolean
*columns* — it does not pre-filter rows. The first pass at this
comparison used `df.voxel` (all rows) instead of
`df.loc[df.sel_B, 'voxel']` (the voxels that actually pass the null-gated
criterion `cvR²(shift) > cvR²(null)`).

Run over the whole ROI (~85% non-signal voxels), the answer was: standard
wins, 18/20 subjects, p=0.004 — because on non-tuned voxels any extra
parameter just overfits training-fold noise, and cross-validation punishes
that. This looked like a clean null result and was reported as one before
the bug was caught. See `feedback_unfiltered_selection_columns` memory.

## Second gotcha (bigger): the "signal voxel" selection wasn't model-neutral

Restricted to criterion-B (`cvR²(session-shift) > cvR²(null)`) voxels, the
comparison flipped hard: session-shift beat standard in 24/24 subjects
(Wilcoxon p<0.00001), and fully-shifted beat standard in 20/24 but lost to
session-shift in 18/24 (p=0.004) — a clean "some flexibility, not full"
story that looked, and was reported as, resolved.

The problem: **the selection criterion conditions on session-shift's own
cross-validated performance**, then that same statistic gets used as one
arm of the head-to-head comparison. This is a winner's-curse setup —
selecting voxels because model A scored well (even on held-out folds)
optimistically biases A's reported score on that same selected set,
relative to its true performance. Standard and fully-shifted weren't the
selection target, so they don't get the equivalent boost. The 24/24
result was real numbers but an unfair fight.

**Fix:** select on `standard`'s own cvR² vs. null instead
(`cvR²(standard) > cvR²(null)`). `standard` is the simplest model, nested
inside every shift variant, and isn't one of the models whose relative
ranking is in question in the same way — if anything this tilts the deck
*toward* standard (same winner's-curse logic now favors standard's
reported score), making it a conservative test of whether shift still
wins.

### Result under neutral (standard-gated) selection

| comparison | biased (session-shift-gated) | neutral (standard-gated) |
|---|---|---|
| session-shift vs. standard | 24/24 favor shift, p=1.2e-7 | **2/24** favor shift, p=9.1e-5 → **standard wins** |
| fully-shifted vs. standard | 20/24 favor fully-shifted, p=0.0001 | **3/24** favor fully-shifted, p<0.001 → **standard wins** |
| fully-shifted vs. session-shift | 6/24 favor fully-shifted, p=0.004 | 11/24, p=0.30 (n.s.) |

Complete reversal. Under the selection criterion least favorable to the
"shift is fake" hypothesis (biasing toward standard, not against it),
**standard is the clear, significant winner** over both shift variants.

## Where this actually leaves us

Best current read: the apparent superiority of session-shift over standard
was substantially — possibly entirely — a selection-bias artifact. The
weak-but-significant cross-condition mode correlation (r≈0.15-0.18,
Fisher-z p<0.001) was *also* computed on criterion-B (session-shift-gated)
voxels and has **not yet been re-checked** under neutral selection — do
that before trusting it further. Likely candidate for a third reversal;
don't assume it survives.

The within-condition marginal-distribution finding (preferred values look
closer to uniform coverage than density-matched or surprise-weighted) used
criterion-B selection too and should also be re-derived under
standard-gated selection before leaning on it.

## Related

- `abstract_values/visualize/compare_voxel_selection.py` — criterion
  A/B selection, `_collect_all()`.
- `abstract_values/visualize/model_comparison_selection_bias.py` — the
  formalized biased-vs-neutral comparison figure.
- `abstract_values/visualize/shifted_preferred_value.py` — main
  preferred-value figure; still defaults to criterion B (session-shift-
  gated) — **reconsider this default** given the above.
- `notes/figures/compare_voxel_selection.pdf`,
  `notes/figures/shifted_preferred_value.pdf` /`.tsv`,
  `notes/figures/model_comparison_selection_bias.pdf`.

## Next

- Re-run the cross-condition correlation (r≈0.15) and the marginal-
  distribution (uniform-coverage) analyses under standard-gated selection
  — both currently rest on the same biased criterion as the reversed
  result above.
- Extend the ladder with `fwhm-only-shift` (cv fits running) under
  **neutral** selection from the start.
- Decide whether `shifted_preferred_value.py`'s default selection
  (currently criterion B) should change to the standard-gated criterion.
