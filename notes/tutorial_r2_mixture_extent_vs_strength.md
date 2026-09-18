# More area, or stronger signal? An R² mixture tutorial

*For Maike (and her Claude). Context: nPRF fits in 35 dyscalculics and 35 controls;
more vertices survive threshold in controls; is that extent or amplitude?*

**Don't roll your own mixture.** `braincoder` has the machinery, a rendered example,
and a 4 MB downloadable *numerosity* demo dataset to test it on (§2). A hand-written
2-component GMM is easy to get subtly and unfixably wrong (§5).

---

## 1. The idea

"Vertices with R² > t" confounds **extent** (how much cortex is tuned) with
**strength** (how well the nPRF explains the tuned vertices) — and with plain
noisiness, which shifts the whole R² distribution. Sweeping `t` doesn't help: a
lower-amplitude and a lower-prevalence distribution both make the count fall off
faster.

So model the R² distribution instead, as two components — *noise* (untuned
vertices) and *signal* (tuned ones) — fit on `logit(R²) = log(R²/(1−R²))`. Per
subject, with no threshold anywhere:

| Parameter | | Reads as |
|---|---|---|
| `signal_weight` | w_s | **Extent** — fraction of the search space that is tuned |
| `signal_mu` | μ_s | **Strength** — typical logit-R² of a tuned vertex |
| `signal_sigma` | σ_s | Heterogeneity of the tuned population |
| `noise_mu` | μ_n | **Noise floor** — a data-quality readout (§7) |

`w_s × N_vertices` is a threshold-free vertex count and `μ_s` an amplitude, fit
jointly so neither absorbs the other. Two clean comparisons instead of one
ambiguous one.

## 2. The infrastructure

<https://github.com/Gilles86/braincoder> · docs <https://braincoder-devs.github.io/> ·
cite [10.5281/zenodo.10778413](https://doi.org/10.5281/zenodo.10778413)

```bash
pip install git+https://github.com/Gilles86/braincoder.git   # mixture needs only numpy/sklearn/scipy
```

```python
from braincoder.utils.stats import (
    fit_r2_mixture, r2_posterior_signal, r2_fdr_threshold,
    r2_p_signal_threshold, plot_r2_mixture, get_rsq)
```

**Use `fit_r2_mixture` (logit-Gaussian), from `main`.** Two things to know:

- The module is `braincoder.utils.stats`, not `braincoder.stats`.
- `fit_r2_f_beta_mixture` exists, but only on the **unmerged** branch
  `feature/f-beta-r2-mixture` (last touched 2026-05-15; `main` moved on without it,
  and its cleaned-up `stats.py` has no F+Beta). That branch's own module docstring
  says: *"**Default to** `fit_r2_mixture` **(logit-Gaussian)** unless you have a
  specific reason to prefer F+Beta."* §5 shows what goes wrong when you don't.

**Run the existing example first:**
`examples/01_decoding_pipeline/01_voxel_selection_fdr.py` — fit, three threshold
flavours, diagnostic plot, apply in ROI. It runs on a ~4 MB auto-downloaded
**numerosity** extract from Prat-Carrabin, de Hollander, Bedi, Gershman & Ruff
(2025), [10.1101/2025.09.25.675916](https://doi.org/10.1101/2025.09.25.675916):

```python
from braincoder.utils.data import load_pratcarrabin2025_npc
bundle = load_pratcarrabin2025_npc()
bundle['r2_wholebrain']  # within-sample R² → the mixture
bundle['cv_r2']          # cross-validated R² (NPCr) → reporting
```

One minute, no data on disk, and it shows you what the mixture actually looks like
on numerosity nPRF R² — which is not the clean textbook case (§6).

## 3. What to feed it

- **One fixed anatomical ROI**, same in both groups (fsaverage IPS/parietal labels).
  Never a functionally selected space. `w_s` is diluted by how generous the ROI is,
  so always report it next to the ROI vertex count.
- **In-sample R², not cvR².** The mixture needs R² ∈ (0,1); `fit_r2_mixture` drops
  everything outside `(0, 0.99)`, and cvR² is negative often enough to gut the noise
  component. Keep cvR² for generalisation reporting and the check in §7.
- **Per subject, then a group test on the parameters.** Never pool vertices across
  subjects into one mixture.

The braincoder example fits whole-brain and only *applies* the threshold in the ROI
(it needs plenty of genuine-noise voxels to anchor the noise component). Your
quantity of interest is ROI prevalence, and a surface IPS ROI has thousands of
vertices — so fit in-ROI, but fit whole-brain too: that gives you a noise component
estimated where you're confident nothing is numerosity-tuned, which §6 wants.

## 4. The per-subject loop

```python
import numpy as np, pandas as pd
from braincoder.utils.stats import fit_r2_mixture

def _inv_logit(z): return 1.0 / (1.0 + np.exp(-z))

rows = []
for subject, group in cohort:                       # 70 subjects
    r2 = load_r2_in_roi(subject)                    # 1-D over ROI vertices
    r2 = r2[np.isfinite(r2)]
    try:
        fit = fit_r2_mixture(r2, n_init=8, seed=0)  # needs ≥50 values in (0, 0.99)
    except ValueError as e:
        print(f"{subject}: {e}"); continue
    fit["delta_mu"]      = fit["signal_mu"] - fit["noise_mu"]     # separability
    fit["sigma_ratio"]   = fit["signal_sigma"] / fit["noise_sigma"]
    fit["noise_tail_r2"] = _inv_logit(fit["noise_mu"] + 2 * fit["noise_sigma"])
    rows.append(dict(subject=subject, group=group, n_vertices=len(r2),
                     n_tuned=fit["signal_weight"] * len(r2),      # extent, no threshold
                     **fit))

df = pd.DataFrame(rows).to_csv("r2_mixture_per_subject.tsv", sep="\t", index=False)
```

`n_init=8` is the number of **EM restarts** (sklearn's default is 1). Mixture
likelihoods are non-convex, so a single start can land in a poor local optimum —
and, worse, *different subjects can land in different optima*, manufacturing group
differences out of optimiser noise. Keep 8, fix `seed`, and see the stability check
in §5.

Non-equal-area vertices (fsnative)? Weight by midthickness vertex area and report
mm², or `n_tuned` inherits mesh-density differences between subjects.

## 5. The particulars are not details

**A 2-component GMM never fails.** Hand it any unimodal right-skewed distribution —
which is exactly what an R² map is — and EM returns two components, weights summing
to 1, and a `signal_weight` you can put in a table. Fitting it is not evidence that
signal exists. Everything therefore rides on specification choices:

**Which component is "signal".** sklearn returns components in arbitrary order, and
the order changes across subjects and seeds. `fit_r2_mixture` fixes this by
construction: `argmin(means)` = noise, `argmax(means)` = signal. Without that
convention you silently swap labels for some subjects, and your group test on
"signal weight" is partly a test on noise weight.

**w_s > w_n is a red flag.** In a whole-brain or generous ROI, most cortex is not
numerosity-tuned, so a signal component outweighing the noise component means one
of: labels swapped (above); the two Gaussians are carving up a single skewed blob
(below); or the ROI is so tight that "prevalence" isn't a meaningful quantity in it.
Check `signal_weight` against `noise_weight` in every fit, and against the
whole-brain fit for the same subject.

**The logit is load-bearing.** On raw R², two Gaussians fit the *skew* of one
distribution and you learn nothing; a Beta mixture on raw R² can go pathologically
U-shaped. The logit stretches the near-zero region where noise vertices pile up so
the two populations can separate at all, and keeps both components Gaussian. A
mixture fit on raw R², z-scores or −log10(p) is a different model whose parameters
don't mean the same thing.

**The support filter moves μ_n.** `(0, 0.99)` is a choice; include R² ≤ 0, or trim
the tail elsewhere, and the noise component shifts. Whatever you pick must be
identical for all 70 subjects.

**The F+Beta trap (`fit_r2_f_beta_mixture`).** Anchoring the noise to the null F
distribution is the right *instinct*, but the implementation fixes only the
numerator dof (`d1_noise`); the denominator dof `d2` is fit freely by EM. It runs
to whatever makes the noise component narrowest, and the signal Beta then swallows
the whole distribution. Signature, from real numerosity fits: whole-brain
w_signal = 0.98 / 0.97 / 0.64 across three subjects; FDR thresholds of R² = 0.000,
i.e. every voxel passes; fitted `d2` ranging 276–5284 across panels of one dataset
when it should be ≈ n−k and near-constant; whole brain coming out *less* noisy than
NPC_R. Those w_signal values swing 20-fold between subjects in a quantity that ought
to be comparable — a group test on them compares fit instability, not biology. Use
the logit-Gaussian; and if you want a model-anchored null, get it empirically as in
§6.2 rather than parametrically.

*Would freeing both dofs help?* No — it's strictly worse, and it's already the
default. `R² ~ Beta(d1/2, d2/2)` is the same two-parameter family as
`F ~ F(d1, d2)`, so "both dofs free" **is** a free Beta; the function's own
docstring says so: *"``None`` (default) leaves both noise dofs free — in which case
the noise component is just a free Beta and the F-labelling is cosmetic."* That's
the all-free 2-Beta the `main` docstring explicitly warns against. The fit is
already under-determined with one dof free; freeing the second gives you two
2-parameter families competing to fit one skewed blob, distinguished only by their
weight. **The fix runs the other way — fix *both* dofs**, so the noise component has
zero free parameters. The reason that wasn't done: for a *nonlinear* nPRF fit by
grid search + gradient descent, and with autocorrelated residuals, the nominal
`F(k, n−k)` null is wrong — the true null is wider. So calibrate it once: estimate
the effective `(d1, d2)` from a permutation null on two or three subjects, then
**fix those values for the whole cohort**. Refitting `d2` per subject per ROI, as in
the figure above, is what lets it absorb everything and destroys comparability.

**Nothing anchors the noise component** in the plain logit-Gaussian fit either. A
genuine fit from `abstract_values` (sub-08, whole brain, 84k voxels):

```
noise:   μ = -4.446  (R² = 0.0116),  σ = 0.421,  w = 0.507
signal:  μ = -3.878  (R² = 0.0203),  σ = 0.450,  w = 0.493
```

Nothing errors. But the means are 0.57 logit units apart with σ ≈ 0.43 each, and the
weights are 50/50 — that is not two populations, it is one skewed distribution split
down the middle. "49% of the brain is tuned" read off that is an artifact of the
parameterisation. Guard rails:

```python
assert fit["signal_mu"] > fit["noise_mu"]                        # labelling
if fit["signal_weight"] > fit["noise_weight"]: warn("w_s > w_n") # §5
if fit["delta_mu"] < 0.5 or fit["sigma_ratio"] > 1.4:            # separability
    warn("components not separated — see §6")
```

(Those cut-offs come from `abstract_values`, tuned on volumetric whole-brain data.
Look at your own distribution of `delta_mu` before trusting the numbers.)

**Seed-stability check.** Refit each subject with `seed=0..9` and confirm `w_s`
varies by less than the group difference you want to claim. Cheap, and decisive if
it fails.

## 6. Expect overlap — and make it identifiable

The braincoder numerosity example is honest that on real data the signal mode is
**not** well separated: the posterior threshold caps out near P ≈ 0.85 and tail-FDR
only converges at permissive α, so `r2_p_signal_threshold(fit, p=0.95)` and
`r2_fdr_threshold` return `inf`. Always check `np.isfinite(thr)`. Its operating point
is the **noise-tail quantile**, `_inv_logit(noise_mu + 2*noise_sigma)` — "R² unlikely
under noise alone" — which stays finite and selective under overlap. Use it.

For your question, overlap means w_s and μ_s **trade off**: the same histogram fits
about as well as "few, strong" or "many, weak", so the extent/strength split is
weakly identified exactly where you want to use it. Two remedies:

1. **Bootstrap vertices** within subject (500 resamples, refit) for CIs on w_s and
   μ_s, and check their correlation across draws. Strongly anti-correlated ⇒ not
   identified for that subject; say so rather than testing it.
2. **Pin the noise component externally**, in whichever of these you need:

   a. **Free: use your own whole-brain fit.** Most of the brain is not
      numerosity-tuned, so the whole-brain mixture's noise component is already a
      good estimate of the null. Carry its μ_n, σ_n into the ROI fit and leave only
      w_s, μ_s, σ_s free. You are computing the whole-brain fit anyway — this costs
      nothing and removes most of the trade-off.
   b. **Permutation null**, if (a) still leaves w_s and μ_s trading off. Refit the
      nPRF on permuted trial→numerosity assignments and take μ_n, σ_n from the
      resulting R² distribution. Note one permutation already gives you *N_vertices*
      draws from the pooled null, so it pins μ_n/σ_n well — but it is a single draw
      of the *permutation*, and permutations differ in how much design structure they
      leave behind. Run ~10–20 and pool them; check the pooled null is stable across
      them. `ParameterFitter` + `get_rsq` make each refit cheap, but budget for
      70 × 10–20 fits.

## 7. The noise floor will get you in review

In-sample R² for a k-parameter nPRF is inflated by noise: a noisier subject overfits
*more*, and one with fewer usable runs more still. If dyscalculics moved more or lost
more runs, μ_n and μ_s both shift and your "group effect" is data quality.

- **Test `noise_mu` between groups first.** If it differs, fix the design (match
  runs/TRs, covary motion) before interpreting μ_s.
- Report mean FD and n_runs per group in the same table.
- **Cross-validated check**, independent of the mixture: count vertices where
  cvR²(nPRF) > cvR²(null), leave-one-run-out. Immune to the overfitting bias; should
  move with w_s. If they disagree, believe the cross-validated one.
- Page all 70 `plot_r2_mixture(fit, r2=r2, threshold=thr)` panels into one PDF and
  actually look at them.

## 8. The group test

Two tests on 70 numbers each, both on the **logit scale** (where the Gaussians live
and variance is homogeneous). Back-transform only for reporting.

```python
df["logit_w"] = np.log(df.signal_weight / (1 - df.signal_weight))
smf.ols("logit_w   ~ group + mean_fd + n_runs + noise_mu", data=df).fit().summary()
smf.ols("signal_mu ~ group + mean_fd + n_runs + noise_mu", data=df).fit().summary()
```

| w_s | μ_s | Interpretation |
|---|---|---|
| ↓ | = | **Less area.** Same tuning quality where it exists; smaller numerotopic map. |
| = | ↓ | **Weaker signal.** As much cortex tuned, but the nPRF explains it less well. |
| ↓ | ↓ | Both — check `noise_mu` hard; a global data-quality difference does exactly this. |
| = | = | The count difference was the noise floor, not tuning. Also a result. |

Lower μ_s means *this nPRF* fits worse — which could be noise, or tuning that is real
but shaped differently (broader σ, shifted preferred numerosity). Follow up by
comparing fitted nPRF parameters in vertices both groups agree are tuned, rather than
concluding "weaker representation".

## 9. Reporting, and pointers

Report per group: `signal_weight` (% of ROI), `signal_mean_r2`, `noise_mean_r2`, mean
`delta_mu`, the seed/bootstrap stability from §5–6, the two regressions, and the
cvR² > cvR²_null count as convergent evidence. For figures, `r2_posterior_signal(r2,
fit)` gives a continuous per-vertex P(signal | R²) map — better than a binary blob,
and it doesn't commit you to a cut. If you do need a cut, make it subject-specific
(§6): a fixed R² threshold gives every subject a different false-positive rate, driven
by the noise floor.

- braincoder: `examples/01_decoding_pipeline/01_voxel_selection_fdr.py`,
  `braincoder/utils/stats.py`, `braincoder/utils/data.py::load_pratcarrabin2025_npc`,
  `docs/tutorial/lesson1–7`
- abstract_values (production wrapper — operational reference, volumetric/whole-brain,
  don't copy directly): `encoding_models/compute_r2_mixture.py`,
  `visualize/check_r2_mixture.py`
