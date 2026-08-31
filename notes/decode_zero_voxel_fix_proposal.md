# Proposal — make "no voxel survives the cvR² gate" a recorded outcome, not a crash

**Status:** proposal, nothing implemented.
**Date:** 2026-08-31.
**Scope:** `abstract_values/encoding_models/decode_gabor.py`, `decode_value.py`,
`abstract_values/snakemake/Snakefile`.

## The condition

In the `n_voxels=0` arm ("use every voxel that generalises"), each outer fold runs a
nested leave-one-run-out CV inside its training runs and keeps the voxels with
`cv_r2 > 0`. For a small ROI on a weak subject, that set can be **empty**. There is
nothing pathological about it — the encoding model simply explains no out-of-sample
variance in that ROI for that subject.

## What the code does today — five call sites, three different answers

| File | Model path | Lines | Behaviour when `len(sel) == 0` |
|---|---|---|---|
| `decode_gabor.py` | linear (`_run_linear_folds`) | 221–228 | falls back to **top-1 voxel** by cv-R² |
| `decode_gabor.py` | vonmises (main loop) | 483–493 | **`raise SystemExit`** → non-zero exit, no output |
| `decode_value.py` | weighted | 252–254 | **no guard** — empty `sel` flows into `ResidualFitter` |
| `decode_value.py` | linear | 407–418 | falls back to **top-1 voxel** by cv-R² |
| `decode_value.py` | aprf | 737–739 | **no guard** |

Only the `decode_value.py` linear path carries a comment explaining the choice. The
FDR / `p_signal` arms are unaffected — they already fall back to `fdr_fallback_n_voxels`.

Each of the three behaviours is wrong in its own way:

- **`SystemExit`** is scientifically honest but breaks the pipeline's bookkeeping. No
  `.done` sentinel is written, so Snakemake re-plans the job on *every* driver
  generation, the DAG never reaches `REMAINING = 0`, and `run_driver.sh` always ends
  the chain on its "stalled" guard instead of on completion. The 8 affected cells have
  already been resubmitted 4–6 times each (~40 wasted jobs and counting).
- **top-1 fallback** decodes from a single voxel whose *out-of-sample* R² is ≤ 0 — i.e.
  from noise — and writes a posterior that is indistinguishable downstream from a real
  one. `_meta.tsv` records `n_voxels_selected = 1`, which reads identically to "exactly
  one voxel legitimately passed". This is the dangerous one: it silently enters group
  analyses.
- **no guard** crashes inside `ResidualFitter` with an opaque error, or produces
  undefined behaviour.

## Blast radius today (verified on the cluster, 2026-08-31)

Exactly 8 missing `nv-0` decode_gabor sentinels — 4 (subject, ROI, smoothing)
combinations × 2 λ:

| Subject | ROI | Smoothing |
|---|---|---|
| sub-16 | BensonV1 | unsmoothed |
| sub-19 | NPCr | smoothed |
| sub-23 | BensonV1 | smoothed |
| sub-28 | NPCr | smoothed |

λ is irrelevant to the failure — voxel selection happens before the noise model — so
the λ sweep exactly doubles the wasted jobs. All `nv-0` decode_value cells currently
pass (the two missing ones are simply still running).

## Proposed fix

### A. One shared selector

New `abstract_values/encoding_models/voxel_selection.py` exposing a single function
used by all five call sites:

```python
def select_voxels(cv_r2, *, n_voxels, fdr_alpha=None, p_signal_thr=None,
                  fdr_fallback_n_voxels=..., ...) -> tuple[pd.Index, str]:
    """Return (selected voxel index, status).

    status ∈ {'ok', 'fdr_fallback', 'mixture_degenerate', 'empty'}
    """
```

This removes ~60 lines of near-duplicated selection logic per call site and makes the
three divergent behaviours impossible to reintroduce.

### B. "Empty" becomes data, not an exit code

When `status == 'empty'` for a fold: skip that fold, emit no posterior rows for it, and
record it. The script still writes **both** output files and **exits 0**.

`_meta.tsv` gains a `status` column alongside the existing `n_voxels_selected`:

```
session  run  n_voxels_selected  status
1        1    412                ok
1        2    0                  empty
```

If *every* fold is empty, still write a header-only `_pars.tsv` and a full `_meta.tsv`,
print a loud warning, and exit 0. The sentinel gets touched, the DAG completes, and the
fact that nothing was decodable is visible in the data rather than in a SLURM exit code.

### C. Downstream refuses to average empty folds

`_meta.tsv` is already read by `visualize/compare_decoding_selection.py`,
`visualize/check_voxel_count_sweep.py` and `encoding_models/notebooks/decode_hyperparams.ipynb`.
Teach those to filter on `status == 'ok'` and to drop — not silently mean-over — a
(subject, ROI) cell whose folds are all empty. Without part C, part B just moves the
problem downstream.

### D. Ship the Snakefile runtime asymmetry with it

`_decode_runtime` is defined immediately above `rule decode_gabor` and its own docstring
documents nv=0 decodes hitting a 4 h cap — but only `rule decode_value` uses it;
`rule decode_gabor` hardcodes `runtime=60`. The asymmetry looks unintended. It is not
biting today only because these 8 jobs die in 20–100 s; once they run to completion the
1 h limit becomes live. One-line change:

```python
resources: cpus_per_task=4, mem_mb=16_000, runtime=_decode_runtime,
```

## Alternatives considered

| Option | Why not |
|---|---|
| Mark the rule allow-fail / `touch` the sentinel on failure in Snakemake | The DAG completes, but downstream still cannot distinguish "empty result" from "never ran". Hides the condition from the data. |
| Adopt the top-1 fallback everywhere (cheapest) | Knowingly decodes from a voxel with out-of-sample R² ≤ 0. Acceptable *only* if labelled `status='fallback_top1'` so it can be excluded — which is part B anyway. |
| Drop `nv=0` from `decode_n_voxels` for the affected cells | Treats the symptom, and `nv=0` is the one arm with no arbitrary voxel-count choice. Worth keeping. |
| Relax the gate (e.g. `cv_r2 > -0.05`) | Changes the scientific criterion to make a bookkeeping problem go away. |

## Suggested order

1. **D** alone (one line, no behaviour change) — safe to land immediately.
2. **A + B** together, with a parameter-recovery-free smoke test: run one known-empty
   cell (sub-28 NPCr smoothed nv=0 λ=0.0) and confirm it exits 0, writes both files, and
   the meta marks every empty fold.
3. **C**, then re-run the 8 cells and confirm the driver chain reaches `REMAINING = 0`
   for the first time.
