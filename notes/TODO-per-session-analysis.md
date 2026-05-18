# TODO: per-session analysis support

Currently GLMsingle + every downstream encoding/decoding/Fisher script
assumes the **subject-level aggregated** GLMsingle output at
`sub-XX/func/sub-XX_task-abstractvalue_space-T1w_desc-gabor_pe.nii.gz`.
This works for the standard "fit across all sessions jointly" mode but
makes "look at one session at a time" awkward — even though
`fit_glmsingle.py` already accepts `--sessions N`.

## What's needed

1. **`fit_glmsingle.py`** — when `--sessions N` (single), write outputs
   to the per-session path `sub-XX/ses-N/func/...` rather than (or in
   addition to) the subject-level aggregate.
2. **`fit_aprf.py` + variants** — gain a `--session N` flag; when set,
   read GLMsingle from `sub-XX/ses-N/func/...` and write outputs to
   `derivatives/encoding_models/aprf/sub-XX/ses-N/func/...`.
3. **`decode_gabor.py`, `decode_value.py`** — same `--session N` flag,
   same path-resolution helper.
4. **`compute_fisher_information*.py`** — same.

Common refactor: a `Subject.get_single_trial_estimates(sessions=...)`
that picks the right directory based on whether `sessions` is a single
int (per-session path), a list (must match aggregated path), or `None`
(aggregated path).

## Why not now

- Touches 6–8 files.
- Adds breadth at a moment when sub-07 reset + sub-08 first-run are
  the active fires.
- Daily use case ("look at one session for debugging") can be served
  ad-hoc by symlinking ses-N's GLMsingle output to the subject-level
  path temporarily.

## Why someday

- Genuine value: "did the subject learn the mapping in ses-1 already,
  before the explicit task in ses-2?"
- Session-shift modeling already exists; per-session encoding fits would
  let us compare it against fitting each session independently.

## Trigger

Revisit when:
- You actually want per-session results AND
- The `--allow-incomplete` escape hatch on `require_complete_sessions()`
  is being used more than once.
