> **RESOLVED — 2026-08-20. The premise of this report was wrong; nothing was lost at the scanner.**
>
> `Run_2` (series 17, 367 volumes) was acquired, saved and archived normally. What happened is that
> the BIDS conversion on 2026-08-10 (runner job 2440) hit a **dcm2bids output-name collision**: the
> aborted attempts shared their protocol name with their retries, so series 015 (28-vol aborted
> Run_2) *and* 017 (the real Run_2) both mapped to `..._run-2_bold`, while series 029 (5-vol aborted
> Run_6) *and* 031 (Run_6) both mapped to `..._run-6_bold`. Both pairs fell back to the generic
> `run-01`/`run-02` names and overwrote each other, leaving us with the **Run_2 images carrying the
> Run_6 sidecars** — and the true **Run_6** images silently dropped. So the run actually absent from
> our copy was Run_6, not Run_2.
>
> The give-away: our `run-01` file's sidecar claims series 29, but the conversion log records series
> 29 as only **5** volumes, while that file holds **28** — i.e. it is series 015, the aborted *Run_2*
> attempt. Likewise the 367-volume `run-02` file (sidecar: series 31) is byte-identical to the file
> the corrected conversion now delivers as `run-2` (series 17).
>
> The session was re-converted on the department share on 2026-08-20 (runner job 2473) after the two
> aborted series were removed from the study: 8 clean runs, series 13/17/19/21/27/31/33/35 →
> `run-1`…`run-8`, every sidecar correctly paired. Verified here by md5 against our existing copies
> and against the job log's `SIDECAR PAIRING` block. sub-26 has been re-ingested from that data.
>
> No action is needed from IT / the scanner team — please disregard the request below. Apologies for
> the noise; the fault was on our conversion side, not yours.

# Missing functional run: sub-26, session 1, `Run_2`

**Study:** Abstract Values (orientation→CHF value learning, BDM auction task)
**Site:** SNS Lab, Siemens MAGNETOM Cima.X, 3T
**Subject / session:** sub-26 / ses-1
**Date / time:** 2026-08-10, ~11:07–11:59 (from behavioral log timestamps)
**Protocol:** `fMRI_G2_SMS3`, 8 planned functional runs (`Run_1`…`Run_8`) + 3 fieldmaps + T1w, task `abstractvalue`

## Summary

Of the 8 functional runs the subject completed behaviorally in session 1, only **7 have a corresponding BOLD series** in the data delivered to us (locally and on the department share). `Run_2` is missing entirely — no DICOM/NIfTI series, partial or otherwise, exists for it anywhere we have access to. We'd like IT/the scanner team to check whether it exists anywhere upstream (console, PACS, export logs) and, if so, have it delivered; if not, confirm it was never successfully acquired so we can document the loss.

## Evidence

**Behavioral log** (`sourcedata/behavior/sub-26/ses-1/`) shows all 8 runs completed normally — 535 trial-log rows and exactly 367 scanner trigger pulses each, run-01 through run-08, no aborts:

| Behavioral run | Start time (`.edf` mtime) | Gap from previous |
|---|---|---|
| 01 | 11:07 | — |
| 02 | 11:17 | 10 min |
| 03 | 11:24 | 7 min |
| 04 | 11:31 | 7 min |
| 05 | 11:38 | 7 min |
| 06 | 11:46 | 8 min |
| 07 | 11:52 | 6 min |
| 08 | 11:59 | 7 min |

(Run duration is ~6.3 min; the run 01→02 gap is ~3 min longer than the rest of the session, consistent with something extra happening between behavioral runs 1 and 2.)

**Delivered MRI series** (from the JSON sidecars' `SeriesNumber`/`SeriesDescription`, sorted by series number, session 1 only):

| SeriesNumber | Description | Notes |
|---:|---|---|
| 5 | T1w | |
| 9–11 | fieldmap 1 | intended for Run_1, Run_2 |
| 13 | `fMRI_G2_SMS3_Run_1` | 367 vol, complete |
| **—** | **`fMRI_G2_SMS3_Run_2` — absent** | **no series at all, not even a partial/aborted one** |
| 19 | `fMRI_G2_SMS3_Run_3` | 367 vol, complete |
| 21 | `fMRI_G2_SMS3_Run_4` | 367 vol, complete |
| 23–25 | fieldmap 2 | intended for Run_3–6 |
| 27 | `fMRI_G2_SMS3_Run_5` | 367 vol, complete |
| 29 | `fMRI_G2_SMS3_Run_6` (attempt 1) | **aborted after 28/367 volumes** |
| 31 | `fMRI_G2_SMS3_Run_6` (attempt 2) | 367 vol, complete — successful retry |
| 33 | `fMRI_G2_SMS3_Run_7` | 367 vol, complete |
| 35 | `fMRI_G2_SMS3_Run_8` | 367 vol, complete |
| 37–39 | fieldmap 3 | intended for Run_7, Run_8 |

The series-number sequence jumps 13 → 19 (Run_1 → Run_3), skipping two increments where `Run_2` (and possibly a retry of it, by analogy with `Run_6`) should be. For comparison, **session 2 of the same subject has all 8 runs present with no gaps** (series 12, 15, 18, 21, 28, 31, 37, 40 — clean), so this is isolated to session 1.

**Cross-checked in two independent copies** — local working copy and the department SMB share's `sourcedata/mri` — file-for-file, byte-for-byte (MD5) identical, 8 functional NIfTI files in both (`Run_1, Run_3, Run_4, Run_5, Run_6×2 attempts, Run_7, Run_8`), zero discrepancy and no hidden/duplicate `Run_2` under an unexpected name. We also found a third copy on the department share (an already-BIDS-converted `data/sub-26/...` folder) — same 7-run content, just relabeled `run-1`…`run-7`; not an independent source.

Total: 7 complete runs × 367 volumes + 1 aborted run × 28 volumes = 2597 volumes converted, vs. 8 × 367 = 2936 volumes expected from the behavioral/trigger record. Deficit = exactly one full run (367 volumes).

## Why this matters downstream (already affecting existing derivatives)

Our BIDS conversion script renumbers the 7 surviving functional runs sequentially (`run-1`…`run-7`) by acquisition order, since it has no way to know a run is missing rather than simply not-yet-acquired. This silently shifts every run from `Run_3` onward down by one slot in our file naming, which has two concrete consequences:

1. **Fieldmap misassignment**, already baked into the existing fmriprep output for this subject: `Run_3` (labeled `run-2` in our BIDS tree) received fieldmap 1's distortion correction instead of fieldmap 2's; `Run_7` (labeled `run-6`) received fieldmap 2's instead of fieldmap 3's. Fieldmap 3's `IntendedFor` also references a `run-8` file that doesn't exist in the final BIDS tree.
2. **Behavior/BOLD pairing**: our pipeline pairs BOLD runs with behavioral logs by run number. Since behavior still has all 8 runs logged, our naive pairing would associate the wrong scan with the wrong trial-onset data for every run from `Run_3` onward, once we get around to fitting this subject (caught before it caused any actual analysis error — GLMsingle hasn't been run for sub-26 yet).

We're fixing both on our end regardless of the outcome here (an explicit per-subject exception in our data-loading code). We just need to know whether `Run_2` can be recovered before we finalize the fix.

## Ask

Could someone check the scanner console / local storage / PACS for `sub-26`, session date **2026-08-10, ~11:07–11:20** (SNS Lab, Cima.X), for a functional series named `fMRI_G2_SMS3_Run_2` (expected `SeriesNumber` roughly 15–17, between `Run_1`=13 and `Run_3`=19)? We'd like to know either way:
- If it exists (even partially) — please export/deliver it so we can complete the session.
- If it doesn't — a confirmation that the sequence failed to start/save (rather than something being lost in export) would help us close this out.

Happy to provide any further file listings or logs if useful.
