# Running an abstract-values session

One participant session: learning phases in the testing room, then 8 fMRI runs. Irini owns
everything scanner-side (positioning, protocol, starting acquisitions); you own the stimulus PC.

**Thursday 24 September: two participants, `33` and `34`, both session 1.**

| Participant | Type in the prompts | Booklet |
|---|---|---|
| 33 | `33`, session `1` | **B** (`inverse_cdf`) |
| 34 | `34`, session `1` | **A** (`cdf`) |

Plain integers — not `01`, not `sub-33`. Everything below is one participant's session; it runs
twice. Second sessions are next week, with Gilles. Gilles is reachable by phone (you have the
number) and Irini is in the console room.

| When | Where | What |
|---|---|---|
| −60 | Console room | Forms: MR safety, consent info + signature, instruction booklet |
| −50 | Testing room | Paperwork, booklet (~10 min self-paced), then `run_single_subject.ps1` (phases 1–2, ~40 min) |
| −10 | Scanner | Hand over to Irini |
| 0 | Console room | `run_fmri.ps1`: practice during the anatomical, then 8 runs |
| +60 | Console room | Earnings screen (no payout — that is session 2), check the logs |

Participant time ≈ 2.5 h. Each session uses a *different* angle→value mapping — worth saying out
loud, since what they learned on the other day does not transfer.

## Booklet

Thursday's two are in the table above. The general rule, for any other session: the software
derives the mapping from participant number + session, and **you never type it**. Each launcher
prints the mapping it picked — check that line against the booklet in front of you.

| Participant number | Session 1 | Session 2 |
|---|---|---|
| Odd (31, 33, …) | **B** = `inverse_cdf` | **A** = `cdf` |
| Even (32, 34, …) | **A** = `cdf` | **B** = `inverse_cdf` |

Mismatch found after they started reading: swap in the right booklet and start the reading over
— the table above is the whole rule, so you don't need to check with anyone. Never fix it the other
way round, in the software. Mention it to Gilles afterwards, since they've seen the wrong mapping
figure.

## Testing room — phases 1 and 2

```powershell
cd $env:USERPROFILE\experiments\abstract_values\experiment
powershell.exe -ExecutionPolicy Bypass -File .\run_single_subject.ps1     # asks subject, session
```

Both commands are in this account's PowerShell history from yesterday — `Get-History`, or arrow up
/ `Ctrl+R`, beats retyping the path.

Phase 1 (study, self-paced: ←/→ to browse, space once all 25 seen) runs into phase 2 (slider
estimates with feedback, 230 trials in 10 blocks, ~32 min, breaks between blocks). Mouse-driven
slider — give it room.

## Scanner

| | |
|---|---|
| Runs × volumes | 8 × 367, TR 0.996 s |
| Trigger char | `5`, task waits for 20 per run (~20 s) |

Each run starts itself once the triggers arrive; you never press anything to start one. Tell the
participant: eyes on the fixation cross throughout, don't move between runs either.

Pre-flight before they're in the room — 3 dummy trials, also confirms stimulus geometry
(press `5` twenty times to clear the trigger screen, then delete `logs\sub-99\`):

```powershell
.\.venv\Scripts\python.exe task.py 99 1 99 cdf --settings sns_fmri --n_trials 3
```

`sns_fmri` is the only correct settings file here; the others describe a desk monitor at 60 cm
instead of the projector at 100 cm, so everything comes out ~1.7× off.

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\run_fmri.ps1     # subject, session, eyetracker? y
```

Then it runs by itself: practice run (36 trials, no triggers — do it **during the anatomical**),
eyetracker calibration, run 1, … run 8, earnings screen, log copy to the share. After each run an
earnings screen waits for **space** before the next one. **Don't close the PowerShell window** —
the whole session is one script.

## When something goes wrong

**`Q` aborts the current run and immediately continues with the next one.** Abort run 3 and the
script moves to run 4; run 3 is simply missing. Let 4–8 finish, then redo it by hand with the
mapping the script printed (same filenames, so the partial files are overwritten):

```powershell
& .\.venv\Scripts\python.exe task.py 31 1 3 inverse_cdf --settings sns_fmri --eyetracker
#                                    sub ses run mapping
```

Crash, reboot or closed window: `run_fmri_resume.ps1` → `t` = practice, `1`–`8` = start from run N
and continue to 8, `e` = earnings only, `all` = everything. It can't redo a single run in the
middle — use the command above for that.

| Symptom | What to do |
|---|---|
| `running scripts is disabled on this system` | Use the `-ExecutionPolicy Bypass` form above; for files copied off a share, `Get-ChildItem *.ps1 \| Unblock-File`. |
| `No Python environment found at ...\.venv\...` | Not built in this Windows account: `uv sync` in the experiment folder (README, "Installation on a stimulus PC"). Never point the launcher at some other Python — that's how you end up on a different PsychoPy. |
| Trigger counter not moving | Triggers aren't reaching the stim PC — ask Irini (sequence actually running? cable?). `Q` and redo the run. |
| `WARNING: backup location not reachable` | Share not mapped in this account. Data are safe in `experiment\logs`; copy by hand (below). |
| `No reward files found`, or `session 1 has 0/8 runs` on a session 2 | The share fallback found nothing, so the total is wrong. Map the drive or set `$env:ABSTRACT_VALUES_BACKUP`, rerun `run_fmri_resume.ps1` option `e`. Don't improvise a payment figure. |
| Gabor or text the wrong size | Check the settings name in the console line: `run_fmri.ps1` passes `sns_fmri`, which is correct for this room. If it says that and still looks wrong, the projector resolution or Windows display scaling differs from 1920×1200 at 100 %. Don't edit `settings\`. |
| Windows dialog on top | Dismiss it on the stim PC; the task gets no input while one is open. |

## End of session

**No money changes hands on Thursday** — participants are paid after session 2, by Gilles. The
earnings screen still comes up and shows the participant their running total; let them read it,
that's all.

It reads the local `logs` folder first and the department share for anything missing, so next
week's session 2 finds session 1 even from another stim PC. It prints where each session came from
and warns `session 1 has 6/8 runs; missing run(s): [7, 8]` — worth a glance, since that warning is
what next week's payment depends on.

If the log copy warned instead of printing `Logs copied successfully!`:

```powershell
Copy-Item -Path .\logs\sub-* -Recurse -Force -Destination `
  "Z:\Department\projects\2026\dehollander_bedi_ruff_abstract_values\data\sourcedata\behavior\"
```

(`$env:ABSTRACT_VALUES_BACKUP` overrides the drive letter — it was `T:\` on the old stim PC.)
Don't delete anything under `logs\`: session 2's earnings read session 1's reward files there.

Completeness check before leaving, in `logs\sub-{nr}\ses-{session}\` — eight `_events.tsv`, eight
`.edf`, eight `reward_*.txt`, plus the practice `task-training` TSV and the earnings TSV.

Then message Gilles: participant, session, mapping, runs completed, anything odd (aborts, motion,
missing triggers, eyetracker trouble).
