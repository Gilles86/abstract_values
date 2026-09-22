# Running an abstract-values session on your own

Step-by-step protocol for one participant session: forms, the learning phases in the testing
room, and the 8 fMRI runs. Written for running the session without Gilles present.

> **Fill in before the participant arrives** — Gilles gives you the first two:
>
> | | |
> |---|---|
> | Participant number | `____` (plain integer, e.g. `31`) |
> | Session | `1` / `2` |
> | Instruction booklet (A or B) | follows from the two above — see [§2](#2-forms-to-collect-from-the-console-room) |
> | Scan start time | `____` |
> | Gilles' phone | `____` |
> | Irini's phone | `____` |
>
> **The participant number is the one thing that must be right.** It determines which
> orientation→CHF mapping the software uses, and it has to match the printed booklet you hand
> out. Almost anything else can be repaired afterwards; a booklet/software mismatch cannot.

---

## 0. Shape of the day

| When | Where | What |
|---|---|---|
| −60 min | Console room | Collect the forms (§2) |
| −50 min | Testing room | MR safety + consent, instruction booklet, then learning phases (§3–§4) |
| −10 min | Scanner | Hand over to Irini for positioning (§5) |
| 0 min | Stim PC, console room | `run_fmri.ps1`: practice during the anatomical, then 8 runs (§6) |
| +60 min | Console room | Earnings screen, payment, check the logs (§8) |

Participant time ≈ 2.5 h.

---

## 1. What the participant does (so you can explain it)

Two sessions on two different days. Each session has three phases:

1. **Phase 1 — study (testing room).** All 25 gabors with their CHF values, self-paced.
   Right arrow = next, left arrow = back, space = continue (only possible once they have seen
   all of them).
2. **Phase 2 — practice with feedback (testing room).** A gabor appears; they set a mouse slider
   to its CHF value; the correct value is then shown. 230 trials in 10 blocks, ~32 min.
3. **Phase 3 — auction (scanner).** A gabor appears; they place a **bid** with the slider. No
   feedback. 8 runs of ~6 min.

Each session uses a **different** angle→value mapping. Say this explicitly: what they learn today
is not what they learned (or will learn) on the other day.

**Payment:** 30 CHF show-up fee per session plus variable earnings from the auction. Bidding
exactly what they think the gabor is worth is the optimal strategy (BDM auction: the price is
drawn at random up to their bid, so overbidding costs them and underbidding loses deals).
Expected total ≈ 77 CHF across both sessions.

---

## 2. Forms to collect from the console room

In the back of the scanner console room. Per participant:

- [ ] **MR safety screening form**
- [ ] **Informed consent — information text** (what they read)
- [ ] **Informed consent — signature form** (what they sign)
- [ ] **Instruction booklet — mapping A _or_ B** (below)

### Which booklet?

The first page reads **"Instructions (mapping A)"** or **"Instructions (mapping B)"**. Pick by
the parity of the participant number:

| Participant number | Session 1 | Session 2 |
|---|---|---|
| **Odd** (31, 33, 35 …) | **B** | **A** |
| **Even** (32, 34, 36 …) | **A** | **B** |

Session 2 always gets *the other* booklet than session 1.

Behind the scenes A = `cdf` (values bunched towards the high end), B = `inverse_cdf` (bunched
towards the low end). The software works this out from participant number + session on its own —
**you never type the mapping**. Each launcher prints the mapping it picked; check that line
against the booklet before the participant starts reading.

> Mismatch discovered after they started reading? Stop, phone Gilles, restart with the correct
> booklet. Never "fix" it by changing the software.

---

## 3. Paperwork with the participant (testing room)

1. Greet them. One sentence: "Two sessions of about 2.5 hours, gabor patterns and money values;
   you learn the mapping, then bid on gabors in the scanner. Today is **session X of 2**."
2. **MR safety form** — go through it *with* them, don't hand it over to fill in alone. Metal,
   implants, surgery, tattoos, claustrophobia, glasses/contacts. Anything borderline: **Irini
   decides, not you.**
3. **Consent information text** — they read it.
4. **Consent signature form** — they sign, you sign as experimenter.
5. **Instruction booklet** — ~10 min, self-paced. Tell them: "Read it carefully, your payment
   depends on understanding the rules; ask me anything, also while reading."
6. Ask **"is anything unclear?"** before Phase 1 starts, and again after Phase 2.
7. Before going anywhere near the scanner: phone, keys, cards, coins, jewellery, hairpins out.

---

## 4. Learning phases in the testing room (Phases 1–2)

⚠️ **Confirm with Gilles**: which PC/login in the single-subject room, and where the repo lives
on it. The commands below work from whatever folder the scripts are in — that's the point of the
recent refactor — but you still need to know *which* folder that is.

1. Log in on the stimulus PC in the single-subject testing room.
2. Open **PowerShell** (Start → type `powershell`).
3. Go to the experiment folder, e.g.:

   ```powershell
   cd $env:USERPROFILE\experiments\abstract_values\experiment
   ```

   If that path doesn't exist, find it: `Get-ChildItem -Path C:\ -Filter run_single_subject.ps1 -Recurse -ErrorAction SilentlyContinue`

4. Start it. Windows blocks `.ps1` files by default, so launch it this way:

   ```powershell
   powershell.exe -ExecutionPolicy Bypass -File .\run_single_subject.ps1
   ```

5. It asks two things:

   ```
   Enter subject_id (integer)   →  e.g. 31     (NOT "01", NOT "sub-31")
   Enter session_id (1 or 2)    →  1 or 2
   ```

   It then prints e.g. `Running single-subject experiment for subject 31, session 1: inverse_cdf`.
   **Check that against the booklet** (`cdf` = A, `inverse_cdf` = B).

6. Phase 1 (examples) starts. Sit with them through the first few gabors, then let them work.
   Phase 2 (training, ~32 min) follows automatically, with breaks between the 10 blocks. The
   slider is mouse-driven — make sure the mouse has room and a decent surface.
7. When Phase 2 ends, ask again whether anything is unclear, then walk them over to the scanner.

---

## 5. In the scanner

**Irini knows the scanning side — defer to her.** Positioning, ear protection, coil, squeeze
ball, the protocol on the console, starting each acquisition: hers. Your job is the stimulus PC
and the participant's understanding of the task.

Two things to tell the participant before they go in:

- "Keep your eyes on the fixation cross the whole time, also when nothing is happening."
- "Try not to move, especially your head — also not between runs."

Task-relevant scanner settings (Irini has the protocol, but these are worth knowing):

| | |
|---|---|
| Functional runs | **8** |
| Volumes per run | **367** |
| TR | 0.996 s |
| Trigger character sent to the stim PC | `5` |
| Task waits for | **20 triggers** (~20 s) at the start of every run |

The task shows *"Waiting for scanner"* with a live trigger counter and starts by itself once 20
triggers have arrived. **You never press anything to start a run** — the scanner starts it.

---

## 6. Running the fMRI task (stim PC in the console room)

1. On the stim PC in the console room, open PowerShell and go to the experiment folder:

   ```powershell
   cd $env:USERPROFILE\experiments\abstract_values\experiment
   ```

2. Run:

   ```powershell
   powershell.exe -ExecutionPolicy Bypass -File .\run_fmri.ps1
   ```

3. Three questions:

   ```
   Enter subject_id (integer)   →  the same number as in the testing room
   Enter session_id (1 or 2)    →  the same session
   Use eyetracker? (y/n)        →  y
   ```

   It prints the mapping again — check it against the booklet once more.

4. From here it runs by itself:

   | # | What | Notes |
   |---|---|---|
   | 1 | **Practice run**, 36 trials with feedback, ~5 min | Do this **during the anatomical scan**. No trigger needed, starts immediately. |
   | 2 | **Run 1 of 8** | Eyetracker calibration first (only before run 1), then "Waiting for scanner". |
   | 3 | **Runs 2–8** | Each waits for its own 20 triggers. After each run a screen shows that run's and the cumulative earnings — press **space** to continue to the next run. |
   | 4 | **Earnings screen** | The participant's total payment. Any key closes it. |
   | 5 | **Log copy to the department share** | Automatic; watch for `Logs copied successfully!` (§8). |

   Between runs, check in over the intercom, then press space and let Irini start the next
   acquisition.

> **Don't close the PowerShell window between runs.** The whole session is one script; closing it
> means continuing in resume mode (§7).

---

## 7. When something goes wrong

### Cancelling a run

Press **`Q`** in the stimulus window. This aborts the **current** run — and then the script
**immediately continues with the next one**. So:

- the aborted run's data are incomplete and its earnings are not counted;
- if you aborted run 3, the script goes on to run 4 and run 3 is simply missing.

Cleanest repair: let runs 4–8 finish, then redo the missing run by hand at the end, using the
mapping the script printed at the start:

```powershell
& .\.venv\Scripts\python.exe task.py 31 1 3 inverse_cdf --settings sns_fmri --eyetracker
#                                    ^sub ^ses ^run ^mapping
```

This writes the same filenames as the aborted attempt, i.e. the partial files are replaced —
which is what you want. Tell Irini you need one extra acquisition of that run.

### Restarting halfway (crash, reboot, closed window)

Use the resume script — it doesn't redo what's already done:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\run_fmri_resume.ps1
```

Same subject/session questions, then:

```
  t   = practice run only
  1-8 = start from run N (e.g. "3" runs 3,4,5,6,7,8)
  e   = earnings screen only
  all = everything (practice + 8 runs + earnings)
```

Crashed during run 5 → enter `5`; it runs 5–8 and then shows earnings. It always continues
through run 8, so it can't redo a single run in the middle — use the manual `task.py` command
above for that.

### "Running scripts is disabled on this system"

Windows' execution policy. The `powershell.exe -ExecutionPolicy Bypass -File ...` form above
sidesteps it. If a script copied from the network drive is still refused:

```powershell
Get-ChildItem *.ps1 | Unblock-File
```

### Other problems

| Symptom | What to do |
|---|---|
| `No Python environment found at ...\.venv\...` | The environment isn't built in this Windows account. `uv sync` in the experiment folder (see README, "Installation on a stimulus PC"). If `uv` is missing too, phone Gilles — don't improvise with another Python. |
| Stuck on "Waiting for scanner", counter not moving | Triggers aren't reaching the stim PC. Ask Irini: is the sequence actually running, is the trigger cable/USB connected? Abort with `Q` and redo the run. |
| Slider doesn't move | Mouse not being registered. Check this **before** run 1. |
| `WARNING: backup location not reachable` at the end | Normal-ish: the share isn't mapped in this account. The data are safe in `experiment\logs`; copy them over by hand (§8). |
| Earnings screen: `No reward files found` | Don't improvise a payment figure — phone Gilles. Means neither the local logs nor the share had anything for this participant. |
| Console warns `session 1 has 0/8 runs` on a session 2 | The share isn't mapped in this account (so the fallback found nothing) — the displayed total is missing session 1. Map the drive, or set `$env:ABSTRACT_VALUES_BACKUP` and rerun `run_fmri_resume.ps1` with option `e`. |
| Gabor looks wrong / wrong screen / wrong size | Stop and phone Gilles. Do not edit anything in `settings\`. |
| A Windows dialog pops up and everything freezes | Dismiss it on the stim PC — the task can't receive input while a dialog is open. |

---

## 8. End of session

1. Irini gets the participant out; check they're OK.
2. **Payment:** the earnings screen shows the total (session 1: 30 CHF + this session's variable
   earnings; session 2: both sessions and the grand total).

   Where those numbers come from: the local `logs` folder first, and the department share
   (`Z:\...\sourcedata\behavior`) for any run that isn't there. So a second session finds its
   first session even when that was recorded on a different stim PC. The console says so
   explicitly, e.g. `Session 1: 8 run(s) read from Z:\...\sub-31\ses-1`, and it warns
   `WARNING: session 1 has 6/8 runs; missing run(s): [7, 8]` if anything is incomplete —
   **read those lines before paying out.**

   To double-check a payment away from the stim PC:
   `python calculate_earnings.py --bids_folder <share or dataset root>`.

   ⚠️ **Confirm with Gilles**: who pays out, in what form, and at which session.
3. If this was session 1: confirm the date of session 2 and write it down.
4. **Check the log copy.** The script prints either `Logs copied successfully!` or a red warning.
   If it warned, copy by hand — the destination drive letter differs per machine (`Z:\` on the
   current stim PC, `T:\` on the old one):

   ```powershell
   Copy-Item -Path .\logs\sub-* -Destination `
     "Z:\projects\2026\dehollander_bedi_ruff_abstract_values\data\sourcedata\behavior\" -Recurse -Force
   ```

   You can also set the destination once per shell before starting a session:
   `$env:ABSTRACT_VALUES_BACKUP = "T:\projects\2026\...\sourcedata\behavior"`.
5. Message Gilles: participant number, session, mapping, which runs completed, and anything odd
   (aborted runs, lots of motion, participant confusion, missing triggers, eyetracker trouble).

---

## 9. Don'ts

- Don't type a zero-padded number (`01`) or `sub-31` — plain integer.
- Don't set the mapping by hand; it follows from the participant number.
- Don't edit anything in `settings\`.
- Don't delete anything under `logs\`. Session 2's earnings screen reads session 1's reward
  files from there, falling back to the share for whatever is missing — so both copies matter.
- Don't reuse a participant number that already exists.

---

## 10. Reference: what a complete session leaves behind

On the stim PC, in `logs\sub-{nr}\ses-{session}\`:

```
sub-31_ses-1_task-training.inverse_cdf_events.tsv          practice run
sub-31_ses-1_run-01_task-estimate.inverse_cdf_events.tsv   × 8 runs
sub-31_ses-1_run-01_task-estimate.inverse_cdf.edf          eyetracking, × 8 runs
reward_31_1_1.txt ... reward_31_1_8.txt                    earnings per run
sub-31_ses-1_earnings_events.tsv
```

Eight `_events.tsv`, eight `.edf` and eight `reward_*.txt` — that's the fastest completeness
check before you leave the building.
