# Abstract Values Experiment

A behavioral fMRI experiment measuring how participants learn and apply value associations between oriented visual stimuli (gabors) and monetary amounts, using a Becker-DeGroot-Marschak (BDM) auction mechanism.

---

## Overview

Participants view flickering annular gratings (gabors) at 23 different orientations (7.5°–172.5°). Each orientation maps to a monetary value between 2 and 42 CHF via one of two nonlinear mappings (**cdf** or **inverse_cdf**). Across two sessions, each participant is exposed to both mappings in counterbalanced order. The task tests whether participants learn the orientation-to-value mapping and use it to bid honestly in an auction.

---

## Counterbalancing

The mapping assignment is determined by **subject ID parity** and **session number**:

| Subject ID | Session 1   | Session 2   |
|------------|-------------|-------------|
| Odd (1, 3, 5, …)  | `inverse_cdf` | `cdf`         |
| Even (2, 4, 6, …) | `cdf`         | `inverse_cdf` |

This is a standard AB/BA counterbalancing design: every participant sees both mappings, with order flipped between odd and even subjects. The assignment is implemented in `run_task.ps1` (lines 24–28).

### Mappings

Defined in `settings/sns_multisubject.yml` under `mappings:`. All three span 2–42 CHF across 25 orientations (0°–180° in 7.5° steps):

- **`cdf`** — values bunched at the **high end**: the mapping rises steeply at low orientations then flattens (ceiling effect).
- **`inverse_cdf`** — values bunched at the **low end**: the mapping is flat at low orientations then rises steeply (floor effect).
- **`linear`** — uniform spacing, used only for pilot/testing purposes.

---

## Experiment Structure

Each session consists of three phases run back-to-back via `run_task.ps1`:

### Phase 1 — Examples (`examples.py`)
Self-paced familiarization. Participants navigate through all 25 orientations using the left/right arrow keys. Each orientation is displayed alongside its CHF value. Participants must view all orientations before proceeding.

### Phase 2 — Training (`training.py`)
Estimation task with feedback. Participants view a gabor and use a mouse slider to estimate its CHF value. After responding, the correct value is shown.

- **Trials:** 23 orientations × 10 repeats = 230 trials, split into 10 blocks
- **Trial timeline:** green fixation (0.3 s) → white fixation (0.7 s) → gabor (1.5 s) → slider response (5 s) → feedback (1.5 s) → ITI (1.0 s)
- **Duration:** ~32 minutes

### Phase 3 — Main Task (`task.py`)
BDM auction task without feedback. Participants bid on gabors. Each bid is compared to a randomly drawn price; if the bid exceeds the price, the participant buys at the random price and sells at the true value.

- **Runs:** 8 runs (behavioral and fMRI)
- **Trials per run:** 20 (behavioral) / 23 (fMRI — all orientations exactly once)
- **Trial timeline:** green fixation (0.3 s) → white fixation (0.7 s) → gabor (1.5 s) → ISI (4.0–5.5 s, jittered) → slider response (3.0 s behavioral / 3.5 s fMRI) → feedback (1.0 s) → ITI (1.5 s)
- **Fixed trial duration:** the ITI is automatically shortened when the participant responds early, so each trial always lasts exactly **8.0 s + ISI** (behavioral) / **8.5 s + ISI** (fMRI)
- **ISI jitter (behavioral):** the 4 ISI values [4.0, 4.5, 5.0, 5.5 s] each occur **exactly 5 times** per run (20 trials = 4 × 5), guaranteeing a fixed run duration of 255 s
- **ISI jitter (fMRI):** ISI is drawn randomly from [4.0, 4.5, 5.0, 5.5 s] per trial; run duration is variable (~347 s expected)
- **Wait trials (fMRI only):** two 10 s fixation-only rest periods inserted after trial 7 and trial 15, controlled by `main_task.wait_duration`
- **Baseline fixation (fMRI only):** 20 s fixation at the start and end of each run, controlled by `main_task.baseline_duration`

#### Behavioral run duration (exact)

| Component | Duration |
|---|---|
| 20 trials × 8.0 s base | 160.0 s |
| 5 × each ISI (4.0 + 4.5 + 5.0 + 5.5) s | 95.0 s |
| **Total** | **255 s (4:15)** |

Each behavioral run starts with a self-paced instruction screen (click to continue).

#### fMRI run duration (expected)

| Component | Duration |
|---|---|
| Dummy-trigger wait (20 × TR) | ~19.9 s |
| 23 trials × 8.5 s base | 195.5 s |
| 23 ISIs (mean 4.75 s) | ~109.3 s |
| 2 × rest fixation | 20.0 s |
| Post-task fixation | 20.0 s |
| **Total (expected)** | **~365 s** |

Each fMRI run starts with a `DummyWaiterTrial` that counts sync triggers before the protocol begins (no instruction screen).

> **SCANNER PROTOCOL — SET VOLUMES TO 367**
>
> ISIs are drawn randomly from [4.0, 4.5, 5.0, 5.5 s], so run duration varies. The **expected** duration is **~365 s ÷ 0.996 s TR ≈ 367 volumes**. Set your scan to **367 volumes**. The post-task fixation may be cut a few seconds short or long depending on that run's ISIs — this is intentional and fine.

#### Total Phase 3 duration

| Setup | Settings file | Runs | Per run | Total |
|---|---|---|---|---|
| Behavioral | `sns_multisubject` | 8 | 255 s (exact) | **34:00 min** |
| fMRI | `sns_fmri` | 8 | ~365 s (variable) | **~49 min** |

fMRI run structure:
- **Dummy-trigger wait** — 20 triggers × TR 0.996 s ≈ **20 s** (fixation screen; serves as pre-task baseline in GLM)
- **Task trials 1–7** — ~112 s (7 trials, ISI-jittered)
- **Rest fixation** — 10 s
- **Task trials 8–15** — ~124 s (8 trials, ISI-jittered)
- **Rest fixation** — 10 s
- **Task trials 16–23** — ~124 s (8 trials, ISI-jittered)
- **Post-task fixation** — 20 s

> The four fixation periods (dummy wait ~20 s + 10 s + 10 s + post-task 20 s = **~60 s total**) serve as the fMRI baseline in the GLM.

### Earnings Display (`earnings.py`)
After all runs, total variable earnings and final payment are displayed on screen.

---

## Payment

- **Show-up fee:** 30 CHF
- **Variable earnings:** sum of auction outcomes across all trials, divided by the reward scaling factor (184.0 for 8-run behavioral; 207.0 for 9-run fMRI)
- **Auction outcome per trial:** if `bid > random_price`, earn `true_value − random_price`; otherwise earn 0
- **Optimal strategy:** honest bidding (bid = true value) maximizes expected earnings
- **Expected total:** ~77 CHF with optimal play

---

## Installation on a stimulus PC (or in a new Windows account)

The experiment is a **[uv](https://docs.astral.sh/uv/) project** rooted at this
`experiment/` folder. `pyproject.toml` is the single source of truth: Python
`3.10.*`, `psychopy>=2026.1`, `exptools2` (installed straight from GitHub, not
PyPI), `seaborn`, and `sr-research-pylink` for the Eyelink.

`uv sync` builds a virtual environment at `experiment\.venv`. That folder is
**per-account and never committed** — every Windows user that runs the
experiment creates their own, inside their own copy of the repo. Nothing here
needs administrator rights.

### Step 1 — Install `uv` (once per Windows account)

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

It installs into `%USERPROFILE%\.local\bin` and puts that on the user's `PATH`.
**Close and reopen PowerShell**, then check:

```powershell
uv --version
```

If that still says `uv is not recognized`, the PATH change has not reached the
shell — reopen it again, or call it by full path
(`& "$env:USERPROFILE\.local\bin\uv.exe" --version`).

### Step 2 — Get the code into *this* account's profile

Do not run out of another user's `C:\Users\...` folder — it will not be
readable, and the logs have to land somewhere this account can write.

```powershell
cd $env:USERPROFILE
mkdir experiments -Force
cd experiments
git clone https://github.com/Gilles86/abstract_values.git
cd abstract_values\experiment
```

(No git on the machine? `winget install --id Git.Git -e` — or copy the repo
folder over from a USB stick / the department share; only `experiment\` is actually needed on
the stim PC.)

### Step 3 — Build the environment

```powershell
uv sync
```

This downloads a private **Python 3.10** (no system Python required), resolves
every dependency and creates `experiment\.venv`. Expect a few minutes on first
run — PsychoPy and wxPython are large wheels. Verify:

```powershell
.\.venv\Scripts\python.exe -c "import psychopy, exptools2; print(psychopy.__version__)"
```

> **Note on reproducibility:** there is currently no `uv.lock` committed, so a
> fresh `uv sync` resolves whatever versions are current *today* — not
> necessarily what earlier participants ran on. If you want the stim PC frozen
> to a known-good set, run `uv lock` on the machine that works and commit the
> resulting `uv.lock`; from then on `uv sync` reproduces it exactly.

### Step 4 — Nothing to edit

There are no hardcoded paths left in the launchers. All of them start with

```powershell
. "$PSScriptRoot\_common.ps1"
```

and `_common.ps1` derives everything from where it sits:

```powershell
$expDir = $PSScriptRoot                                   # this folder
$python = Join-Path $expDir ".venv\Scripts\python.exe"    # the uv env next to it
```

So the same checkout runs under any Windows account, from any folder, with no
per-machine edits. If the `.venv` is missing, the script stops immediately with
a message telling you to run `uv sync` — rather than halfway through a session.

`_common.ps1` also defines the log-backup destination and the
`Copy-LogsToBackup` helper the launchers call at the end of a session (Step 5).

### Step 5 — Allow PowerShell to run the scripts

See **[First run on a new PC — allow PowerShell scripts](#first-run-on-a-new-pc--allow-powershell-scripts)**
below. Short version, once per account, no admin needed:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
Get-ChildItem *.ps1 | Unblock-File
```

### Step 6 — Check the backup drive is mapped *in this account*

Drive letters are per-user on Windows; they do not carry over from another
login. At the end of a session the launchers copy the logs to the project
folder on the department share:

```
Z:\projects\2026\dehollander_bedi_ruff_abstract_values\data\sourcedata\behavior
```

Check it from this account before the participant arrives:

```powershell
Test-Path "Z:\projects\2026\dehollander_bedi_ruff_abstract_values\data\sourcedata\behavior"
```

If the share is mounted under a different letter on this machine (it was `T:\`
on the previous stim PC), do **not** edit the scripts — set the environment
variable instead, which `_common.ps1` picks up:

```powershell
# this session only
$env:ABSTRACT_VALUES_BACKUP = "T:\projects\2026\dehollander_bedi_ruff_abstract_values\data\sourcedata\behavior"

# or permanently, for this Windows account
[Environment]::SetEnvironmentVariable(
    "ABSTRACT_VALUES_BACKUP",
    "T:\projects\2026\dehollander_bedi_ruff_abstract_values\data\sourcedata\behavior",
    "User")
```

An unreachable backup never aborts a session: the logs are always written to
`experiment\logs\` first, and the copy step just prints a loud warning telling
you to move them by hand.

### Step 7 — Smoke test before the participant arrives

A three-trial dummy run, written as `sub-99`:

```powershell
.\.venv\Scripts\python.exe task.py 99 1 99 cdf --settings sns_multisubject --n_trials 3
```

Equivalently, without naming the interpreter, `uv run python task.py 99 1 99 cdf
--settings sns_multisubject --n_trials 3` — `uv run` resolves the project's
`.venv` itself (and re-syncs it if it drifted from `pyproject.toml`). Then
delete `logs\sub-99\`.

For the fMRI settings, add `--settings sns_fmri`; note it waits for scanner
triggers, so test that one in the console with the scanner or expect it to sit
at the dummy-trigger screen.

### Updating an existing installation

```powershell
cd $env:USERPROFILE\experiments\abstract_values
git pull
cd experiment
uv sync
```

`uv sync` is cheap when nothing changed, and is the only step needed after a
dependency edit in `pyproject.toml`.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No Python environment found at ...\.venv\Scripts\python.exe` | The environment was never built here — run `uv sync` in `experiment\` (Step 3). |
| `Copying logs ... WARNING: backup location not reachable` | The backup drive is not mapped in this account, or it is not `Z:\` here — Step 6. Logs are safe in `experiment\logs\`. |
| `uv is not recognized` | Reopen PowerShell after installing uv, or call `%USERPROFILE%\.local\bin\uv.exe`. |
| `...ps1 cannot be loaded because running scripts is disabled` | Step 5 / the execution-policy section below. |
| `ModuleNotFoundError: psychopy` | A different interpreter is being used — always go through `.venv\Scripts\python.exe` or `uv run`. |
| `uv sync` fails on `exptools2` | No network access to GitHub from the stim PC, or git missing (uv needs git for a git dependency). |
| Eyelink import fails | `sr-research-pylink` is installed by `uv sync`, but the run also needs SR Research's own runtime installed on the machine; `--eyetracker` is optional, drop it to test. |
| Window opens on the wrong screen / wrong size | Not an install problem — see `settings/` (`sns_fmri.yml`, `sns_multisubject.yml`). |

---

## Running the Experiment

### First run on a new PC — allow PowerShell scripts

Windows refuses to run `.ps1` files by default (its **execution policy**), so a
fresh stim PC greets you with:

```
.\run_fmri.ps1 : File ...\run_fmri.ps1 cannot be loaded because running scripts
is disabled on this system.
```

**One-off**, changes nothing on the machine — good when you just need to scan now:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\run_fmri.ps1
```

**Permanent for your user** — do this once on the stim PC and `.\run_fmri.ps1`
works normally from then on. No admin rights needed:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

`RemoteSigned` permits local scripts while still blocking unsigned ones that
arrived from elsewhere — which is why a copy pulled off `T:\` or unzipped from a
download can *still* be refused afterwards. That is Windows' "mark of the web";
strip it with:

```powershell
Get-ChildItem *.ps1 | Unblock-File
```

**Still refusing?** Group Policy is overriding you — the SNS stim PCs are
UZH-domain machines. Check which scope is winning:

```powershell
Get-ExecutionPolicy -List
```

If `MachinePolicy` or `UserPolicy` reads anything other than `Undefined`, neither
command above will stick and SNS IT has to adjust it.

### Behavioral session (`run_task.ps1`)

```powershell
cd path\to\experiment
.\run_task.ps1
```

Prompts for `subject_id` and `session_id`, then runs all phases in sequence: `examples.py` → `training.py` → `task.py` × 8 runs → `earnings.py`. Logs are copied to the department share afterwards (destination in `_common.ps1`, see [Step 6](#step-6--check-the-backup-drive-is-mapped-in-this-account)).

### fMRI session (`run_fmri.ps1`)

```powershell
.\run_fmri.ps1
```

Prompts for `subject_id`, `session_id`, and whether to use the eyetracker, then:
1. **Practice run** (`training.py`, 30 trials, ~5 min) — run during the anatomical scan, no trigger needed
2. **8 functional runs** (`task.py --settings sns_fmri`) — each waits for sync triggers before starting
3. **Earnings display** (`earnings.py`)

### Eyetracker

The experiment supports an **Eyelink** eyetracker (right eye, 500 Hz, HV5 calibration). It is optional and off by default — `run_fmri.ps1` will ask `Use eyetracker? (y/n)` at startup.

When enabled (`--eyetracker` flag), each functional run:
1. Runs a full Eyelink calibration before the dummy-trigger wait
2. Starts recording immediately after `start_experiment()`
3. Saves the `.edf` file automatically on `close()`

The eyetracker is **not** used during the practice run (training during anatomical scan). To run task.py manually with the eyetracker:
```powershell
& $python task.py 01 1 1 cdf --settings sns_fmri --eyetracker
```

Counterbalancing is applied automatically (same logic as `run_task.ps1`).

---

## Scripts

| Script              | Phase          | Description                                      |
|---------------------|----------------|--------------------------------------------------|
| `examples.py`       | Phase 1        | Self-paced orientation–value learning            |
| `training.py`       | Phase 2        | Estimation with feedback                         |
| `task.py`           | Phase 3        | BDM auction task (called once per run)           |
| `earnings.py`       | End of session | Aggregates reward files and displays payment     |
| `utils.py`          | —              | `get_value()` helper, instruction trial base class |
| `stimuli.py`        | —              | `AnnulusGrating` and `FixationCross` classes     |
| `response_slider.py`| —              | Interactive mouse slider for responses           |
| `_common.ps1`       | —              | Shared path resolution + `Copy-LogsToBackup` for all `.ps1` launchers |

All scripts accept `--settings <name>` to select a settings file from `settings/`. The default for data collection is `sns_multisubject`.

---

## Settings

Settings files live in `settings/`. Key parameters in `sns_multisubject.yml`:

| Parameter                    | Value                              |
|------------------------------|------------------------------------|
| Screen resolution            | 1920 × 1080, fullscreen            |
| Grating spatial frequency    | 1.0 cycles/degree                  |
| Grating temporal frequency   | 2.0 Hz                             |
| Grating size / hole size     | 7.5° / 1.5°                        |
| Grating contrast             | 0.1                                |
| Slider range                 | 2–42 CHF                           |
| Training blocks / repeats    | 10 blocks × 10 repeats             |
| Main task runs               | 8 (behavioral and fMRI)            |
| ISI range (main task)        | 4.0, 4.5, 5.0, 5.5 s (jittered)   |
| Wait trial duration (fMRI)   | 10 s (at 1/3 and 2/3 of each run) |
| Reward scaling factor        | 184.0 (behavioral) / 207.0 (fMRI) |

---

## Output Files

Logs are written to `logs/sub-{subject}/session-{session}/`:

```
sub-{sub}_ses-{ses}_task-examples.{mapping}_events.tsv
sub-{sub}_ses-{ses}_task-training.{mapping}_events.tsv
sub-{sub}_ses-{ses}_run-{run}_task-estimate.{mapping}_events.tsv   (× 8 runs)
reward_{subject}_{session}_{run}.txt                               (× 8 runs)
sub-{sub}_ses-{ses}_earnings_events.tsv
```
where `{sub}` is zero-padded (e.g. `01`), `{ses}` is not (e.g. `1`), `{run}` is zero-padded (e.g. `01`).

Each `_events.tsv` contains one row per trial phase with columns: `trial_nr`, `onset`, `event_type`, `phase`, `response`, `nr_frames`, `orientation`, `value`, `response_time`, `onset_abs`, `duration`.

Each `reward_*.txt` contains a single CHF value (the variable earnings for that run).

Alongside each `.tsv`, a `_expsettings.yml` snapshot of the settings used is saved for reproducibility.
