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
- **Trial timeline:** green fixation (0.3 s) → white fixation (0.7 s) → gabor (1.5 s) → ISI (4.0–5.5 s, jittered) → slider response (3.5 s) → feedback (1.0 s) → ITI (1.5 s)
- **Fixed trial duration:** the ITI is automatically shortened when the participant responds early, so each trial always lasts exactly **8.5 s + ISI**
- **ISI jitter (behavioral):** the 4 ISI values [4.0, 4.5, 5.0, 5.5 s] each occur **exactly 5 times** per run (20 trials = 4 × 5), guaranteeing a fixed run duration of 255 s
- **ISI jitter (fMRI):** ISI is drawn randomly from [4.0, 4.5, 5.0, 5.5 s] per trial; run duration is variable (~347 s expected)
- **Wait trials (fMRI only):** two 10 s fixation-only rest periods inserted after trial 7 and trial 15, controlled by `main_task.wait_duration`
- **Baseline fixation (fMRI only):** 20 s fixation at the start and end of each run, controlled by `main_task.baseline_duration`

#### Behavioral run duration (exact)

| Component | Duration |
|---|---|
| 20 trials × 8.5 s base | 170.0 s |
| 5 × each ISI (4.0 + 4.5 + 5.0 + 5.5) s | 95.0 s |
| **Total** | **265 s (4:25)** |

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
| Behavioral | `sns_multisubject` | 8 | 265 s (exact) | **35:20 min** |
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

## Running the Experiment

### Behavioral session (`run_task.ps1`)

```powershell
cd path\to\experiment
.\run_task.ps1
```

Prompts for `subject_id` and `session_id`, then runs all phases in sequence: `examples.py` → `training.py` → `task.py` × 8 runs → `earnings.py`. Logs are copied to `N:\client_write\gilles\experiment\logs\`.

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
