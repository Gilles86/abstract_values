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

- **Runs:** 8 runs (behavioral) / 10 runs (fMRI), each with **20 trials**
- **Trial timeline:** green fixation (0.3 s) → white fixation (0.7 s) → gabor (1.5 s) → ISI (4.0–5.5 s, jittered) → slider response (3 s) → feedback (1.0 s) → ITI (1.5 s)
- **Fixed trial duration:** the ITI is automatically shortened when the participant responds early, so each trial always lasts exactly **8.0 s + ISI**
- **ISI jitter:** the 4 ISI values [4.0, 4.5, 5.0, 5.5 s] each occur **exactly 5 times** per run (20 trials = 4 × 5), guaranteeing a fixed run duration

#### Exact run duration

| Component | Duration |
|---|---|
| 20 trials × 8.0 s base | 160.0 s |
| 5 × each ISI (4.0 + 4.5 + 5.0 + 5.5) s | 95.0 s |
| **Total per run** | **255 s (4:15)** |

Each run also starts with a self-paced instruction screen (spacebar to continue), not included above.

#### Total Phase 3 duration

| Setup | Settings file | Runs | Per run (acquired) | Total (excl. instruction screens) |
|---|---|---|---|---|
| Behavioral / pilot | `sns_multisubject` | 8 | 255 s | **34:00 min (exact)** |
| fMRI | `sns_fmri` | 10 | 20 + 255 + 20 = 295 s | **49:10 min (exact)** |

fMRI run structure (acquired data only, after dummy scans):
- **Baseline fixation** — 20 s white fixation cross
- **20 task trials** — 255 s
- **Baseline fixation** — 20 s white fixation cross

fMRI dummy scans (discarded, before experiment starts): 20 volumes × TR 0.996 s ≈ 19.92 s per run.

### Earnings Display (`earnings.py`)
After all 8 runs, total variable earnings and final payment are displayed on screen.

---

## Payment

- **Show-up fee:** 10 CHF
- **Variable earnings:** sum of auction outcomes across all trials, divided by the reward scaling factor (160.0 for 8-run behavioral; 200.0 for 10-run fMRI)
- **Auction outcome per trial:** if `bid > random_price`, earn `true_value − random_price`; otherwise earn 0
- **Optimal strategy:** honest bidding (bid = true value) maximizes expected earnings
- **Expected total:** ~57 CHF with optimal play

---

## Running the Experiment

On the stimulus PC, open PowerShell and run:

```powershell
cd path\to\experiment
.\run_task.ps1
```

You will be prompted for:
- `subject_id` — integer
- `session_id` — 1 or 2

The script then activates the virtual environment and runs all four phases in sequence (`examples.py` → `training.py` → `task.py` × 8 runs → `earnings.py`). At the end, logs are copied to `N:\client_write\gilles\experiment\logs\`.

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
| Main task runs               | 8                                  |
| ISI range (main task)        | 4.0, 4.5, 5.0, 5.5 s (jittered)   |
| Reward scaling factor        | 184.0                              |

---

## Output Files

Logs are written to `logs/sub-{subject}/session-{session}/`:

```
sub-{s}_ses-{ss}_task-examples.{mapping}_events.tsv
sub-{s}_ses-{ss}_task-training.{mapping}_events.tsv
sub-{s}_ses-{ss}_run-{rr}_task-estimate.{mapping}_events.tsv   (× 8 runs)
reward_{subject}_{session}_{run}.txt                            (× 8 runs)
sub-{s}_ses-{ss}_earnings_events.tsv
```

Each `_events.tsv` contains one row per trial phase with columns: `trial_nr`, `onset`, `event_type`, `phase`, `response`, `nr_frames`, `orientation`, `value`, `response_time`, `onset_abs`, `duration`.

Each `reward_*.txt` contains a single CHF value (the variable earnings for that run).

Alongside each `.tsv`, a `_expsettings.yml` snapshot of the settings used is saved for reproducibility.
