# Experiment

## How to run
`run_task1.ps1` should run all the scripts in order.


## Phases

  * **Training**: This is self-paced, shouldn't take much more than a few minutes
  * **Feedback**: ..
  * **Task**: 

## Duration

### Feedback

We have *23* unique orientations that are all between 0 and 180 degrees.

The feedback trials consist of the following phases:

1. Green fixation (300 ms)
    - Green fixation cross signals trial start.

2. White fixation (700 ms)
    - White fixation cross to center attention.

3. Stimulus presentation (1500 ms)  
    - Oriented gabor stimulus drawn from the 23 orientations (0–180°).

4. Response window (up to 5000 ms)  
    - Participant reports perceived orientation using the slider.
    - Trial advances immediately after response (not fixed duration).

5. Feedback (1500 ms)  
    - Display actual orientation value with the participant's response highlighted.

6. Inter-trial interval (1000 ms)  
    - Blank screen before next trial begins.

**Single trial duration:** 5-10 seconds depending on response speed
- Minimum (instant response): 5.0 seconds (300 + 700 + 1500 + 0 + 1500 + 1000 ms)
- Maximum (full timeout): 10.0 seconds (300 + 700 + 1500 + 5000 + 1500 + 1000 ms)
- **Average: ~7 seconds per trial**

**23 trials per block:** ~2.5-4 minutes (depending on response speed)

**10 blocks (230 trials total):** ~27 minutes of trials + breaks = **~32 minutes total**

### Main Task

The main task consists of 23 orientations per run, repeated across 8 runs (184 trials total).

Trial phases:

1. Green fixation (300 ms)
    - Green fixation cross signals trial start.

2. White fixation (700 ms)
    - White fixation cross to center attention.

3. Stimulus presentation (1500 ms)  
    - Oriented gabor stimulus drawn from the 23 orientations (0–180°).

4. ISI (4000-5500 ms, randomly 4.0/4.5/5.0/5.5 seconds, equally distributed)
    - Inter-stimulus interval before response.

5. Response window (3000 ms, fixed)  
    - Participant places bid using the slider.
    - Trial duration is fixed (early responses extend the ITI).

6. Feedback (1000 ms)  
    - Display outcome and reward earned (if feedback enabled).

7. Inter-trial interval (1500 ms, variable to maintain fixed trial duration)  
    - Blank screen before next trial begins.

**Single trial duration:** 11.0-13.5 seconds (fixed per-trial despite variable ISI)
- With ISI=4.0s: 12.0 seconds (300 + 700 + 1500 + 4000 + 3000 + 1000 + 1500 ms)
- With ISI=4.5s: 12.5 seconds 
- With ISI=5.0s: 13.0 seconds
- With ISI=5.5s: 13.5 seconds
- **Mean: 12.75 seconds per trial**

**Per run:** 23 trials × 12.75s = 293 seconds (~4.9 minutes)

**8 runs total:** ~39 minutes of trials + instructions/breaks = **~45 minutes total**

---

**Complete experiment duration:**
- Instructions/setup: ~10 minutes
- Feedback training (10 blocks): ~32 minutes  
- Main task (8 runs): ~45 minutes
- **Total: ~87 minutes (1.5 hours)**

---

**Payment calculation (for optimal play):**
- Starting fee: 30 CHF
- Budget per trial: 42 CHF (kept if no purchase)
- 184 task trials with reward_scaling of 184.0 (payment = total_reward / 184)
- If bidding correctly (bid = true value):
  - Win probability depends on gabor value (higher values win less often)
  - When winning, pay random price (uniform 2-42), get true value back
  - Expected profit per trial ≈ 5 CHF (varies by gabor value)
- **Expected total payment: ~77 CHF** (30 CHF base + ~47 CHF from task)
