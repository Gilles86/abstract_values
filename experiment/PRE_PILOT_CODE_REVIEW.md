# Pre-Pilot Code Review - Abstract Values Experiment

**Date:** 2 March 2026  
**Reviewer:** AI Assistant  
**Status:** Ready for pilot with minor fixes recommended

---

## 🔴 CRITICAL ISSUES (Fix before pilot)

### 1. **Redundant/Confusing Trial Creation Logic** ([task.py](task.py#L99-L104))
```python
# Line 102 - this line seems redundant/leftover
self.orientations = self.orientations.copy() * self.settings['main_task'].get('n_repeats')
```
**Issue:** Orientations are already being repeated later. This could cause double-repetition.  
**Fix:** Remove line 102 or clarify the logic.

### 2. **Typo in Variable Name** ([examples.py](examples.py#L36))
```python
self.visisted_all_orientations = False  # Should be "visited"
```
**Impact:** Code works but inconsistent naming could cause confusion.  
**Fix:** Rename to `visited_all_orientations`.

### 3. **Debug Print Statement** ([training.py](training.py#L118))
```python
print(phase_durations)  # Debug leftover
```
**Fix:** Remove before pilot.

---

## 🟡 IMPORTANT ISSUES (Should fix)

### 4. **No Error Handling for File Operations**
Multiple locations write files without try/except:
- [task.py](task.py#L75-L77): Reward file writing
- [earnings.py](earnings.py#L25-L27): Reading reward files

**Risk:** Disk full, permissions errors, or network issues could crash the experiment.

**Recommended fix:**
```python
try:
    with open(reward_file, 'w') as f:
        f.write(f'{total_reward:.2f}\n')
except IOError as e:
    print(f"ERROR: Could not save reward file: {e}")
    # Log error and continue or show error to experimenter
```

### 5. **Potential Missing Response Handling**
[task.py](task.py#L221): Assumes response exists:
```python
if self.parameters['response'] > bid:
```

**Issue:** If subject never clicks during response window, `response` key won't exist.  
**Current status:** Actually might be OK because response is only checked inside `if self.session.mouse.getPressed()[0]:` block, but should verify during pilot.

### 6. **ISI Array Extension Uses Repetition**
[task.py](task.py#L107-L109):
```python
while len(isis) < n_trials:
    isis = isis + isis
```

**Issue:** For very large n_trials, this could use lots of memory. Also, doubling creates patterns.  
**Better approach:** Use `np.tile()` or `np.random.choice()` to sample ISIs.

### 7. **Training Block Distribution**
[training.py](training.py#L73): Integer division could create uneven blocks.
```python
block_size = len(self.orientations) // n_blocks
```

**Example:** 95 trials / 10 blocks = 9 trials/block, but last block gets 14 trials.  
**Status:** Handled correctly in lines 81-82, but worth verifying during pilot.

---

## 🟢 GOOD TO FIX (Lower priority)

### 8. **No Validation of Settings at Startup**
If settings file is malformed or missing required keys, experiment crashes mid-run.

**Recommendation:** Add validation function:
```python
def validate_settings(self):
    required_keys = ['durations', 'slider', 'mappings', 'grating', 'main_task']
    for key in required_keys:
        if key not in self.settings:
            raise ValueError(f"Missing required setting: {key}")
```

### 9. **Mouse Position Exception Handling**
[task.py](task.py#L203-L206) and [training.py](training.py#L183-L186):
```python
except Exception as e:
    print(e)
```

**Issue:** Exception is printed but not logged to file. Will be invisible after experiment.  
**Fix:** Log exceptions properly.

### 10. **No Progress Feedback**
Subjects don't know how many trials remain in a run. Consider adding trial counter like "Trial 15/92" during ITI.

### 11. **Orientation/Value Array Length Validation**
No check that `orientations` and value arrays (`linear`, `cdf`, `inverse_cdf`) have matching lengths.

**Recommendation:** Add assertion in settings validation:
```python
assert len(settings['mappings']['orientations']) == len(settings['mappings']['cdf'])
```

---

## ✅ GOOD PRACTICES OBSERVED

1. ✓ Using `parameters.get('reward', 0.0)` with default values
2. ✓ Proper use of pathlib for file paths
3. ✓ Separate settings files for different experimental setups
4. ✓ Clear phase names for trial structure
5. ✓ Response precision/snapping implemented correctly
6. ✓ Random ISI shuffling to avoid predictability
7. ✓ Reward scaling parameter for easy adjustment
8. ✓ Separate logs per subject/session

---

## 📋 PRE-PILOT CHECKLIST

### Before first pilot session:
- [ ] Remove debug `print(phase_durations)` from training.py
- [ ] Fix typo: `visisted` → `visited` in examples.py
- [ ] Verify line 102 in task.py is needed (trial repetition logic)
- [ ] Test on actual experimental computer (not this Mac)
- [ ] Verify network drive N:\ is accessible and writable
- [ ] Run test_copy.ps1 to verify log backup works
- [ ] Check that all paths in run_task.ps1 are correct
- [ ] Test full experiment sequence with a test subject ID

### During pilot monitoring:
- [ ] Watch for any error messages in terminal
- [ ] Verify log files are being created properly
- [ ] Check that reward files are created for each run
- [ ] Ensure logs are copied to network drive successfully
- [ ] Monitor that subjects understand the slider interface
- [ ] Check timing - is everything comfortable pace?
- [ ] Verify earnings calculation at the end is correct

### Data validation after each pilot:
- [ ] Check .tsv event files have all expected columns
- [ ] Verify response values are within expected range [2, 42]
- [ ] Check that orientation values match expected set
- [ ] Confirm reward calculations make sense
- [ ] Validate that ISIs match expected values
- [ ] Ensure timestamps are monotonically increasing

---

## 🧪 TESTING RECOMMENDATIONS

### Quick smoke test (5 min):
```bash
# Run with very few trials to test workflow
python examples.py test 1 cdf --settings sns_multisubject
python training.py test 1 cdf --settings sns_multisubject --n_trials 3
python task.py test 1 1 cdf --settings sns_multisubject --n_trials 3
python earnings.py test 1 --settings sns_multisubject
```

### Full test run (30-40 min):
Run the entire run_task.ps1 with a test subject to verify:
- All phases run smoothly
- Timings feel right
- Instructions are clear
- Data files are created correctly
- Network backup works
- Earnings display is accurate

---

## 📊 DATA STRUCTURE VERIFICATION

### Expected files per subject/session:
```
logs/sub-{ID}/session-{XX}/
  ├── sub-{ID}_ses-{XX}_task-training.{mapping}_events.tsv
  ├── sub-{ID}_ses-{XX}_task-training.{mapping}_expsettings.yml
  ├── sub-{ID}_ses-{XX}_run-{YY}_task-estimate.{mapping}_events.tsv (x8)
  ├── sub-{ID}_ses-{XX}_run-{YY}_task-estimate.{mapping}_expsettings.yml (x8)
  ├── reward_{ID}_{XX}_1.txt through reward_{ID}_{XX}_8.txt
  └── sub-{ID}_ses-{XX}_earnings_expsettings.yml
```

### Expected columns in events.tsv:
- event_type
- onset
- duration
- phase (trial phase name)
- orientation (degrees)
- value (assigned value)
- response (subject's slider response)
- response_time (RT in seconds)
- reward (points earned)

---

## 💡 OPTIONAL ENHANCEMENTS (Post-pilot)

1. **Add data checksum/validation** - Verify files aren't corrupted
2. **Implement autosave** - Save data every N trials
3. **Add escape key** - Allow experimenter to abort gracefully (exptools2 might already do this)
4. **Better error reporting** - Write errors to a separate error log file
5. **Add trial counter** - Show "Trial X of Y" during ITI
6. **Confirmation dialogs** - Ask "Start next run?" between runs
7. **Better "Too late" feedback** - Make it more salient
8. **Response validation** - Ensure marker was actually moved from initial position

---

## 🎯 SUMMARY

**Overall code quality:** Good  
**Ready for pilot:** Yes, with minor fixes  
**Major risks:** Low - mostly edge cases and error handling

The code is well-structured and follows good practices. The main recommendations are:
1. Remove debug code
2. Add error handling for file I/O
3. Test thoroughly on actual experimental setup
4. Monitor first few pilots closely for any issues

**Most important:** Run a complete test session yourself before the first real pilot!
