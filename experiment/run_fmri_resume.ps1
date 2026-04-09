$python     = "C:\ExpFiles\researchers\gdehol\psychopy\python.exe"
$expDir     = "C:\Users\gdehol\experiments\abstract_values\experiment"

if (-not (Test-Path $python))  { throw "Python not found: $python" }
if (-not (Test-Path $expDir))  { throw "Experiment directory not found: $expDir" }

Write-Host "=== RESUME MODE ==="
Write-Host "Use this script to continue an interrupted session."
Write-Host ""

# Prompt for subject_id and validate it's an integer
$subject_id = Read-Host "Enter subject_id (integer)"
if (-not ($subject_id -match '^\d+$')) {
    Write-Host "Error: subject_id must be an integer."
    exit 1
}
$subject_id = [int]$subject_id

# Prompt for session_id and validate it's 1 or 2
$session_id = Read-Host "Enter session_id (1 or 2)"
if (-not ($session_id -match '^[12]$')) {
    Write-Host "Error: session_id must be 1 or 2."
    exit 1
}
$session_id = [int]$session_id

# Counterbalanced mapping assignment
if ($subject_id % 2 -eq 0) {
    $mapping = if ($session_id -eq 1) { "cdf" } else { "inverse_cdf" }
} else {
    $mapping = if ($session_id -eq 1) { "inverse_cdf" } else { "cdf" }
}

Write-Host ("Subject {0}, session {1}, mapping: {2}" -f $subject_id, $session_id, $mapping)
Write-Host ""

# Ask what to resume from
Write-Host "What do you want to run?"
Write-Host "  t  = training only"
Write-Host "  1-8 = start from run N (e.g., '3' runs 3,4,5,6,7,8)"
Write-Host "  e  = earnings display only"
Write-Host "  all = full session (training + all 8 runs + earnings)"
$resume = Read-Host "Enter choice"

# Ask whether to use the eyetracker
$use_eyetracker = Read-Host "Use eyetracker? (y/n)"
$eyetracker_flag = if ($use_eyetracker -eq 'y') { '--eyetracker' } else { '' }

Write-Host ""

if ($resume -eq 't') {
    Write-Host "Running training only..."
    & $python "$expDir\training.py" $subject_id $session_id $mapping --settings sns_fmri --n_trials 36
}
elseif ($resume -eq 'e') {
    Write-Host "Running earnings display only..."
    & $python "$expDir\earnings.py" $subject_id $session_id --settings sns_fmri
}
elseif ($resume -eq 'all') {
    Write-Host "Running full session..."
    & $python "$expDir\training.py" $subject_id $session_id $mapping --settings sns_fmri --n_trials 36
    for ($run = 1; $run -le 8; $run++) {
        Write-Host ("Running main task - Run {0} of 8" -f $run)
        & $python "$expDir\task.py" $subject_id $session_id $run $mapping --settings sns_fmri $eyetracker_flag
    }
    & $python "$expDir\earnings.py" $subject_id $session_id --settings sns_fmri
}
elseif ($resume -match '^[1-8]$') {
    $start_run = [int]$resume
    Write-Host ("Resuming from run {0}..." -f $start_run)
    for ($run = $start_run; $run -le 8; $run++) {
        Write-Host ("Running main task - Run {0} of 8" -f $run)
        & $python "$expDir\task.py" $subject_id $session_id $run $mapping --settings sns_fmri $eyetracker_flag
    }
    Write-Host "Displaying total earnings..."
    & $python "$expDir\earnings.py" $subject_id $session_id --settings sns_fmri
}
else {
    Write-Host "Invalid choice. Exiting."
    exit 1
}

# Copy logs to network drive
Write-Host "Copying logs to T:\projects\2026\dehollander_bedi_ruff_abstract_values\data\sourcedata\behavior..."
Copy-Item -Path "$expDir\logs\sub-*" -Destination "T:\projects\2026\dehollander_bedi_ruff_abstract_values\data\sourcedata\behavior\" -Recurse -Force
Write-Host "Logs copied successfully!"

Read-Host
