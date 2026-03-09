$python     = "C:\ExpFiles\researchers\gdehol\psychopy\python.exe"
$expDir     = "C:\Users\gdehol\experiments\abstract_values\experiment"

if (-not (Test-Path $python))  { throw "Python not found: $python" }
if (-not (Test-Path $expDir))  { throw "Experiment directory not found: $expDir" }

Write-Host "Hello! Please enter the following details:"

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

# Counterbalanced mapping assignment (matches run_task.ps1)
if ($subject_id % 2 -eq 0) {
    $mapping = if ($session_id -eq 1) { "cdf" } else { "inverse_cdf" }
} else {
    $mapping = if ($session_id -eq 1) { "inverse_cdf" } else { "cdf" }
}

Write-Host ("Running fMRI experiment for subject {0}, session {1}: {2}" -f $subject_id, $session_id, $mapping)

# Practice run during anatomical scan (~5 min, no scanner trigger needed)
Write-Host "Running practice run during anatomical scan (~5 min)..."
& $python "$expDir\training.py" $subject_id $session_id $mapping --settings sns_fmri --n_trials 30

# Run the main task for 9 runs
for ($run = 1; $run -le 9; $run++) {
    Write-Host ("Running main task - Run {0} of 9" -f $run)
    & $python "$expDir\task.py" $subject_id $session_id $run $mapping --settings sns_fmri
}

# Display total earnings
Write-Host "Displaying total earnings..."
& $python "$expDir\earnings.py" $subject_id $session_id --settings sns_fmri

# Copy logs to network drive
Write-Host "Copying logs to N:\client_write\gilles\experiment\logs..."
Copy-Item -Path "$expDir\logs\sub-*" -Destination "N:\client_write\gilles\experiment\logs\" -Recurse -Force
Write-Host "Logs copied successfully!"

Read-Host
