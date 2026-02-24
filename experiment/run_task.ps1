

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

# Set mapping variable
if ($subject_id % 2 -eq 0) {
    $mapping = if ($session_id -eq 1) { "cdf" } else { "inverse_cdf" }
} else {
    $mapping = if ($session_id -eq 1) { "inverse_cdf" } else { "cdf" }
}

Write-Host "Activating virtual environment..."
# & "n:\client_write\gilles\venv\exptools\Scripts\Activate.ps1"
& "c:\Expfiles\gilles\venv\exptools\Scripts\Activate.ps1"

# Run the Python command

Write-Host ("Running experiment for subject {0}, session {1}: {2}" -f $subject_id, $session_id, $mapping)

python examples.py $subject_id $session_id $mapping --settings sns_multisubject

python training.py $subject_id $session_id $mapping --settings sns_multisubject

# Run the main task for 10 runs
for ($run = 1; $run -le 10; $run++) {
    Write-Host ("Running main task - Run {0} of 10" -f $run)
    python task.py $subject_id $session_id $run $mapping --settings sns_multisubject
}

# Display total earnings
Write-Host "Displaying total earnings..."
python earnings.py $subject_id $session_id --settings sns_multisubject

Read-Host