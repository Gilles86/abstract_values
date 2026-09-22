# Paths and the log-backup helper come from _common.ps1, which resolves
# everything relative to this folder -- no account-specific paths here.
. "$PSScriptRoot\_common.ps1"

Write-Host "Please enter the following details:"

$subject_id = Read-Host "Enter subject_id (integer)"
if (-not ($subject_id -match '^\d+$')) {
    Write-Host "Error: subject_id must be an integer."
    exit 1
}
$subject_id = [int]$subject_id

$session_id = Read-Host "Enter session_id (1 or 2)"
if (-not ($session_id -match '^[12]$')) {
    Write-Host "Error: session_id must be 1 or 2."
    exit 1
}
$session_id = [int]$session_id

if ($subject_id % 2 -eq 0) {
    $mapping = if ($session_id -eq 1) { "cdf" } else { "inverse_cdf" }
} else {
    $mapping = if ($session_id -eq 1) { "inverse_cdf" } else { "cdf" }
}

Write-Host ("Running single-subject experiment for subject {0}, session {1}: {2}" -f $subject_id, $session_id, $mapping)

Write-Host "Running examples..."
& $python "$expDir\examples.py" $subject_id $session_id $mapping --settings single_subject

Write-Host "Running training..."
& $python "$expDir\training.py" $subject_id $session_id $mapping --settings single_subject

# Archive the learning-phase logs too -- nothing used to copy these off the
# testing-room PC, which is why sourcedata holds no phase-1/2 files.
Copy-LogsToBackup

Read-Host