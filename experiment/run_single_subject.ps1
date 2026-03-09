$python = "C:\ExpFiles\gilles\psychopy\python.exe"
$expDir = "C:\Users\gdehol\abstract_values\experiment"

if (-not (Test-Path $python)) { throw "Python not found: $python" }
if (-not (Test-Path $expDir)) { throw "Experiment directory not found: $expDir" }

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
& $python "$expDir\examples.py" $subject_id $session_id $mapping

Write-Host "Running training..."
& $python "$expDir\training.py" $subject_id $session_id $mapping

Read-Host