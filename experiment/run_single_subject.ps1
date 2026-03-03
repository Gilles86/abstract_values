param (
    [string]$subject_id = "06",
    [switch]$practice
)

$python      = "C:\ExpFiles\gilles\psychopy\python.exe"
$entryScript = "C:\Users\gdehol\abstract_values\experiment\examples.py"

if (-not (Test-Path $python))      { throw "Python not found: $python" }
if (-not (Test-Path $entryScript)) { throw "Entry script not found: $entryScript" }

& $python $entryScript $subject_id 1 cdf