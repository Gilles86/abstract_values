# Quick test script to verify experimental setup
# Run this before each experimental session to catch issues early

# Paths and the backup destination come from _common.ps1, which resolves
# everything relative to this folder -- no account-specific paths here.
. "$PSScriptRoot\_common.ps1"

Write-Host "=== Abstract Values Experiment - Quick System Check ===" -ForegroundColor Cyan
Write-Host ""

$test_subject = "test"
$test_session = 99
$test_mapping = "cdf"

# Test 1: Check the uv environment
Write-Host "[1/6] Checking Python environment..." -ForegroundColor Yellow
try {
    $pythonVersion = & $python --version
    Write-Host "  [OK] $python" -ForegroundColor Green
    Write-Host "  [OK] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] Could not run $python" -ForegroundColor Red
    Write-Host "  Run 'uv sync' in $expDir first." -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
    exit 1
}

# Test 2: Check required Python packages
Write-Host "[2/6] Checking Python packages..." -ForegroundColor Yellow
$packages = @("psychopy", "exptools2", "numpy", "pandas", "yaml")
$missing = @()
foreach ($pkg in $packages) {
    & $python -c "import $pkg" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] $pkg installed" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] $pkg NOT found" -ForegroundColor Red
        $missing += $pkg
    }
}
if ($missing.Count -gt 0) {
    Write-Host "  Missing packages: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "  Fix with: uv sync" -ForegroundColor Red
    exit 1
}

# Test 3: Check settings file
Write-Host "[3/6] Checking settings file..." -ForegroundColor Yellow
if (Test-Path (Join-Path $expDir "settings\sns_multisubject.yml")) {
    Write-Host "  [OK] Settings file found" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] settings\sns_multisubject.yml not found!" -ForegroundColor Red
    exit 1
}

# Test 4: Check the log backup destination
Write-Host "[4/6] Checking backup destination..." -ForegroundColor Yellow
Write-Host "  $backupDir" -ForegroundColor Cyan
if (Test-Path $backupDir) {
    Write-Host "  [OK] Backup location reachable" -ForegroundColor Green
    try {
        $probe = Join-Path $backupDir "test_write.tmp"
        "test" | Out-File $probe -ErrorAction Stop
        Remove-Item $probe
        Write-Host "  [OK] Backup location is writable" -ForegroundColor Green
    } catch {
        Write-Host "  [FAIL] Backup location is not writable!" -ForegroundColor Red
        Write-Host "  Error: $_" -ForegroundColor Red
    }
} else {
    Write-Host "  [FAIL] Backup location not reachable!" -ForegroundColor Red
    Write-Host "  Is the drive mapped in THIS Windows account? Logs will NOT be backed up." -ForegroundColor Red
}

# Test 5: Run a mini experiment (3 trials)
Write-Host "[5/6] Running mini test experiment (3 trials)..." -ForegroundColor Yellow
Write-Host "  This will open a window - press SPACE to continue through screens" -ForegroundColor Cyan
Write-Host "  You'll see 3 trials, just click anywhere during response phase" -ForegroundColor Cyan

try {
    & $python "$expDir\task.py" $test_subject $test_session 99 $test_mapping --settings sns_multisubject --n_trials 3
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Test experiment completed successfully" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Test experiment failed!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "  [FAIL] Test experiment crashed!" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
    exit 1
}

# Test 6: Check that log files were created
Write-Host "[6/6] Verifying log files were created..." -ForegroundColor Yellow
$log_dir = Join-Path $expDir "logs\sub-$test_subject\ses-$test_session"
if (Test-Path "$log_dir\*events.tsv") {
    Write-Host "  [OK] Event log file created" -ForegroundColor Green

    # Check reward file
    if (Test-Path "$log_dir\reward_*.txt") {
        $reward_content = Get-Content "$log_dir\reward_*.txt"
        Write-Host "  [OK] Reward file created: $reward_content CHF" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Reward file not found!" -ForegroundColor Red
    }
} else {
    Write-Host "  [FAIL] Log files not created in $log_dir" -ForegroundColor Red
    exit 1
}

# Cleanup test files
Write-Host ""
Write-Host "Cleaning up test files..." -ForegroundColor Yellow
$test_sub_dir = Join-Path $expDir "logs\sub-$test_subject"
if (Test-Path $test_sub_dir) {
    Remove-Item -Path $test_sub_dir -Recurse -Force
    Write-Host "  Test logs cleaned up" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== ALL CHECKS PASSED ===" -ForegroundColor Green
Write-Host ""
Write-Host "System is ready for experiment!" -ForegroundColor Green
Write-Host "Remember to:" -ForegroundColor Cyan
Write-Host "  - Close other applications" -ForegroundColor Cyan
Write-Host "  - Ensure subject is comfortable" -ForegroundColor Cyan
Write-Host "  - Check subject ID and session number before starting" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to close"
