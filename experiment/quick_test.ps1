# Quick test script to verify experimental setup
# Run this before each experimental session to catch issues early

Write-Host "=== Abstract Values Experiment - Quick System Check ===" -ForegroundColor Cyan
Write-Host ""

$test_subject = "test"
$test_session = 99
$test_mapping = "cdf"

# Test 1: Check Python virtual environment
Write-Host "[1/6] Checking Python environment..." -ForegroundColor Yellow
try {
    & "c:\Expfiles\gilles\venv\exptools\Scripts\Activate.ps1"
    $pythonVersion = python --version
    Write-Host "  ✓ Python environment active: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Failed to activate Python environment!" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
    exit 1
}

# Test 2: Check required Python packages
Write-Host "[2/6] Checking Python packages..." -ForegroundColor Yellow
$packages = @("psychopy", "numpy", "pandas", "pyyaml")
$missing = @()
foreach ($pkg in $packages) {
    $check = python -c "import $pkg" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ $pkg installed" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $pkg NOT found" -ForegroundColor Red
        $missing += $pkg
    }
}
if ($missing.Count -gt 0) {
    Write-Host "  Missing packages: $($missing -join ', ')" -ForegroundColor Red
    exit 1
}

# Test 3: Check settings file
Write-Host "[3/6] Checking settings file..." -ForegroundColor Yellow
if (Test-Path "settings\sns_multisubject.yml") {
    Write-Host "  ✓ Settings file found" -ForegroundColor Green
} else {
    Write-Host "  ✗ settings\sns_multisubject.yml not found!" -ForegroundColor Red
    exit 1
}

# Test 4: Check network drive access
Write-Host "[4/6] Checking network drive access..." -ForegroundColor Yellow
if (Test-Path "N:\client_write\gilles\experiment") {
    Write-Host "  ✓ Network drive N:\ accessible" -ForegroundColor Green
    # Try to create a test file
    try {
        "test" | Out-File "N:\client_write\gilles\experiment\test_write.tmp" -ErrorAction Stop
        Remove-Item "N:\client_write\gilles\experiment\test_write.tmp"
        Write-Host "  ✓ Network drive is writable" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Network drive is not writable!" -ForegroundColor Red
        Write-Host "  Error: $_" -ForegroundColor Red
    }
} else {
    Write-Host "  ✗ Network drive N:\client_write\gilles\experiment not accessible!" -ForegroundColor Red
    Write-Host "  Logs will NOT be backed up!" -ForegroundColor Red
}

# Test 5: Run a mini experiment (3 trials)
Write-Host "[5/6] Running mini test experiment (3 trials)..." -ForegroundColor Yellow
Write-Host "  This will open a window - press SPACE to continue through screens" -ForegroundColor Cyan
Write-Host "  You'll see 3 trials, just click anywhere during response phase" -ForegroundColor Cyan

try {
    python task.py $test_subject $test_session 99 $test_mapping --settings sns_multisubject --n_trials 3
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Test experiment completed successfully" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Test experiment failed!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "  ✗ Test experiment crashed!" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
    exit 1
}

# Test 6: Check that log files were created
Write-Host "[6/6] Verifying log files were created..." -ForegroundColor Yellow
$log_dir = "logs\sub-$test_subject\session-$test_session"
if (Test-Path "$log_dir\*events.tsv") {
    Write-Host "  ✓ Event log file created" -ForegroundColor Green
    
    # Check reward file
    if (Test-Path "$log_dir\reward_*.txt") {
        $reward_content = Get-Content "$log_dir\reward_*.txt"
        Write-Host "  ✓ Reward file created: $reward_content CHF" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Reward file not found!" -ForegroundColor Red
    }
} else {
    Write-Host "  ✗ Log files not created!" -ForegroundColor Red
    exit 1
}

# Cleanup test files
Write-Host ""
Write-Host "Cleaning up test files..." -ForegroundColor Yellow
if (Test-Path $log_dir) {
    Remove-Item -Path $log_dir -Recurse -Force
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
