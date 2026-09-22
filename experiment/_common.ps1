# Shared setup for every launcher in this folder. Dot-source it as the first
# line of a script:
#
#     . "$PSScriptRoot\_common.ps1"
#
# Nothing here is machine- or account-specific. The experiment directory is
# simply wherever this file lives, and the interpreter is the uv-managed
# virtual environment sitting next to it -- so the same checkout works under
# any Windows account, on any stim PC, from any folder.

$expDir = $PSScriptRoot
$python = Join-Path $expDir ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw @"
No Python environment found at:
    $python

Create it once, from $expDir :
    uv sync

(See the 'Installation on a stimulus PC' section of README.md.)
"@
}

# Where finished logs are archived after a session: the project folder on the
# department share. The drive letter differs per machine (it was T:\ on the old
# stim PC, Z:\ on the current one), so override it without editing this file:
#
#     $env:ABSTRACT_VALUES_BACKUP = "T:\projects\2026\...\sourcedata\behavior"
#
if ($env:ABSTRACT_VALUES_BACKUP) {
    $backupDir = $env:ABSTRACT_VALUES_BACKUP
} else {
    $backupDir = "Z:\projects\2026\dehollander_bedi_ruff_abstract_values\data\sourcedata\behavior"
}

# Copy this session's logs to $backupDir. Never fails the session: the local
# copy in $expDir\logs is always the primary one, the share is the backup.
function Copy-LogsToBackup {
    $source = Join-Path $expDir "logs\sub-*"

    if (-not (Test-Path $source)) {
        Write-Host "No logs found in $expDir\logs - nothing to copy." -ForegroundColor Yellow
        return
    }

    if (-not (Test-Path $backupDir)) {
        Write-Host ""
        Write-Host "WARNING: backup location not reachable:" -ForegroundColor Red
        Write-Host "  $backupDir" -ForegroundColor Red
        Write-Host "Is the drive mapped in THIS Windows account? Logs are kept locally in" -ForegroundColor Red
        Write-Host "  $expDir\logs" -ForegroundColor Red
        Write-Host "and must be copied over by hand." -ForegroundColor Red
        Write-Host ""
        return
    }

    Write-Host "Copying logs to $backupDir ..."
    try {
        Copy-Item -Path $source -Destination "$backupDir\" -Recurse -Force -ErrorAction Stop
        Write-Host "Logs copied successfully!" -ForegroundColor Green
    } catch {
        Write-Host "Copy FAILED: $_" -ForegroundColor Red
        Write-Host "Logs are still in $expDir\logs - copy them by hand." -ForegroundColor Red
    }
}
