# Test script for copying logs to the backup share.
# $python, $expDir and Copy-LogsToBackup, all relative to this folder
. "$PSScriptRoot\_common.ps1"

Write-Host "=== Testing Log Copy Operation ==="
Write-Host ""

$source = Join-Path $expDir "logs\sub-*"

# Show what's in the local logs folder
Write-Host "=== LOCAL LOGS (what will be copied) ==="
if (Test-Path $source) {
    Get-ChildItem -Path $source -Directory | ForEach-Object { Write-Host $_.Name }
} else {
    Write-Host "No sub-* folders found in $expDir\logs"
}
Write-Host ""

# Show what's currently in the destination
Write-Host "=== CURRENT DESTINATION ($backupDir) ==="
if (Test-Path $backupDir) {
    Get-ChildItem -Path $backupDir -Directory | ForEach-Object { Write-Host $_.Name }
} else {
    Write-Host "Destination not reachable (drive not mapped in this account?)"
}
Write-Host ""

# Confirm before copying
Write-Host "This will copy all sub-* folders from $expDir\logs to $backupDir"
Write-Host "Existing files with same names will be overwritten."
Write-Host "Other subject folders already at the destination will NOT be affected."
Write-Host ""
$confirm = Read-Host "Proceed with copy? (yes/no)"

if ($confirm -eq "yes") {
    Write-Host ""
    Copy-LogsToBackup
    Write-Host ""

    # Show what's in destination after copy
    if (Test-Path $backupDir) {
        Write-Host "=== DESTINATION AFTER COPY ==="
        Get-ChildItem -Path $backupDir -Directory | ForEach-Object { Write-Host $_.Name }
    }
} else {
    Write-Host "Copy cancelled."
}

Write-Host ""
Read-Host "Press Enter to close"
