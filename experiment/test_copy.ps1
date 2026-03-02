# Test script for copying logs to network drive

Write-Host "=== Testing Log Copy Operation ==="
Write-Host ""

# Show what's in the local logs folder
Write-Host "=== LOCAL LOGS (what will be copied) ==="
if (Test-Path "logs\sub-*") {
    Get-ChildItem -Path "logs\sub-*" -Directory | ForEach-Object { Write-Host $_.Name }
} else {
    Write-Host "No sub-* folders found in logs\"
}
Write-Host ""

# Show what's currently in the destination
Write-Host "=== CURRENT DESTINATION (N:\client_write\gilles\experiment\logs) ==="
if (Test-Path "N:\client_write\gilles\experiment\logs") {
    Get-ChildItem -Path "N:\client_write\gilles\experiment\logs" -Directory | ForEach-Object { Write-Host $_.Name }
} else {
    Write-Host "Destination path doesn't exist yet"
}
Write-Host ""

# Confirm before copying
Write-Host "This will copy all sub-* folders from logs\ to N:\client_write\gilles\experiment\logs\"
Write-Host "Existing files with same names will be overwritten."
Write-Host "Other subject folders already on N:\ will NOT be affected."
Write-Host ""
$confirm = Read-Host "Proceed with copy? (yes/no)"

if ($confirm -eq "yes") {
    Write-Host ""
    Write-Host "Copying logs to N:\client_write\gilles\experiment\logs..."
    Copy-Item -Path "logs\sub-*" -Destination "N:\client_write\gilles\experiment\logs\" -Recurse -Force
    Write-Host "Logs copied successfully!"
    Write-Host ""
    
    # Show what's in destination after copy
    Write-Host "=== DESTINATION AFTER COPY ==="
    Get-ChildItem -Path "N:\client_write\gilles\experiment\logs" -Directory | ForEach-Object { Write-Host $_.Name }
} else {
    Write-Host "Copy cancelled."
}

Write-Host ""
Read-Host "Press Enter to close"
