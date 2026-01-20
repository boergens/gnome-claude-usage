# Claude Usage System Tray - PowerShell launcher

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check for Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Python not found. Please install Python 3.8+"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check for required packages
$checkPackages = python -c "import pystray, PIL" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing required packages..."
    pip install pystray pillow torch
}

# Run the tray app
python claude_usage_tray.py
