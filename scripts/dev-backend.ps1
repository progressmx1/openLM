# Run the backend and serve the UI in a normal browser at http://127.0.0.1:8765
# (builds the UI first if its sources changed).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build-ui.ps1')
Write-Host "Open http://127.0.0.1:8765 in your browser."
Push-Location (Join-Path $root 'backend')
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
Pop-Location
