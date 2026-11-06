# Run the backend test suite (uses a fake adapter; no model download needed).
$root = Split-Path -Parent $PSScriptRoot
Push-Location (Join-Path $root 'backend')
python -m pytest -q
Pop-Location
