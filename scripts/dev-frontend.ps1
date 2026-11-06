# Run the Vite dev server for the frontend (http://localhost:5173).
# Requires the backend to be running on 127.0.0.1:8765 (scripts/dev-backend.ps1).
$root = Split-Path -Parent $PSScriptRoot
Push-Location (Join-Path $root 'desktop\frontend')
npm run dev
Pop-Location
