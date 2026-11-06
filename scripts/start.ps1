# Build the UI (if needed) and launch the Electron desktop app.
# The Electron shell starts the Python backend automatically.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$electron = Join-Path $root 'desktop\electron'
$cache = Join-Path $root '.npm-cache'

# 1. Build the React UI so the backend can serve it (only when sources changed).
& (Join-Path $PSScriptRoot 'build-ui.ps1')

# Warn (but continue) when no llama.cpp runtime is installed: models then run
# in-process on the CPU instead of per-GPU.
$runtimes = Join-Path $root 'runtimes\llama.cpp'
if (-not (Get-ChildItem -Path $runtimes -Recurse -Filter 'llama-server.exe' -ErrorAction SilentlyContinue | Select-Object -First 1)) {
    Write-Host "No llama.cpp GPU runtime found. Run .\scripts\install-llama-runtimes.ps1 -WithCuda to use the RTX 4070 and the eGPU." -ForegroundColor Yellow
}

# 2. Install Electron on first run.
if (-not (Test-Path (Join-Path $electron 'node_modules\electron'))) {
    Write-Host "Installing Electron (first run only, downloads ~120 MB)..."
    Push-Location $electron
    npm install --no-audit --no-fund --cache $cache
    Pop-Location
}

# 3. Launch the desktop app.
Push-Location $electron
npm start
Pop-Location
