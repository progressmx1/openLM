# Build the React UI into desktop\frontend\dist when it is missing or older than its sources.
# Used by start.ps1 (Electron) and dev-backend.ps1 (browser at http://127.0.0.1:8765).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$frontend = Join-Path $root 'desktop\frontend'
$cache = Join-Path $root '.npm-cache'

# esbuild (used by Vite) writes temp files; some Windows setups deny writes to
# the OS temp dir, so redirect temp into the project during the build.
$env:TMP = Join-Path $root '.tmp'
$env:TEMP = $env:TMP
$env:TMPDIR = $env:TMP
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null

$distIndex = Join-Path $frontend 'dist\index.html'
$needsBuild = -not (Test-Path $distIndex)
if (-not $needsBuild) {
    $builtAt = (Get-Item $distIndex).LastWriteTime
    $sources = @(
        Get-ChildItem -Path (Join-Path $frontend 'src'), (Join-Path $frontend 'public') -Recurse -File -ErrorAction SilentlyContinue
        Get-Item (Join-Path $frontend 'index.html'), (Join-Path $frontend 'tailwind.config.js'), (Join-Path $frontend 'vite.config.ts')
    )
    $newer = $sources | Where-Object { $_.LastWriteTime -gt $builtAt } | Select-Object -First 1
    $needsBuild = $null -ne $newer
}
if ($needsBuild) {
    Write-Host "Building the UI..."
    Push-Location $frontend
    try {
        if (-not (Test-Path (Join-Path $frontend 'node_modules'))) {
            npm install --no-audit --no-fund --cache $cache
        }
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "UI build failed (npm run build exited with $LASTEXITCODE)." }
    } finally {
        Pop-Location
    }
}
