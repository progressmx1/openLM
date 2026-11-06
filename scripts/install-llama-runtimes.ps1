<#
.SYNOPSIS
    Downloads prebuilt llama.cpp runtimes (llama-server) into runtimes\llama.cpp.

.DESCRIPTION
    The app runs every model in its own llama-server process, pinned to one GPU
    (or split across several). This script installs the builds it needs:

      vulkan  (default, ~40 MB)  Sees every GPU: the AMD eGPU, the RTX 4070 and
                                 the Intel iGPU. Required for the AMD card and
                                 for splitting one model across NVIDIA + AMD.
      cuda    (-WithCuda, ~670 MB) Fastest option for the RTX 4070. When present,
                                 the app prefers it for the NVIDIA card.

    Nothing is installed system-wide; delete runtimes\ to remove everything.

.PARAMETER Build
    llama.cpp build tag. Defaults to the build the app was tested with.

.PARAMETER WithCuda
    Also install the CUDA 12.4 build (plus its CUDA runtime DLLs).

.PARAMETER Force
    Re-download even if a runtime folder already exists.

.EXAMPLE
    .\scripts\install-llama-runtimes.ps1
    .\scripts\install-llama-runtimes.ps1 -WithCuda
#>
param(
    [string]$Build = 'b11960',
    [switch]$WithCuda,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # the progress bar makes downloads very slow
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root 'runtimes\llama.cpp'
$downloads = Join-Path $root '.tmp\llama-downloads'
$base = "https://github.com/ggml-org/llama.cpp/releases/download/$Build"
New-Item -ItemType Directory -Force -Path $dest, $downloads | Out-Null

function Install-Runtime([string]$name, [string[]]$assets) {
    $target = Join-Path $dest $name
    $exe = Join-Path $target 'llama-server.exe'
    if ((Test-Path $exe) -and -not $Force) {
        Write-Host "[$name] already installed at $target (use -Force to reinstall)."
        return
    }
    if (Test-Path $target) { Remove-Item -Recurse -Force $target }
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    foreach ($asset in $assets) {
        $zip = Join-Path $downloads $asset
        if (-not (Test-Path $zip)) {
            Write-Host "[$name] downloading $asset ..."
            Invoke-WebRequest -Uri "$base/$asset" -OutFile "$zip.part" -UseBasicParsing
            Move-Item -Force "$zip.part" $zip
        }
        Write-Host "[$name] extracting $asset ..."
        Expand-Archive -Path $zip -DestinationPath $target -Force
    }
    # Release zips sometimes nest everything in one folder; flatten it.
    if (-not (Test-Path $exe)) {
        $found = Get-ChildItem -Path $target -Recurse -Filter 'llama-server.exe' | Select-Object -First 1
        if ($found) {
            Get-ChildItem -Path $found.DirectoryName | Move-Item -Destination $target -Force
        }
    }
    if (-not (Test-Path $exe)) { throw "[$name] llama-server.exe not found after extracting." }
    Get-ChildItem -Path $target -Recurse | Unblock-File -ErrorAction SilentlyContinue
    Set-Content -Path (Join-Path $target 'BUILD.txt') -Value $Build
    Write-Host "[$name] installed: $exe"
}

Install-Runtime 'vulkan' @("llama-$Build-bin-win-vulkan-x64.zip")
if ($WithCuda) {
    Install-Runtime 'cuda' @("llama-$Build-bin-win-cuda-12.4-x64.zip", 'cudart-llama-bin-win-cuda-12.4-x64.zip')
}

Write-Host ''
Write-Host 'GPUs each runtime can see:'
# llama.cpp logs to stderr; don't let Windows PowerShell treat that as a failure.
$ErrorActionPreference = 'Continue'
foreach ($dir in Get-ChildItem -Path $dest -Directory) {
    $exe = Join-Path $dir.FullName 'llama-server.exe'
    if (Test-Path $exe) {
        Write-Host "--- $($dir.Name) ---"
        & $exe --list-devices 2>&1 | ForEach-Object { "$_" } |
            Where-Object { $_ -match '^\s+\S+\d+: ' -or $_ -match '^Available devices' } |
            ForEach-Object { Write-Host $_ }
    }
}
Write-Host ''
Write-Host 'Done. Restart the app (scripts\start.ps1); models can now be placed per GPU in Manage models.'
