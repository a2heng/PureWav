# PureWav - embedded Python 3.8 bootstrap (Windows)
# Downloads and installs full Python 3.8 (with tkinter) into py38\.
# Idempotent: skips if py38\python.exe already exists.
#
# Usage: powershell -ExecutionPolicy Bypass -File bootstrap_python38.ps1
$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyDir = Join-Path $root "py38"
$py = Join-Path $pyDir "python.exe"
$pyVer = "3.8.10"

# 1. Install Python 3.8 (full installer, includes tkinter)
if (-not (Test-Path $py)) {
    Write-Host "==> Downloading Python $pyVer..."
    $dlDir = Join-Path $pyDir "_dl"
    New-Item -ItemType Directory -Force -Path $dlDir | Out-Null
    $installer = Join-Path $dlDir "python-installer.exe"
    & curl.exe -fsSL -o $installer "https://www.python.org/ftp/python/$pyVer/python-$pyVer-amd64.exe"
    Write-Host "==> Installing Python $pyVer..."
    & $installer /quiet InstallAllUsers=0 TargetDir="$pyDir" PrependPath=0 Include_launcher=0 Include_test=0
    Remove-Item $dlDir -Recurse -Force
    Write-Host "  Python installed: $py"
}

# 2. Upgrade pip (official installer ships old pip 21.1.1 with SSL issues)
& $py -m pip install --upgrade pip 2>$null
if ($LASTEXITCODE -ne 0) {
    # Fallback: download pip wheel with system Python, then install offline
    Write-Host "==> Upgrading pip (offline fallback)..."
    $dlDir = Join-Path $pyDir "_dl"
    New-Item -ItemType Directory -Force -Path $dlDir | Out-Null
    python -m pip download "pip==23.3.2" -d $dlDir --no-deps -q 2>$null
    $whl = (Get-ChildItem "$dlDir\pip*whl").FullName
    & $py -m pip install $whl --force-reinstall --no-index 2>$null
    Remove-Item $dlDir -Recurse -Force
}

# 3. Install deps (use --proxy="" to bypass system proxy if needed)
& $py -m pip install soundfile numpy onnxruntime tkinterdnd2 -q --proxy=""

Write-Host "Ready: $py"
