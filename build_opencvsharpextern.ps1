param(
    [int]$Jobs = 4
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Require-Command($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Required command '$name' not found in PATH."
    }
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Error @"
cmake was not found in PATH.

To fix this, choose one of the following:
  1. Install CMake via winget:
       winget install Kitware.CMake
     Then reopen this terminal so PATH is updated.
  2. Install CMake 3.20+ from https://cmake.org/download/ and add it to PATH.
  3. Launch this script from a developer shell that includes cmake in PATH.
"@
    exit 1
}
Require-Command git

# ---------------------------------------------------------------------------
# Detect Visual Studio generator via vswhere
# ---------------------------------------------------------------------------
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found at '$vswhere'. Install Visual Studio or Build Tools first."
}
$vsInstallVersion = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion 2>$null
$vsDisplayName    = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property displayName    2>$null
$vsInstallPath    = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath   2>$null
if (-not $vsInstallVersion) {
    throw "No Visual Studio installation with C++ tools found. Install 'Desktop development with C++' workload."
}
$vsMajor = [int]($vsInstallVersion.Split('.')[0])
$generatorMap = @{
    17 = "Visual Studio 17 2022"
    18 = "Visual Studio 18 2026"
}
$vsGenerator = $generatorMap[$vsMajor]
if (-not $vsGenerator) {
    throw "Unsupported Visual Studio major version: $vsMajor (from '$vsInstallVersion'). Visual Studio 2022 or 2026 is required."
}
Write-Host "Using generator: $vsGenerator ($vsDisplayName)"

# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
$sourceDir   = "$RepoRoot/src/OpenCvSharpExtern"
$buildDir    = "$RepoRoot/src/build"
$opencvDir   = "$RepoRoot/opencv_artifacts"



# --- 3. CONFIGURE ---
Write-Host "Configuring OpenCvSharpExtern..." -ForegroundColor Cyan
cmake `
    -S "$sourceDir" `
    -B "$buildDir" `
    -G "$vsGenerator" -A x64 `
    -D "CMAKE_GENERATOR_INSTANCE=$vsInstallPath" `
    -D "OpenCV_DIR=$opencvDir" `
    -D "ENABLED_CUDA=1"


