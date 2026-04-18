# build_opencv_cuda.ps1
# Based on the professional OpenCV build workflow.

param(
    [int]$Jobs = 8  # Increased to 8 for faster CUDA compilation
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# --- 1. PREREQUISITE CHECKS ---
function Require-Command($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Required command '$name' not found in PATH."
    }
}
Require-Command cmake
Require-Command git

# Verify submodules
if (-not (Test-Path "$RepoRoot/opencv/CMakeLists.txt")) {
    throw "opencv submodule not found. Run: git submodule update --init --recursive"
}
if (-not (Test-Path "$RepoRoot/opencv_contrib/modules")) {
    throw "opencv_contrib submodule not found. Run: git submodule update --init --recursive"
}

# --- 2. DETECT VISUAL STUDIO ---
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found. Install Visual Studio or Build Tools first."
}
$vsInstallVersion = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion
$vsInstallPath    = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vsMajor = [int]($vsInstallVersion.Split('.')[0])
$generatorMap = @{ 17 = "Visual Studio 17 2022"; 18 = "Visual Studio 18 2026" }
$vsGenerator = $generatorMap[$vsMajor]

Write-Host "Using generator: $vsGenerator" -ForegroundColor Cyan

# --- 3. PATH CONFIGURATION ---
$buildDir     = "$RepoRoot/opencv/build-cuda"
$installDir   = "$RepoRoot/opencv_artifacts"
$optionsFile  = "$RepoRoot/cmake/opencv_build_options_cuda.cmake"

# Resolve vcpkg (for Tesseract/dependencies)
$vcpkgRoot = $env:VCPKG_INSTALLATION_ROOT
if (-not $vcpkgRoot) {
    $vcpkgCmd = Get-Command vcpkg -ErrorAction SilentlyContinue
    if ($vcpkgCmd) { $vcpkgRoot = Split-Path $vcpkgCmd.Source }
}
if (-not $vcpkgRoot) { throw "vcpkg not found. Please set VCPKG_INSTALLATION_ROOT." }
$vcpkgToolchain = "$vcpkgRoot/scripts/buildsystems/vcpkg.cmake"

$cudaRoot     = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
$cudnnInclude = "C:/Program Files/NVIDIA/CUDNN/v9.20/include/12.9"
$cudnnLib     = "C:/Program Files/NVIDIA/CUDNN/v9.20/lib/12.9/x64/cudnn.lib"
$videoSdkDir  = "D:/Video_Codec_SDK_13.0.37"

$vcpkgRoot = $env:VCPKG_INSTALLATION_ROOT
if (-not $vcpkgRoot) { $vcpkgRoot = Split-Path (Get-Command vcpkg).Source }
$vcpkgToolchain = "$vcpkgRoot/scripts/buildsystems/vcpkg.cmake"

$env:CUDA_PATH = $cudaRoot
$env:PATH = "$cudaRoot/bin;$cudaRoot/libnvvp;$env:PATH"

# --- 4. CONFIGURE (THE CMAKE CALL) ---
if (Test-Path $buildDir) {
    Write-Host "Removing old build directory for a clean CUDA build..." -ForegroundColor Yellow
    Remove-Item $buildDir -Recurse -Force
}
New-Item -ItemType Directory -Path $buildDir

Write-Host "Options file path: $optionsFile" -ForegroundColor Yellow
if (-not (Test-Path $optionsFile)) {
    throw "CMake options file NOT FOUND at: $optionsFile"
}

Write-Host "Configuring OpenCV with CUDA..." -ForegroundColor Cyan
cmake `
    -C "$optionsFile" `
    -S "$RepoRoot/opencv" `
    -B "$buildDir" `
    -G "$vsGenerator" -A x64 `
    -D "CUDA_TOOLKIT_ROOT_DIR=$cudaRoot" `
    -D "CUDA_NVCC_EXECUTABLE=$cudaRoot/bin/nvcc.exe" `
    -D "CUDNN_INCLUDE_DIR=$cudnnInclude" `
    -D "CUDNN_LIBRARY=$cudnnLib" `
    -D "VIDEO_CODEC_SDK_DIR=$videoSdkDir" `
    -D "OPENCV_EXTRA_MODULES_PATH=$RepoRoot/opencv_contrib/modules" `
    -D "CMAKE_INSTALL_PREFIX=$installDir" `
    -D "CMAKE_GENERATOR_INSTANCE=$vsInstallPath" `
    -D "CMAKE_TOOLCHAIN_FILE=$vcpkgToolchain" `
    -D "VCPKG_TARGET_TRIPLET=x64-windows" `
    -D "CUDA_NVCC_FLAGS=-allow-unsupported-compiler" 

# --- 5. BUILD & INSTALL ---
Write-Host "Building OpenCV CUDA (This will take a long time)..." -ForegroundColor Cyan
cmake --build "$buildDir" --config Release -j $Jobs
cmake --install "$buildDir" --config Release

Write-Host ""
Write-Host "Done! Your CUDA-enabled OpenCV is in: $installDir" -ForegroundColor Green
