# Create (or refresh) the dedicated build environment, .venv-build\.
#
# Why a dedicated venv: PyInstaller bundles whatever is importable from the
# interpreter it runs under. Building from the global Python picked up a CUDA
# build of torch and shipped 2.6 GB of NVIDIA DLLs the app never loads -- the
# v1.5.1 zip was 1.99 GB against 272 MB for v1.5.0. This venv holds exactly
# requirements-dev.lock (CPU-only torch, hash-checked) and nothing else.
#
# Usage (from the repository root):
#   .\packaging\make_venv.ps1            # create/sync .venv-build from the lock
#   .\packaging\make_venv.ps1 -Relock    # re-resolve requirements*.txt into the
#                                        # two .lock files first, then sync
#   .\packaging\make_venv.ps1 -Recreate  # delete the venv and start over
#
# Needs uv (https://docs.astral.sh/uv/): `pip install uv` or `winget install
# astral-sh.uv`. uv syncs the venv to the lock exactly -- stray packages are
# removed -- which plain pip cannot do.

param(
    [switch]$Relock,
    [switch]$Recreate,
    [string]$Python = "3.13",
    [string]$VenvDir = ".venv-build"
)

$ErrorActionPreference = "Stop"

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) { throw "uv is not on PATH. Install it with 'pip install uv' or 'winget install astral-sh.uv'." }
$uv = $uv.Source

if ($Relock) {
    # The locks are resolved for the platform the release targets, not for
    # whatever machine runs this: Windows x64, CPython 3.13.
    # unsafe-best-match lets torch come from the PyTorch CPU index while
    # everything else resolves from PyPI; the hashes then pin each artifact.
    $compileArgs = @(
        'pip', 'compile',
        '--generate-hashes',
        '--emit-index-url',
        '--index-strategy', 'unsafe-best-match',
        '--python-version', $Python,
        '--python-platform', 'x86_64-pc-windows-msvc',
        '--no-header',
        '--quiet'
    )
    foreach ($pair in @(@('requirements.txt', 'requirements.lock'),
                        @('requirements-dev.txt', 'requirements-dev.lock'))) {
        Write-Host "Resolving $($pair[0]) -> $($pair[1]) ..."
        & $uv @compileArgs $pair[0] -o $pair[1]
        if ($LASTEXITCODE -ne 0) { throw "uv pip compile $($pair[0]) failed (exit $LASTEXITCODE)" }
    }
}

if ($Recreate -and (Test-Path $VenvDir)) {
    Write-Host "Removing $VenvDir ..."
    Remove-Item -Recurse -Force $VenvDir
}

$venvPython = Join-Path $VenvDir 'Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating $VenvDir with Python $Python ..."
    & $uv venv $VenvDir --python $Python
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed (exit $LASTEXITCODE)" }
}

Write-Host "Syncing $VenvDir to requirements-dev.lock ..."
# sync, not install: anything not in the lock is uninstalled, so the build
# cannot quietly depend on a package somebody added by hand.
& $uv pip sync --python $venvPython --index-strategy unsafe-best-match --require-hashes requirements-dev.lock
if ($LASTEXITCODE -ne 0) { throw "uv pip sync failed (exit $LASTEXITCODE)" }

# The whole point of this venv. Fail here, in seconds, rather than after a
# ten-minute build that produces a 2 GB zip.
$torchCheck = 'import torch, sys; print(torch.__version__); sys.exit(1 if torch.version.cuda else 0)'
$torchVersion = & $venvPython -c $torchCheck
if ($LASTEXITCODE -ne 0) { throw "$VenvDir has a CUDA build of torch ($torchVersion); the app must ship the CPU wheel." }
Write-Host "Build venv ready: $venvPython (torch $torchVersion, CPU-only)"
