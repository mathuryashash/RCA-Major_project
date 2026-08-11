# Build the RCA Desktop app and collector companion into distributable folders.
# Run from the repository root: .\packaging\build.ps1

$ErrorActionPreference = "Stop"

# A previously built app still running holds its own _internal DLLs open, and
# Windows refuses to delete those, so the clean step failed with "Access is
# denied" for anyone who had the app open -- which is the normal case.
$running = Get-Process -Name RCA-Desktop, RCA-Collector -ErrorAction SilentlyContinue
if ($running) {
    Write-Host "Stopping running build: $(($running | ForEach-Object { $_.ProcessName }) -join ', ')"
    # SilentlyContinue: a process that exits on its own between the two calls
    # would otherwise abort the build, since $ErrorActionPreference escalates
    # Stop-Process's error to terminating.
    $running | Stop-Process -Force -ErrorAction SilentlyContinue
    # Wait for teardown rather than guessing: unmapping torch's DLLs can take
    # longer than a fixed sleep allows, and costs nothing when it is quick.
    $running | Wait-Process -Timeout 30 -ErrorAction SilentlyContinue
}

Write-Host "Cleaning previous build..."
# Windows does not release a memory-mapped DLL the instant its process exits,
# so the first delete after stopping the app can still fail even though
# nothing is really holding the file. Retry briefly before giving up: a single
# attempt failed the build, and simply removing again a moment later worked.
foreach ($leftover in 'build', 'dist') {
    for ($attempt = 1; $attempt -le 20 -and (Test-Path $leftover); $attempt++) {
        Remove-Item -Recurse -Force $leftover -ErrorAction SilentlyContinue
        if (Test-Path $leftover) { Start-Sleep -Milliseconds 500 }
    }
    if (Test-Path $leftover) {
        # Name the culprit rather than guessing: the retry has already proven
        # this is not a transient handle.
        Remove-Item -Recurse -Force $leftover -ErrorAction Stop
    }
}

# $ErrorActionPreference does not apply to native executables in PowerShell 5.1,
# so a failed PyInstaller run has to be caught by its exit code.
Write-Host "Running PyInstaller..."
pyinstaller packaging\rca_desktop.spec --noconfirm
if ($LASTEXITCODE -ne 0) { throw "Desktop build failed (exit $LASTEXITCODE)" }

$collectorArgs = @(
    'src\telemetry\collector_entry.py',
    '--name', 'RCA-Collector',
    '--console',
    '--onedir',
    '--noconfirm',
    '--distpath', 'dist',
    '--workpath', 'build\collector',
    '--specpath', 'build\collector',
    '--paths', 'src',
    '--additional-hooks-dir', 'packaging\hooks',
    '--hidden-import', 'win32evtlog',
    # Absolute: --specpath sends the spec to build\collector, and PyInstaller
    # resolves a relative --icon against the spec directory rather than the
    # working directory, so 'assets\logo.ico' became
    # 'build\collector\assets\logo.ico' and the build died looking for it.
    '--icon', (Resolve-Path 'assets\logo.ico').Path
)

$collectorArgs += Get-Content 'packaging\excludes.txt' |
    Where-Object { $_ -and -not $_.StartsWith('#') } |
    ForEach-Object { '--exclude-module', $_ }

pyinstaller @collectorArgs
if ($LASTEXITCODE -ne 0) { throw "Collector build failed (exit $LASTEXITCODE)" }

Write-Host "Build complete: dist\RCA-Desktop\RCA-Desktop.exe and dist\RCA-Collector\RCA-Collector.exe"
$size = (Get-ChildItem -Recurse dist\RCA-Desktop, dist\RCA-Collector | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("Total distribution size: {0:N0} MB" -f $size)
