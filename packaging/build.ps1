# Build the RCA Desktop app and collector companion into distributable folders.
# Run from the repository root: .\packaging\build.ps1

$ErrorActionPreference = "Stop"

Write-Host "Cleaning previous build..."
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

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
    '--hidden-import', 'win32evtlog'
)

$collectorArgs += Get-Content 'packaging\excludes.txt' |
    Where-Object { $_ -and -not $_.StartsWith('#') } |
    ForEach-Object { '--exclude-module', $_ }

pyinstaller @collectorArgs
if ($LASTEXITCODE -ne 0) { throw "Collector build failed (exit $LASTEXITCODE)" }

Write-Host "Build complete: dist\RCA-Desktop\RCA-Desktop.exe and dist\RCA-Collector\RCA-Collector.exe"
$size = (Get-ChildItem -Recurse dist\RCA-Desktop, dist\RCA-Collector | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("Total distribution size: {0:N0} MB" -f $size)
