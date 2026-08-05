# Build the RCA Desktop app and collector companion into distributable folders.
# Run from the repository root: .\packaging\build.ps1

$ErrorActionPreference = "Stop"

Write-Host "Cleaning previous build..."
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

Write-Host "Running PyInstaller..."
pyinstaller packaging\rca_desktop.spec --noconfirm

$collectorArgs = @(
    'src\telemetry\collector_entry.py',
    '--name', 'RCA-Collector',
    '--console',
    '--onedir',
    '--noconfirm',
    '--clean',
    '--distpath', 'dist',
    '--workpath', 'build\collector',
    '--specpath', 'build\collector',
    '--paths', 'src',
    '--additional-hooks-dir', 'packaging\hooks',
    '--hidden-import', 'win32evtlog'
)

Get-Content 'packaging\excludes.txt' |
    Where-Object { $_ -and -not $_.StartsWith('#') } |
    ForEach-Object {
        $collectorArgs += '--exclude-module'
        $collectorArgs += $_
    }

pyinstaller @collectorArgs

Write-Host "Build complete: dist\RCA-Desktop\RCA-Desktop.exe and dist\RCA-Collector\RCA-Collector.exe"
$size = (Get-ChildItem -Recurse dist\RCA-Desktop, dist\RCA-Collector | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("Total distribution size: {0:N0} MB" -f $size)
