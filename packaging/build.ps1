# Build the RCA Desktop app into a distributable folder.
# Run from the repository root: .\packaging\build.ps1

$ErrorActionPreference = "Stop"

Write-Host "Cleaning previous build..."
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

Write-Host "Running PyInstaller..."
pyinstaller packaging\rca_desktop.spec --noconfirm

Write-Host "Build complete: dist\RCA-Desktop\RCA-Desktop.exe"
$size = (Get-ChildItem -Recurse dist\RCA-Desktop | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("Total distribution size: {0:N0} MB" -f $size)
