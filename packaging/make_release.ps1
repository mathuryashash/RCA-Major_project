# Package a built dist\ into the ZIP people actually download.
#
# The executables alone are not a release: someone who unzips this has no
# licence, no notices for the LGPL components bundled inside, and no
# instructions. Run .\packaging\build.ps1 first, then this.
#
# Usage: .\packaging\make_release.ps1

$ErrorActionPreference = "Stop"

foreach ($required in 'dist\RCA-Desktop\RCA-Desktop.exe', 'dist\RCA-Collector\RCA-Collector.exe') {
    if (-not (Test-Path $required)) { throw "$required is missing. Run .\packaging\build.ps1 first." }
}

# One source of truth for the version: src\version.py, the same string the
# window title and the logs report.
$versionLine = Select-String -Path 'src\version.py' -Pattern '__version__\s*=\s*"([^"]+)"'
if (-not $versionLine) { throw "Could not read __version__ from src\version.py" }
$version = $versionLine.Matches[0].Groups[1].Value

$name = "LocalRCA-v$version-windows-x64"
$staging = "build\release\$name"
Write-Host "Staging $name ..."

Remove-Item -Recurse -Force "build\release" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $staging | Out-Null

Copy-Item -Recurse 'dist\RCA-Desktop'   "$staging\RCA-Desktop"
Copy-Item -Recurse 'dist\RCA-Collector' "$staging\RCA-Collector"

# Ship what a recipient is entitled to and what they need to get started.
foreach ($doc in 'LICENSE', 'THIRD-PARTY-NOTICES.md', 'INSTALL.md', 'README.md') {
    Copy-Item $doc $staging
}

# A first-run reader should not have to guess which file to open.
@"
LocalRCA v$version

Start here:  INSTALL.md
Run:         RCA-Desktop\RCA-Desktop.exe

The FIRST launch takes about a minute -- Windows scans several thousand
newly extracted files. Every launch after that takes under ten seconds.
Nothing is wrong; give it a minute.

Keep each .exe inside its own folder -- the adjacent _internal directory
holds its runtime. Move the whole folder if you need to relocate it.

This build is unsigned, so Windows may show a SmartScreen prompt. Verify the
release came from
https://github.com/mathuryashash/RCA-Major_project/releases
before choosing Run anyway.

Nothing is collected until you agree on first launch, and nothing is ever
uploaded: the application makes no network connections.
"@ | Set-Content "$staging\START-HERE.txt" -Encoding utf8

$zip = "dist\$name.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path $staging -DestinationPath $zip -CompressionLevel Optimal

$hash = (Get-FileHash $zip -Algorithm SHA256).Hash
$sizeMb = [math]::Round((Get-Item $zip).Length / 1MB, 1)

# Publish the checksum with the download: it is the only way a recipient of an
# unsigned build can tell they got the file this repository produced.
"$hash  $name.zip" | Set-Content "dist\$name.zip.sha256" -Encoding ascii

Write-Host ""
Write-Host "Release:  $zip  ($sizeMb MB)"
Write-Host "SHA256:   $hash"
Write-Host "Checksum written to dist\$name.zip.sha256"
