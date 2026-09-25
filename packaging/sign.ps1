# Sign the built executables with a code-signing certificate.
#
# What this buys, and what it does not:
#   - A self-signed certificate (the default here, no cost, no paperwork)
#     proves the two executables in one release were not modified after
#     signing and both came from the same signer. It does NOT clear the
#     SmartScreen "unrecognised publisher" prompt -- that requires a
#     certificate from a CA in Microsoft's trusted root program (a paid
#     EV/OV code-signing certificate) plus enough install reputation with
#     Microsoft, neither of which a self-signed cert provides.
#   - If a real certificate is available (a .pfx from DigiCert, Sectigo,
#     SSL.com, etc.), pass -PfxPath to use it instead; that path does clear
#     SmartScreen once the certificate has built reputation.
#
# Usage:
#   .\packaging\sign.ps1                              # self-signed (default)
#   .\packaging\sign.ps1 -PfxPath cert.pfx -PfxPassword (Read-Host -AsSecureString)

param(
    [string]$PfxPath = "",
    [System.Security.SecureString]$PfxPassword = $null,
    [string]$SelfSignedCertPath = "packaging\localrca-selfsigned.pfx"
)

$ErrorActionPreference = "Stop"

function Find-SignTool {
    $candidates = Get-ChildItem -Path "C:\Program Files (x86)\Windows Kits\10\bin" `
        -Filter "signtool.exe" -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match "\\x64\\" }
    if (-not $candidates) { throw "signtool.exe not found. Install the Windows SDK." }
    return $candidates[0].FullName
}

$signtool = Find-SignTool
Write-Host "Using signtool: $signtool"

if ($PfxPath -and (Test-Path $PfxPath)) {
    Write-Host "Signing with the provided certificate: $PfxPath"
    $pfx = $PfxPath
    $pwdArgs = if ($PfxPassword) { @('/p', [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($PfxPassword))) } else { @() }
} else {
    Write-Host "No certificate provided -- using a self-signed certificate."
    Write-Host "This proves file integrity between build and download; it does" -ForegroundColor Yellow
    Write-Host "NOT remove the Windows SmartScreen warning (see script header)." -ForegroundColor Yellow

    if (-not (Test-Path $SelfSignedCertPath)) {
        Write-Host "Generating a new self-signed code-signing certificate..."
        $cert = New-SelfSignedCertificate `
            -Type CodeSigningCert `
            -Subject "CN=LocalRCA (self-signed, integrity only -- not a trusted publisher)" `
            -KeyUsage DigitalSignature `
            -FriendlyName "LocalRCA self-signed signing cert" `
            -CertStoreLocation "Cert:\CurrentUser\My" `
            -NotAfter (Get-Date).AddYears(3)
        $certPassword = ConvertTo-SecureString -String ([guid]::NewGuid().ToString()) -Force -AsPlainText
        Export-PfxCertificate -Cert $cert -FilePath $SelfSignedCertPath -Password $certPassword | Out-Null
        # Password is regenerated per run in-memory only, never written to
        # disk or printed -- the .pfx is protected by the same password for
        # this script's own immediate use, not meant for later reuse by hand.
        $script:GeneratedPassword = $certPassword
        Write-Host "Certificate saved to $SelfSignedCertPath (gitignored; do not commit it)."
    }
    $pfx = $SelfSignedCertPath
    if (-not $script:GeneratedPassword) {
        Write-Host ""
        Write-Host "Re-signing with an existing certificate from a prior run." -ForegroundColor Yellow
        Write-Host "That certificate's password was a random GUID generated in-memory" -ForegroundColor Yellow
        Write-Host "and never shown or saved -- it cannot be recovered. If you don't" -ForegroundColor Yellow
        Write-Host "have it, delete $SelfSignedCertPath and rerun this script to" -ForegroundColor Yellow
        Write-Host "generate a fresh certificate instead." -ForegroundColor Yellow
        Write-Host ""
        $script:GeneratedPassword = Read-Host -AsSecureString "Password for $SelfSignedCertPath"
    }
    $pwdArgs = @('/p', [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($script:GeneratedPassword)))
}

$targets = @(
    'dist\RCA-Desktop\RCA-Desktop.exe',
    'dist\RCA-Collector\RCA-Collector.exe'
)
foreach ($target in $targets) {
    if (-not (Test-Path $target)) { throw "$target is missing. Run .\packaging\build.ps1 first." }
}

foreach ($target in $targets) {
    Write-Host "Signing $target ..."
    # DigiCert's classic RFC 3161 timestamp endpoint is HTTP by protocol
    # design (no TLS variant); the RFC 3161 response itself is what's
    # authenticated, so this does not weaken the signature, but an HTTPS
    # timestamp authority closes the transport-level gap when one is
    # available. Kept as DigiCert's documented default here since it is the
    # most widely mirrored TSA and this is a self-signed, non-production cert.
    & $signtool sign /f $pfx @pwdArgs /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 $target
    if ($LASTEXITCODE -ne 0) { throw "Signing failed for $target (exit $LASTEXITCODE)" }
}

Write-Host ""
Write-Host "Signed. Verify with:  signtool verify /pa dist\RCA-Desktop\RCA-Desktop.exe"
