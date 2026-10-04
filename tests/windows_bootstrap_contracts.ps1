$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$source = [IO.File]::ReadAllText((Join-Path $root 'Install-JARVIS.cmd'))
# cmd.exe does not join caret/newline pairs inside an open double quote.
# Require one physical command line, then exercise that line in Windows CI.
$match = [regex]::Match($source, '(?m)^powershell\.exe[^\r\n]*-Command "(?<code>[^\r\n]*)"\r?$')
if (-not $match.Success) { throw 'Cannot locate the executable PowerShell command in Install-JARVIS.cmd.' }
$code = $match.Groups['code'].Value
if (($code.Length + 80) -ge 8191) { throw 'The bootstrap exceeds the Windows cmd command-line limit.' }
$tokens = $null
$parseErrors = $null
[void][Management.Automation.Language.Parser]::ParseInput($code, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw ('Invalid PowerShell syntax: ' + ($parseErrors -join '; ')) }

$scratch = Join-Path ([IO.Path]::GetTempPath()) ('jarvis-bootstrap-tests-' + [Guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($scratch)
$shell = (Get-Process -Id $PID).Path
$utf8 = New-Object Text.UTF8Encoding($false)
$cases = @(
    'valid', 'release-order', 'draft', 'missing-asset', 'foreign-url',
    'wrong-manifest-size', 'wrong-manifest-hash', 'wrong-commit', 'wrong-tag',
    'wrong-artifact', 'invalid-hash', 'truncated-download', 'wrong-hash',
    'wrong-github-hash', 'not-executable', 'invalid-pe', 'oversize', 'invalid-date'
)
$passed = 0
$previousFixture = $env:JARVIS_BOOTSTRAP_FIXTURE
try {
    foreach ($case in $cases) {
        $fixture = Join-Path $scratch $case
        [void][IO.Directory]::CreateDirectory($fixture)
        $bytes = New-Object byte[] 1024
        $bytes[0] = 77; $bytes[1] = 90; $bytes[60] = 128; $bytes[128] = 80; $bytes[129] = 69
        if ($case -eq 'not-executable') { $bytes[0] = 0 }
        if ($case -eq 'invalid-pe') { $bytes[128] = 0 }
        $exe = Join-Path $fixture 'installer.bin'
        [IO.File]::WriteAllBytes($exe, $bytes)
        $digest = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
        $sha = 'a' * 40
        $tag = 'jarvis-preview-' + $sha.Substring(0, 12)
        $base = 'https://github.com/cutebody42-web/JARVIS-OS-V.2/releases/download/' + $tag + '/'
        $manifest = [ordered]@{
            commit_sha = $sha; release_tag = $tag; artifact_name = 'JARVIS-Setup.exe'; artifact_sha256 = $digest
        }
        if ($case -eq 'wrong-commit') { $manifest.commit_sha = 'b' * 40 }
        if ($case -eq 'wrong-tag') { $manifest.release_tag = 'another-release' }
        if ($case -eq 'wrong-artifact') { $manifest.artifact_name = 'different.exe' }
        if ($case -eq 'invalid-hash') { $manifest.artifact_sha256 = 'not-a-sha256' }
        if ($case -eq 'wrong-hash') { $manifest.artifact_sha256 = 'b' * 64 }
        $manifestPath = Join-Path $fixture 'manifest.json'
        [IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Compress), $utf8)
        $manifestDigest = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $setupAsset = [ordered]@{
            name = 'JARVIS-Setup.exe'; size = $bytes.Length
            browser_download_url = $base + 'JARVIS-Setup.exe'; digest = 'sha256:' + $digest
        }
        $manifestAsset = [ordered]@{
            name = 'jarvis-update.json'; size = (Get-Item -LiteralPath $manifestPath).Length
            browser_download_url = $base + 'jarvis-update.json'; digest = 'sha256:' + $manifestDigest
        }
        if ($case -eq 'foreign-url') { $setupAsset.browser_download_url = 'https://example.invalid/JARVIS-Setup.exe' }
        if ($case -eq 'truncated-download') { $setupAsset.size += 1 }
        if ($case -eq 'wrong-github-hash') { $setupAsset.digest = 'sha256:' + ('b' * 64) }
        if ($case -eq 'wrong-manifest-size') { $manifestAsset.size += 1 }
        if ($case -eq 'wrong-manifest-hash') { $manifestAsset.digest = 'sha256:' + ('b' * 64) }
        if ($case -eq 'oversize') { $setupAsset.size = 1073741825 }
        $release = [ordered]@{
            draft = ($case -eq 'draft'); tag_name = $tag; published_at = '2026-10-04T01:00:00Z'
            assets = @($setupAsset, $manifestAsset)
        }
        if ($case -eq 'missing-asset') { $release.assets = @($setupAsset) }
        if ($case -eq 'invalid-date') { $release.published_at = 'not-a-date' }
        $releases = @($release)
        if ($case -eq 'release-order') {
            # GitHub can return an older publication first; the newest valid publication must win.
            $older = [ordered]@{
                draft = $false; tag_name = 'jarvis-preview-bbbbbbbbbbbb'; published_at = '2026-10-03T23:59:59Z'; assets = @()
            }
            $releases = @($older, $release)
        }
        [IO.File]::WriteAllText((Join-Path $fixture 'releases.json'), (ConvertTo-Json -InputObject $releases -Depth 8 -Compress), $utf8)
        $prefix = @'
$ErrorActionPreference = 'Stop'
$env:LOCALAPPDATA = Join-Path $env:JARVIS_BOOTSTRAP_FIXTURE 'local'
$env:JARVIS_SETUP_VERIFY_ONLY = '1'
function Invoke-RestMethod {
    param($Uri, $Headers, $TimeoutSec)
    if ($Uri -ne 'https://api.github.com/repos/cutebody42-web/JARVIS-OS-V.2/releases?per_page=30') { throw 'Unexpected API URL.' }
    $data = Get-Content -LiteralPath (Join-Path $env:JARVIS_BOOTSTRAP_FIXTURE 'releases.json') -Raw | ConvertFrom-Json
    return ,$data
}
function Invoke-WebRequest {
    param([switch]$UseBasicParsing, $Uri, $OutFile, $TimeoutSec)
    if ($Uri.EndsWith('/jarvis-update.json')) { $name = 'manifest.json' }
    elseif ($Uri.EndsWith('/JARVIS-Setup.exe')) { $name = 'installer.bin' }
    else { throw 'Unexpected download URL.' }
    Copy-Item -LiteralPath (Join-Path $env:JARVIS_BOOTSTRAP_FIXTURE $name) -Destination $OutFile
}
function Start-Process { throw 'Contract tests must never launch an installer.' }
'@
        $runner = Join-Path $fixture 'run.ps1'
        [IO.File]::WriteAllText($runner, ($prefix + "`n" + $code), $utf8)
        $env:JARVIS_BOOTSTRAP_FIXTURE = $fixture
        $output = & $shell -NoLogo -NoProfile -File $runner 2>&1
        $actualExit = $LASTEXITCODE
        $expectedExit = if ($case -in @('valid', 'release-order')) { 0 } else { 1 }
        if ($actualExit -ne $expectedExit) { throw ('Case ' + $case + ' returned ' + $actualExit + ', expected ' + $expectedExit + ': ' + ($output -join "`n")) }
        if ($expectedExit -eq 0 -and ($output -join "`n") -notmatch 'Verification completed; installer launch was not requested\.') {
            throw ('Case ' + $case + ' did not complete download verification.')
        }
        if ($expectedExit -ne 0 -and ($output -join "`n") -notmatch 'Setup stopped:') { throw ('Case ' + $case + ' did not fail through the controlled error handler.') }
        $passed += 1
        Write-Host ('PASS ' + $case)
    }
    Write-Host ($passed.ToString() + ' Windows bootstrap contracts passed; command length ' + $code.Length + ' characters.')
} finally {
    $env:JARVIS_BOOTSTRAP_FIXTURE = $previousFixture
    Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
}
