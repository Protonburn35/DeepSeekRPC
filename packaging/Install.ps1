param(
    [string]$BundleRoot = $PSScriptRoot,
    [switch]$BootstrapOnly
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitProcess) {
    throw 'DeepSeekRPC installation requires native 64-bit Windows.'
}
$BundleRoot = [IO.Path]::GetFullPath($BundleRoot).TrimEnd('\', '/')
$RootPrefix = $BundleRoot + [IO.Path]::DirectorySeparatorChar

function Owned-Path([string]$Relative) {
    if ([IO.Path]::IsPathRooted($Relative) -or $Relative -notmatch '^[A-Za-z0-9_./+\-]+$' -or
        @($Relative.Split('/') | Where-Object { $_ -in @('', '.', '..') }).Count -ne 0) {
        throw 'The bundle contains an invalid relative file path.'
    }
    $Path = [IO.Path]::GetFullPath((Join-Path $BundleRoot $Relative))
    if (-not $Path.StartsWith($RootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The bundle path escapes its install directory.'
    }
    $Cursor = $BundleRoot
    foreach ($Part in $Relative.Split('/')) {
        $Cursor = Join-Path $Cursor $Part
        if ((Test-Path -LiteralPath $Cursor) -and
            ((Get-Item -LiteralPath $Cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw 'The install tree contains a redirected file or directory.'
        }
    }
    return $Path
}

function Read-Json([string]$Path, [long]$Limit = 1048576) {
    if ((Get-Item -LiteralPath $Path).Length -gt $Limit) { throw 'Installer metadata exceeds its bound.' }
    return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
}

function Verify-File([string]$Path, $Record) {
    if ($Record.sha256 -notmatch '^[a-f0-9]{64}$' -or $Record.bytes -lt 1 -or
        (Get-Item -LiteralPath $Path).Length -ne $Record.bytes -or
        (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Record.sha256) {
        throw 'A bundle file failed its pinned size/checksum validation.'
    }
}

function Fetch-Pinned($Record, [string]$Destination) {
    if ($Record.url -notmatch '^https://github.com/astral-sh/' -or $Record.sha256 -notmatch '^[a-f0-9]{64}$' -or
        $Record.bytes -lt 1 -or $Record.bytes -gt 67108864) { throw 'Invalid bounded bootstrap download.' }
    if (Test-Path -LiteralPath $Destination) {
        Verify-File $Destination $Record
        return
    }
    $Temporary = $Destination + '.partial'
    if ((Test-Path -LiteralPath $Temporary) -and
        ((Get-Item -LiteralPath $Temporary).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'Bootstrap temporary file is redirected.'
    }
    Add-Type -AssemblyName System.Net.Http
    $Client = New-Object Net.Http.HttpClient
    $Cancellation = [Threading.CancellationTokenSource]::new(300000)
    $Client.Timeout = [TimeSpan]::FromSeconds(300)
    $Client.DefaultRequestHeaders.UserAgent.ParseAdd('DeepSeekRPC-installer/1')
    try {
        $Response = $Client.GetAsync($Record.url, [Net.Http.HttpCompletionOption]::ResponseHeadersRead, $Cancellation.Token).GetAwaiter().GetResult()
        try {
            $Response.EnsureSuccessStatusCode() | Out-Null
            $InputStream = $Response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
            $OutputStream = [IO.File]::Open($Temporary, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
            try {
                $Buffer = New-Object byte[] 1048576
                [long]$Total = 0
                while (($Count = $InputStream.ReadAsync($Buffer, 0, $Buffer.Length, $Cancellation.Token).GetAwaiter().GetResult()) -gt 0) {
                    $Total += $Count
                    if ($Total -gt $Record.bytes) { throw 'Bootstrap download exceeded its pinned length.' }
                    $OutputStream.Write($Buffer, 0, $Count)
                }
                $OutputStream.Flush($true)
            } finally {
                $OutputStream.Dispose()
                $InputStream.Dispose()
            }
        } finally { $Response.Dispose() }
        Verify-File $Temporary $Record
        [IO.File]::Move($Temporary, $Destination)
    } finally {
        $Client.Dispose()
        $Cancellation.Dispose()
        if (Test-Path -LiteralPath $Temporary) { Remove-Item -LiteralPath $Temporary }
    }
}

function Run-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'An isolated runtime installation/check command failed.' }
}

$ManifestPath = Owned-Path 'BUNDLE.json'
$Manifest = Read-Json $ManifestPath
if ($Manifest.format -ne 1 -or $Manifest.kind -notin @('runtime', 'bootstrap-test') -or
    $Manifest.source_commit -notmatch '^[a-f0-9]{40}$') { throw 'Invalid DeepSeekRPC bundle manifest.' }
if ($Manifest.kind -eq 'bootstrap-test' -and -not $BootstrapOnly) {
    throw 'This is a bootstrap test fixture, not an application runtime bundle.'
}
foreach ($Property in $Manifest.files.PSObject.Properties) {
    Verify-File (Owned-Path $Property.Name) $Property.Value
}
$Allowed = @('BUNDLE.json', '.bootstrap', '.python', 'runtime', '.installation-check', '.install-complete.json',
             '.install-owner', 'installer.lock', 'credentials', 'runtime-A', 'runtime-B', 'runtime-C')
foreach ($Name in $Manifest.files.PSObject.Properties.Name) { $Allowed += $Name.Split('/')[0] }
foreach ($Entry in Get-ChildItem -LiteralPath $BundleRoot -Force) {
    if ($Entry.Name -notin $Allowed -or ($Entry.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'Install into a dedicated extracted bundle directory; unrelated or redirected files are present.'
    }
}
$OwnerPath = Owned-Path '.install-owner'
$OwnerText = 'DeepSeekRPC isolated runtime installation v1'
if (Test-Path -LiteralPath $OwnerPath) {
    if ((Get-Content -LiteralPath $OwnerPath -Raw -Encoding UTF8).Trim() -ne $OwnerText) { throw 'Install ownership marker differs.' }
} else {
    foreach ($Name in @('.bootstrap', '.python', 'runtime', '.installation-check', '.install-complete.json',
                        'installer.lock', 'runtime-A', 'runtime-B', 'runtime-C')) {
        if (Test-Path -LiteralPath (Join-Path $BundleRoot $Name)) {
            throw 'An existing runtime/state directory has no matching DeepSeekRPC install owner. Choose a new extracted bundle directory.'
        }
    }
    [IO.File]::WriteAllText($OwnerPath, $OwnerText, [Text.UTF8Encoding]::new($false))
}
$CompletePath = Owned-Path '.install-complete.json'
if (Test-Path -LiteralPath $CompletePath) {
    $Existing = Read-Json $CompletePath
    if ($Existing.source_commit -ne $Manifest.source_commit) {
        throw 'Unpack the new version in a separate directory; this runtime already belongs to another source version.'
    }
}
$InstallationLock = [IO.File]::Open((Owned-Path 'installer.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
$Saved = @{}
foreach ($Name in @('UV_PYTHON_INSTALL_DIR', 'UV_CACHE_DIR', 'UV_NO_CONFIG', 'UV_NO_VERIFY_HASHES',
                   'UV_PYTHON_DOWNLOADS', 'PYTHONUTF8', 'PIP_DISABLE_PIP_VERSION_CHECK')) {
    $Saved[$Name] = [Environment]::GetEnvironmentVariable($Name, 'Process')
}
try {
    $Bootstrap = Read-Json (Owned-Path 'provenance/windows-bootstrap.lock.json')
    $PythonRoot = Owned-Path '.python'
    $BootstrapRoot = Owned-Path '.bootstrap'
    [IO.Directory]::CreateDirectory($BootstrapRoot) | Out-Null
    $env:UV_PYTHON_INSTALL_DIR = $PythonRoot
    $env:UV_CACHE_DIR = Owned-Path '.bootstrap/cache'
    $env:UV_NO_CONFIG = '1'
    $env:UV_NO_VERIFY_HASHES = $null
    $env:UV_PYTHON_DOWNLOADS = 'automatic'
    $env:PYTHONUTF8 = '1'
    $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
    $ZipPath = Owned-Path '.bootstrap/uv.zip'
    Fetch-Pinned $Bootstrap.uv $ZipPath
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Zip = [IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        $Candidates = @($Zip.Entries | Where-Object { $_.FullName -eq 'uv.exe' -or $_.FullName.EndsWith('/uv.exe') })
        if ($Candidates.Count -ne 1 -or $Candidates[0].Length -gt 209715200) { throw 'Pinned uv archive has an unexpected executable inventory.' }
        $Uv = Owned-Path '.bootstrap/uv.exe'
        [IO.Compression.ZipFileExtensions]::ExtractToFile($Candidates[0], $Uv, $true)
    } finally { $Zip.Dispose() }
    $DownloadMetadata = [Uri]::new((Owned-Path 'provenance/python-windows-download.json')).AbsoluteUri
    Run-Checked $Uv @('python', 'install', $Bootstrap.python.key, '--install-dir', $PythonRoot,
                      '--no-bin', '--no-registry', '--python-downloads-json-url', $DownloadMetadata,
                      '--no-config', '--no-progress')
    $BasePython = Owned-Path ('.python/' + $Bootstrap.python.key + '/python.exe')
    Run-Checked $BasePython @('-c', 'import sys; assert sys.version_info[:3] == (3,12,14)')
    $RuntimeRoot = Owned-Path 'runtime'
    $Python = Owned-Path 'runtime/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        Run-Checked $Uv @('venv', '--no-project', '--no-python-downloads', '--python', $BasePython, $RuntimeRoot)
    }
    Run-Checked $Python @('-c', 'import sys; assert sys.version_info[:3] == (3,12,14); assert sys.prefix != sys.base_prefix')
    if ($BootstrapOnly) {
        Write-Output 'PASS: isolated pinned Windows Python bootstrap; application dependencies/native/GPU checks were not run.'
        return
    }
    Push-Location $BundleRoot
    try {
        if (-not (Test-Path -LiteralPath $CompletePath)) {
            Run-Checked $Uv @('pip', 'install', '--python', $Python, '--require-hashes', '--only-binary', ':all:',
                              '--torch-backend', 'cu130', '--default-index', 'https://pypi.org/simple',
                              '-r', (Owned-Path 'requirements-runtime-windows.lock'))
            Run-Checked $Uv @('pip', 'install', '--python', $Python, '--no-index', '--no-deps', '--require-hashes',
                              '-r', (Owned-Path 'wheels.lock'))
        }
        Run-Checked $Python @('-m', 'pip', 'check')
        Run-Checked $Python @((Owned-Path 'tools/qualify_runtime.py'), '--bundle', $BundleRoot,
                              '--output', (Owned-Path '.installation-check'))
    } finally { Pop-Location }
    $Complete = @{ format = 1; source_commit = $Manifest.source_commit; python = '3.12.14';
                   scope = 'Native imports, Python dependencies and SDK-free device-independent startup; GPU execution not run' }
    [IO.File]::WriteAllText((Owned-Path '.install-complete.json'), ($Complete | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
    Write-Output 'DeepSeekRPC isolated runtime installed. Configure the role files and run local preflight before starting a worker.'
} finally {
    foreach ($Name in $Saved.Keys) { [Environment]::SetEnvironmentVariable($Name, $Saved[$Name], 'Process') }
    $InstallationLock.Dispose()
}
