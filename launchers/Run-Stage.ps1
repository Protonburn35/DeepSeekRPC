param(
    [Parameter(Mandatory = $true)][string]$Config,
    [switch]$Preflight,
    [switch]$SelfTest,
    [switch]$PrintEffectiveConfig
)
$ErrorActionPreference = 'Stop'
$Bundle = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Complete = Join-Path $Bundle '.install-complete.json'
if (-not (Test-Path -LiteralPath $Complete)) { throw 'Run Install.ps1 successfully before launching this bundle.' }
$Installed = Get-Content -LiteralPath $Complete -Raw -Encoding UTF8 | ConvertFrom-Json
$Manifest = Get-Content -LiteralPath (Join-Path $Bundle 'BUNDLE.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($Installed.format -ne 1 -or $Installed.source_commit -ne $Manifest.source_commit -or $Installed.python -ne '3.12.14') {
    throw 'The completed installation does not match this application bundle.'
}
$Python = Join-Path $PSScriptRoot '..\runtime\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Install the isolated DeepSeekRPC runtime bundle first.' }
$StageArgs = @('-m', 'deepseekrpc.cli', 'serve', '--config', $Config)
if ($Preflight) { $StageArgs += '--preflight' }
if ($SelfTest) { $StageArgs += '--self-test' }
if ($PrintEffectiveConfig) { $StageArgs += '--print-effective-config' }
& $Python -I @StageArgs
exit $LASTEXITCODE
