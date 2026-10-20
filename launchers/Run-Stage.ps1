param(
    [Parameter(Mandatory = $true)][string]$Config,
    [switch]$Preflight,
    [switch]$SelfTest,
    [switch]$PrintEffectiveConfig
)
$ErrorActionPreference = 'Stop'
$Python = Join-Path $PSScriptRoot '..\runtime\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Install the isolated DeepSeekRPC runtime bundle first.' }
$StageArgs = @('-m', 'deepseekrpc.cli', 'serve', '--config', $Config)
if ($Preflight) { $StageArgs += '--preflight' }
if ($SelfTest) { $StageArgs += '--self-test' }
if ($PrintEffectiveConfig) { $StageArgs += '--print-effective-config' }
& $Python @StageArgs
exit $LASTEXITCODE
