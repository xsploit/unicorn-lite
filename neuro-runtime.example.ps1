param(
    [ValidateSet('Start', 'Stop', 'Status')]
    [string]$Action = 'Status',
    [Parameter(Mandatory = $false)]
    [string]$ConversationChannelId = $env:UNICORN_DISCORD_CHANNEL_ID
)

$runtimeRoot = $PSScriptRoot
$pythonExe = (Get-Command py).Source
$stdout = Join-Path $runtimeRoot 'neuro-live.out.log'
$stderr = Join-Path $runtimeRoot 'neuro-live.err.log'

function Get-UnicornRuntime {
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(?:\.exe)?$' -and
        $_.CommandLine -like '*-m unicorn_lite discord*'
    }
}

if ($Action -eq 'Status') {
    $running = @(Get-UnicornRuntime)
    [pscustomobject]@{ Running = $running.Count -gt 0; ProcessIds = @($running.ProcessId) }
    if (Test-Path -LiteralPath $stdout) { Get-Content -LiteralPath $stdout -Tail 12 }
    exit 0
}

if ($Action -eq 'Stop') {
    $running = @(Get-UnicornRuntime)
    foreach ($process in $running) { Stop-Process -Id $process.ProcessId -Force }
    [pscustomobject]@{ Stopped = @($running.ProcessId) }
    exit 0
}

if (-not $ConversationChannelId) {
    throw 'Set UNICORN_DISCORD_CHANNEL_ID or pass -ConversationChannelId.'
}

$existing = @(Get-UnicornRuntime)
if ($existing.Count -gt 0) {
    [pscustomobject]@{ Started = $false; Reason = 'already running'; ProcessIds = @($existing.ProcessId) }
    exit 0
}

$env:PYTHONUNBUFFERED = '1'
$arguments = @(
    '-3.10', '-m', 'unicorn_lite', 'discord',
    '--mode', 'active',
    '--writer', 'vercel',
    '--encoder', 'local',
    '--gate-encoder', 'mpc-bert',
    '--device', 'cuda',
    '--db', '.\data\neuro-live.db',
    '--gate-checkpoint', '.\data\response-gate.pt',
    '--gate-threshold-override', "$ConversationChannelId=0.28",
    '--allow-unsolicited',
    '--unsolicited-channel-id', $ConversationChannelId,
    '--decision-audit-channel-id', $ConversationChannelId,
    '--decision-audit-source-channel-id', $ConversationChannelId
)

$process = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList $arguments `
    -WorkingDirectory $runtimeRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

[pscustomobject]@{ Started = $true; ProcessId = $process.Id; Stdout = $stdout; Stderr = $stderr }
