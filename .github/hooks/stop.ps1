# OMG Stop Hook (PowerShell)
# Runs when the agent is about to finish. Emits a commit reminder when worktree
# changes are present, without creating commits automatically.

$Workspace = ''

if ([Console]::IsInputRedirected) {
    $stdinData = [Console]::In.ReadToEnd()
    if ($stdinData) {
        try {
            $parsed = $stdinData | ConvertFrom-Json -ErrorAction Stop
            if ($parsed.workspace) { $Workspace = $parsed.workspace }
        } catch { }
    }
}

if (-not $Workspace) { $Workspace = if ($env:WORKSPACE) { $env:WORKSPACE } else { (Get-Location).Path } }

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Output '{"decision":"approve"}'
    exit 0
}

& git -C $Workspace rev-parse --is-inside-work-tree *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Output '{"decision":"approve"}'
    exit 0
}

$Status = & git -C $Workspace status --porcelain 2>$null
if ($Status) {
    Write-Output '{"decision":"approve","advisory":"Reminder: the worktree has uncommitted changes. Commit your work before closing the task."}'
} else {
    Write-Output '{"decision":"approve"}'
}