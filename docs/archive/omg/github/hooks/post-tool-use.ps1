# OMG Post-Tool-Use Hook (PowerShell)
# Runs after tool execution in VS Code Copilot Agent Mode or Copilot CLI
#
# Input sources (auto-detected):
#   VS Code:  TOOL_NAME / TOOL_INPUT / TOOL_OUTPUT / WORKSPACE environment variables
#   CLI:      JSON via stdin with toolName / toolInput / toolOutput / workspace fields
#
# Use for: logging, state updates, completion checks

# --- Dual-mode input detection ---
$ToolName = ''
$ToolInput = ''
$ToolOutput = ''
$Workspace = ''

if ([Console]::IsInputRedirected) {
    $stdinData = [Console]::In.ReadToEnd()
    if ($stdinData) {
        try {
            $parsed = $stdinData | ConvertFrom-Json -ErrorAction Stop
            if ($parsed.toolName) { $ToolName = $parsed.toolName }
            if ($parsed.toolInput) { $ToolInput = $parsed.toolInput | ConvertTo-Json -Compress }
            if ($parsed.toolOutput) { $ToolOutput = $parsed.toolOutput | ConvertTo-Json -Compress }
            if ($parsed.workspace) { $Workspace = $parsed.workspace }
        } catch {
            # Not valid JSON, fall through to env vars
        }
    }
}

if (-not $ToolName) { $ToolName = if ($env:TOOL_NAME) { $env:TOOL_NAME } else { '' } }
if (-not $ToolInput) { $ToolInput = if ($env:TOOL_INPUT) { $env:TOOL_INPUT } else { '' } }
if (-not $ToolOutput) { $ToolOutput = if ($env:TOOL_OUTPUT) { $env:TOOL_OUTPUT } else { '' } }
if (-not $Workspace) { $Workspace = if ($env:WORKSPACE) { $env:WORKSPACE } else { (Get-Location).Path } }

# --- Tool name normalization ---
switch ($ToolName) {
    'edit'   { $ToolName = 'editFiles' }
    'read'   { $ToolName = 'readFile' }
    'shell'  { $ToolName = 'runInTerminal' }
    'create' { $ToolName = 'createFile' }
    'delete' { $ToolName = 'deleteFile' }
}

function Test-FileMutationTool {
    param([string]$Name)
    return @('editFiles', 'createFile', 'apply_patch', 'create_file', 'functions.apply_patch', 'functions.create_file') -contains $Name
}

function Get-SafePythonPath {
    param([string]$PathValue)

    if (-not $PathValue) { return $null }
    if ($PathValue -notmatch '\.py$') { return $null }
    if ($PathValue -match '[;''&|`$(){}<>]' -or $PathValue -match '\.\.[\/]') { return $null }

    $AbsolutePath = if ([System.IO.Path]::IsPathRooted($PathValue)) {
        $PathValue
    } else {
        Join-Path $Workspace $PathValue
    }

    $FullPath = [System.IO.Path]::GetFullPath($AbsolutePath)
    $WorkspaceFullPath = [System.IO.Path]::GetFullPath($Workspace)
    if (-not $FullPath.StartsWith($WorkspaceFullPath + [System.IO.Path]::DirectorySeparatorChar)) { return $null }

    $RelativePath = [System.IO.Path]::GetRelativePath($WorkspaceFullPath, $FullPath)
    if ($RelativePath -match '^\.venv[\/]' -or
        $RelativePath -match '^outputs[\/]' -or
        $RelativePath -match '^wandb[\/]' -or
        $RelativePath -match '^artifacts[\/]' -or
        $RelativePath -match '^\.git[\/]' -or
        $RelativePath -match '^\.omg[\/]' -or
        $RelativePath -match '^\.omc[\/]') { return $null }

    if (-not (Test-Path $FullPath -PathType Leaf)) { return $null }
    return $FullPath
}

function Get-PythonFilesFromToolInput {
    param([string]$InputText)

    $Files = New-Object System.Collections.Generic.List[string]
    $NormalizedInput = $InputText -replace '\\n', "`n"

    foreach ($Match in [regex]::Matches($NormalizedInput, '"(?:filePath|path)"\s*:\s*"([^"]+\.py)"')) {
        $SafePath = Get-SafePythonPath $Match.Groups[1].Value
        if ($SafePath) { $Files.Add($SafePath) }
    }

    foreach ($Match in [regex]::Matches($NormalizedInput, '\*\*\* (?:Add|Update) File: ([^\r\n"]+\.py)')) {
        $SafePath = Get-SafePythonPath $Match.Groups[1].Value.Trim()
        if ($SafePath) { $Files.Add($SafePath) }
    }

    return $Files | Sort-Object -Unique
}

function Invoke-RuffForFiles {
    param([string[]]$Files)

    if (-not $Files -or $Files.Count -eq 0) { return }

    $ReportFile = Join-Path $OmgStateDir 'ruff-gate.json'
    $WorkspaceFullPath = [System.IO.Path]::GetFullPath($Workspace)
    $RelativeFiles = @($Files | ForEach-Object { [System.IO.Path]::GetRelativePath($WorkspaceFullPath, $_) })
    $Status = 'ok'
    $OutputText = ''

    Push-Location $Workspace
    try {
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            $OutputText = (& uv run --group dev ruff check --fix -- @RelativeFiles 2>&1 | Out-String)
            $FirstExit = $LASTEXITCODE
            if ($FirstExit -eq 0) { $OutputText += (& uv run --group dev ruff format -- @RelativeFiles 2>&1 | Out-String) }
            $SecondExit = $LASTEXITCODE
            if ($FirstExit -eq 0 -and $SecondExit -eq 0) { $OutputText += (& uv run --group dev ruff check -- @RelativeFiles 2>&1 | Out-String) }
            $ThirdExit = $LASTEXITCODE
            if ($FirstExit -ne 0 -or $SecondExit -ne 0 -or $ThirdExit -ne 0) { $Status = 'failed' }
        } elseif (Get-Command ruff -ErrorAction SilentlyContinue) {
            $OutputText = (& ruff check --fix -- @RelativeFiles 2>&1 | Out-String)
            $FirstExit = $LASTEXITCODE
            if ($FirstExit -eq 0) { $OutputText += (& ruff format -- @RelativeFiles 2>&1 | Out-String) }
            $SecondExit = $LASTEXITCODE
            if ($FirstExit -eq 0 -and $SecondExit -eq 0) { $OutputText += (& ruff check -- @RelativeFiles 2>&1 | Out-String) }
            $ThirdExit = $LASTEXITCODE
            if ($FirstExit -ne 0 -or $SecondExit -ne 0 -or $ThirdExit -ne 0) { $Status = 'failed' }
        } else {
            $Status = 'skipped'
            $OutputText = 'Ruff was not found on PATH, and uv is not installed.'
        }
    } finally {
        Pop-Location
    }

    $Timestamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    $SafeOutput = (($OutputText -split "`r?`n") | Select-Object -First 40) -join ' '
    $Payload = [ordered]@{
        status = $Status
        files = ($RelativeFiles -join ' ')
        timestamp = $Timestamp
        details = $SafeOutput
    }
    $Payload | ConvertTo-Json -Compress | Set-Content -Path $ReportFile
}

$OmgStateDir = Join-Path $Workspace '.omg' 'state'

# Ensure state directory exists
if (-not (Test-Path $OmgStateDir)) {
    New-Item -ItemType Directory -Path $OmgStateDir -Force | Out-Null
}

# --- Context byte accumulation for pre-compaction checkpoint ---
$ContextBytesFile = Join-Path $OmgStateDir 'context-bytes.txt'
$CheckpointTrigger = Join-Path $OmgStateDir 'checkpoint-trigger.json'
$OmgContextThreshold = if ($env:OMG_CONTEXT_THRESHOLD) { [int]$env:OMG_CONTEXT_THRESHOLD } else { 400000 }

# Measure bytes of this tool call's I/O
$InputBytes = [System.Text.Encoding]::UTF8.GetByteCount($ToolInput)
$OutputBytes = [System.Text.Encoding]::UTF8.GetByteCount($ToolOutput)
$CallBytes = $InputBytes + $OutputBytes

# Read current accumulation
$Accumulated = 0
if (Test-Path $ContextBytesFile) {
    $content = (Get-Content $ContextBytesFile -Raw).Trim()
    if ($content -match '^\d+$') { $Accumulated = [long]$content }
}
$Accumulated += $CallBytes
Set-Content -Path $ContextBytesFile -Value $Accumulated -NoNewline

# Check if threshold reached — create checkpoint trigger
if ($Accumulated -ge $OmgContextThreshold -and -not (Test-Path $CheckpointTrigger)) {
    $EstimatedTokens = [math]::Floor($Accumulated / 4)
    $Timestamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    $TriggerJson = @"
{"checkpoint_due": true, "context_bytes": $Accumulated, "estimated_tokens": $EstimatedTokens, "threshold": $OmgContextThreshold, "timestamp": "$Timestamp"}
"@
    Set-Content -Path $CheckpointTrigger -Value $TriggerJson
}

# Log tool usage for debugging (optional, enable by setting OMG_DEBUG=1)
if ($env:OMG_DEBUG -eq '1') {
    $LogFile = Join-Path $OmgStateDir 'tool-usage.log'
    $Timestamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    Add-Content -Path $LogFile -Value "[$Timestamp] $ToolName"
}

# Track file modifications for autopilot phase tracking
if ($ToolName -eq 'editFiles' -or $ToolName -eq 'createFile') {
    $ModifiedFiles = Join-Path $OmgStateDir 'modified-files.txt'
    if ($ToolInput -match '"filePath"\s*:\s*"([^"]*)"') {
        $FilePath = $Matches[1]
        if ($FilePath) {
            Add-Content -Path $ModifiedFiles -Value $FilePath
            # Deduplicate
            if (Test-Path $ModifiedFiles) {
                $unique = Get-Content $ModifiedFiles | Sort-Object -Unique
                Set-Content -Path $ModifiedFiles -Value $unique
            }
        }
    }
}

# Check for test failures after terminal commands
if ($ToolName -eq 'runInTerminal') {
    if ($ToolInput -match '(npm test|jest|vitest|pytest|cargo test|go test)') {
        $Timestamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
        $TestResultFile = Join-Path $OmgStateDir 'last-test-result.json'
        if ($ToolOutput -match '(?i)(FAIL|ERROR|failed|error)') {
            Set-Content -Path $TestResultFile -Value "{`"last_test_run`": `"failed`", `"timestamp`": `"$Timestamp`"}"
        } else {
            Set-Content -Path $TestResultFile -Value "{`"last_test_run`": `"passed`", `"timestamp`": `"$Timestamp`"}"
        }
    }
}

# Run Ruff after Python file edits. This keeps agent-written Python formatted and
# applies safe lint fixes immediately after create/edit style tools complete.
if (Test-FileMutationTool $ToolName) {
    $PythonFiles = @(Get-PythonFilesFromToolInput $ToolInput)
    Invoke-RuffForFiles $PythonFiles
}

# --- Plankton: Opt-in type check + lint after file edits ---
if ($env:OMG_LINT_ON_EDIT -eq '1') {
    if ($ToolName -eq 'editFiles' -or $ToolName -eq 'createFile') {
        $FilePath = ''
        if ($ToolInput -match '"filePath"\s*:\s*"([^"]*)"') {
            $FilePath = $Matches[1]
        }

        $QualityReport = Join-Path $OmgStateDir 'quality-gate.json'
        $QualityStatus = 'ok'
        $QualityDetails = ''

        if ($FilePath) {
            # Sanitize FILE_PATH: reject paths containing shell metacharacters or traversal sequences
            if ($FilePath -match '[;''&|`$(){}<>]' -or $FilePath -match '\.\.[\\/]') {
                $QualityStatus = 'invalid-path'
                $QualityDetails = 'FILE_PATH failed sanitization check'
            } else {
                # TypeScript type check (non-blocking, advisory)
                $TsConfig = Join-Path $Workspace 'tsconfig.json'
                if ((Test-Path $TsConfig) -and $FilePath -match '\.(ts|tsx)$') {
                    try {
                        $TsOutput = & npx tsc --noEmit 2>&1 | Select-Object -First 20 | Out-String
                        if ($TsOutput -match 'error TS') {
                            $QualityStatus = 'type-errors'
                            $QualityDetails = $TsOutput
                        }
                    } catch { }
                }

                # ESLint check (non-blocking, advisory)
                $HasEslint = (Get-ChildItem -Path $Workspace -Filter '.eslintrc*' -ErrorAction SilentlyContinue) -or
                             (Get-ChildItem -Path $Workspace -Filter 'eslint.config.*' -ErrorAction SilentlyContinue)
                if ($HasEslint -and $FilePath -match '\.(ts|tsx|js|jsx)$') {
                    try {
                        $LintOutput = & npx eslint $FilePath --max-warnings=0 2>&1 | Select-Object -First 20 | Out-String
                        if ($LintOutput -match '(error|warning)') {
                            $QualityStatus = "$QualityStatus+lint-warnings"
                            $QualityDetails = "$QualityDetails`n$LintOutput"
                        }
                    } catch { }
                }
            }

            # Write quality gate result (advisory — does NOT block tool execution)
            $Timestamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
            $SafeDetails = $QualityDetails -replace '"', "'" -replace "`n", ' '
            $QualityJson = "{`"status`": `"$QualityStatus`", `"file`": `"$FilePath`", `"timestamp`": `"$Timestamp`", `"details`": `"$SafeDetails`"}"
            Set-Content -Path $QualityReport -Value $QualityJson
        }
    }
}

exit 0
