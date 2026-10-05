param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-fA-F]{40}$")]
    [string]$ExpectedCommit
)

$ErrorActionPreference = "Stop"
$previousRequireV07 = [Environment]::GetEnvironmentVariable(
    "REQUIRE_V07_LOCAL_ACCEPTANCE",
    "Process"
)
$previousRequireOllama = [Environment]::GetEnvironmentVariable(
    "REQUIRE_OLLAMA_ACCEPTANCE",
    "Process"
)
$previousPythonUtf8 = [Environment]::GetEnvironmentVariable("PYTHONUTF8", "Process")
$exitCode = 1
$locationPushed = $false

try {
    Push-Location (Split-Path -Parent $PSScriptRoot)
    $locationPushed = $true

    $actualCommit = (git rev-parse --verify HEAD).Trim().ToLowerInvariant()
    if ($LASTEXITCODE -ne 0) { throw "git rev-parse failed" }
    $branch = (git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0) { throw "git branch inspection failed" }
    if (-not $branch) { throw "acceptance requires a named branch, not detached HEAD" }
    $dirty = @(git status --porcelain=v1 --untracked-files=normal)
    if ($LASTEXITCODE -ne 0) { throw "git status failed" }
    if ($dirty.Count -ne 0) {
        throw "acceptance requires a clean working tree; git status reported changes"
    }
    $expected = $ExpectedCommit.ToLowerInvariant()
    if ($actualCommit -ne $expected) {
        throw "expected commit $expected but checked out $actualCommit on branch $branch"
    }
    Write-Host "v0.7 acceptance start: branch=$branch commit=$actualCommit clean=true"

    $env:REQUIRE_V07_LOCAL_ACCEPTANCE = "1"
    $env:REQUIRE_OLLAMA_ACCEPTANCE = "1"
    $env:PYTHONUTF8 = "1"

    uv sync --frozen
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

    uv run --locked pytest -vv `
        tests/test_v07_increment_h.py `
        tests/test_worker_protocol.py::test_forced_process_loss_recovers_same_step_from_postgres `
        tests/test_cli.py `
        tests/test_percept_response_contract.py `
        tests/test_user_prompt_worker_contract.py `
        tests/test_percept_response_failures.py `
        tests/test_artifact_journal.py
    if ($LASTEXITCODE -ne 0) { throw "v2 worker/runtime acceptance failed" }

    # Run the complete deterministic regression suite before the expensive
    # native Ollama gate. This catches policy/contract regressions in seconds
    # instead of allowing them to surface only after multi-minute model runs.
    uv run --locked pytest -q -m "not ollama"
    if ($LASTEXITCODE -ne 0) { throw "deterministic regression suite failed" }

    uv run --locked pytest -vv -s -m ollama
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { throw "Ollama continuity acceptance failed" }
    Write-Host (
        "STRUCTURAL PASS: v0.7 native gate branch=$branch commit=$actualCommit " +
        "clean=true"
    )
    Write-Host (
        "HUMAN REVIEW REQUIRED: judge every printed native Prometheist response " +
        "before accepting this SHA."
    )
}
finally {
    [Environment]::SetEnvironmentVariable(
        "REQUIRE_V07_LOCAL_ACCEPTANCE",
        $previousRequireV07,
        "Process"
    )
    [Environment]::SetEnvironmentVariable(
        "REQUIRE_OLLAMA_ACCEPTANCE",
        $previousRequireOllama,
        "Process"
    )
    [Environment]::SetEnvironmentVariable("PYTHONUTF8", $previousPythonUtf8, "Process")
    if ($locationPushed) {
        Pop-Location
    }
}

exit $exitCode
