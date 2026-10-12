param(
    [switch]$WithOllama,
    [switch]$WithNative,
    [switch]$SkipProduction,
    [ValidateRange(0, 20)][int]$ColdRepeats = 0,
    [ValidateRange(1, 1000)][int]$WarmRepeats = 2,
    [ValidateRange(50, 5000)][int]$SampleMs = 250,
    [string]$ExpectedCommit = ""
)

$ErrorActionPreference = "Stop"
$locationPushed = $false
try {
    Push-Location (Split-Path -Parent $PSScriptRoot)
    $locationPushed = $true
    $argsForBenchmark = @(
        "run",
        "--sample-ms", "$SampleMs",
        "--warm-repeats", "$WarmRepeats"
    )
    if ($WithOllama) {
        $argsForBenchmark += "--with-ollama"
        if (-not $SkipProduction) { $argsForBenchmark += "--with-production" }
    }
    if ($WithNative) { $argsForBenchmark += "--with-native" }
    if ($ColdRepeats -gt 0) {
        $argsForBenchmark += @("--cold-repeats", "$ColdRepeats")
    }
    if ($ExpectedCommit) {
        $argsForBenchmark += @("--expected-commit", $ExpectedCommit)
    }
    $env:PYTHONUTF8 = "1"
    uv run --locked python -m benchmarks.cognitive_effort.runner @argsForBenchmark
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "cognitive-effort benchmark returned exit code $code; inspect the run artifacts"
    }
    Write-Host "Cognitive-effort benchmark complete. Upload evidence.zip from the printed run directory."
}
finally {
    if ($locationPushed) { Pop-Location }
}
