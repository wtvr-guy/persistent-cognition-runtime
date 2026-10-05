param(
    [string]$DatabaseUrl = $env:PROMETHEIST_BENCHMARK_DATABASE_URL,
    [int[]]$EventCounts = @(1000, 10000, 50000),
    [int]$ProbeEvery = 500,
    [int]$ConfusableEvery = 12,
    [int]$CandidateLimit = 500,
    [int]$AssociationLimit = 250,
    [switch]$SkipTests,
    [switch]$SkipPostgres
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "Prometheist Memory Kernel v0.5 validation"
Write-Host "Repository: $root"
Write-Host "Event counts: $($EventCounts -join ', ')"
Write-Host "Probe cadence: every $ProbeEvery generated events"
Write-Host "Confusable cadence: every $ConfusableEvery non-probe generated events"

if (-not $SkipTests) {
    Write-Host "`n[1/4] Running regression suite..."
    & uv run pytest -v
    if ($LASTEXITCODE -ne 0) {
        throw "Regression suite failed. Scale benchmark stopped before results could be misinterpreted."
    }
} else {
    Write-Host "`n[1/4] Regression suite skipped by request."
}

Write-Host "`n[2/4] Materializing deterministic scale corpora..."
& uv run python -m prometheist.scale_corpus `
    --events $EventCounts `
    --probe-every $ProbeEvery `
    --confusable-every $ConfusableEvery
if ($LASTEXITCODE -ne 0) {
    throw "Scale corpus generation failed."
}

Write-Host "`n[3/4] Running full-history in-memory benchmark..."
& uv run python -m prometheist.scale_benchmark `
    --events $EventCounts `
    --probe-every $ProbeEvery `
    --confusable-every $ConfusableEvery `
    --materialize-dir benchmarks/generated `
    --report benchmarks/generated/v05_in_memory_results.json
if ($LASTEXITCODE -ne 0) {
    throw "In-memory scale benchmark failed."
}

if ($SkipPostgres) {
    Write-Host "`n[4/4] PostgreSQL benchmark skipped by request."
} else {
    if (-not $DatabaseUrl) {
        throw "PostgreSQL benchmark requires -DatabaseUrl or PROMETHEIST_BENCHMARK_DATABASE_URL. Use only a dedicated database whose name contains 'test' or 'benchmark'."
    }

    $env:PROMETHEIST_BENCHMARK_DATABASE_URL = $DatabaseUrl
    Write-Host "`n[4/4] Running indexed PostgreSQL associative benchmark..."
    & uv run python -m prometheist.postgres_scale_benchmark `
        --events $EventCounts `
        --probe-every $ProbeEvery `
        --confusable-every $ConfusableEvery `
        --candidate-limit $CandidateLimit `
        --association-limit $AssociationLimit `
        --report benchmarks/generated/v05_postgres_results.json
    if ($LASTEXITCODE -ne 0) {
        throw "PostgreSQL scale benchmark failed."
    }
}

Write-Host "`nCompleted v0.5 validation sequence."
Write-Host "Generated corpora and reports are under benchmarks/generated/."
Write-Host "Do not treat a scale failure as a generic failure: inspect whether it is candidate recall, scoring/routing, or unknown-abstention behavior."
