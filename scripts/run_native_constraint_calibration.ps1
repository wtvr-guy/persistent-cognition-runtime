$ErrorActionPreference = "Stop"
$locationPushed = $false

try {
    Push-Location (Split-Path -Parent $PSScriptRoot)
    $locationPushed = $true
    $env:PYTHONUTF8 = "1"

    if ($env:OS -ne "Windows_NT") {
        throw "Native constraint calibration must run on the intended Windows host."
    }

    uv sync --frozen
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

    $stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
    $output = "benchmarks/results/NATIVE-CONSTRAINTS_$stamp.json"
    uv run --locked python benchmarks/native_constraint_calibration.py --output $output
    if ($LASTEXITCODE -ne 0) { throw "native constraint calibration failed" }

    Write-Host "Native calibration evidence written to $output"
}
finally {
    if ($locationPushed) {
        Pop-Location
    }
}
