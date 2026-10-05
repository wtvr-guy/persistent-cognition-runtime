from pathlib import Path


def test_acceptance_script_binds_run_to_clean_named_branch_and_exact_sha():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_v07_acceptance.ps1"
    ).read_text(encoding="utf-8")

    assert "[Parameter(Mandatory = $true)]" in script
    assert "[string]$ExpectedCommit" in script
    assert "git rev-parse --verify HEAD" in script
    assert "git branch --show-current" in script
    assert "git status --porcelain=v1 --untracked-files=normal" in script
    assert "if ($actualCommit -ne $expected)" in script
    assert "STRUCTURAL PASS: v0.7 native gate branch=$branch commit=$actualCommit" in script
    assert "HUMAN REVIEW REQUIRED: judge every printed native Prometheist response" in script
