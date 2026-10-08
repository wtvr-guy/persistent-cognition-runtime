import pytest
from persistent_cognition.contract_registry import STAGE_CONTRACTS, SEMANTIC_CONTRACTS, contract_manifest
from persistent_cognition.percept_response_runtime import PERCEPT_STAGES, PerceptLLM
from persistent_cognition.situation_runtime import SituationStage


def test_every_stage_has_one_registered_contract_and_capability():
    stages = (*PERCEPT_STAGES, *tuple(SituationStage))
    assert set(STAGE_CONTRACTS) == {s.value for s in stages}
    assert {s.capability for s in stages} == {c.capability for c in STAGE_CONTRACTS.values()}
    assert len({s.capability for s in stages}) == len(stages)
    assert contract_manifest() == contract_manifest()
    assert set(SEMANTIC_CONTRACTS) == set().union(*(c.kinds for c in STAGE_CONTRACTS.values()))


def test_retired_control_contracts_are_absent():
    assert not any("SUFFICIENCY" in k or "MEMORY_REQUIREMENTS" in k for k in SEMANTIC_CONTRACTS)
    assert not any("COMPOSE" in k or "MEMORY_REQUIREMENTS" in k for k in STAGE_CONTRACTS)
    assert not hasattr(PerceptLLM, "assess_memory_sufficiency")
    assert not hasattr(PerceptLLM, "_response_policy")
    assert not hasattr(PerceptLLM, "generate_final_response")


def test_self_model_contracts_are_absent():
    assert not any("SELF" in k for k in SEMANTIC_CONTRACTS)
    assert not any("SELF" in k for k in STAGE_CONTRACTS)


@pytest.mark.parametrize("kind", list(SEMANTIC_CONTRACTS))
def test_registered_prompts_and_schemas_are_resolvable(kind):
    contract = contract_manifest()["semantic_contracts"][kind]
    assert len(contract["schema_sha256"]) == 64
    assert len(contract["prompt_sha256"]) == 64
    assert SEMANTIC_CONTRACTS[kind].output_schema()["type"] == "object"


def test_source_verifier_checks_extraction_paths_and_retirements(tmp_path):
    import hashlib
    import runpy
    from pathlib import Path
    verify = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/verify_source_baseline.py"))["verify_manifest"]
    path = tmp_path / "src/persistent_cognition/example.py"
    path.parent.mkdir(parents=True)
    path.write_text("adapted code")
    source_hash = hashlib.sha256(b"original code").hexdigest()
    manifest = {"files": {
        "src/prometheist/example.py": {
            "source_path": "src/prometheist/example.py", "source_sha256": source_hash,
            "extraction_path": "src/persistent_cognition/example.py",
            "extracted_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "src/prometheist/self_model.py": {
            "source_path": "src/prometheist/self_model.py", "source_sha256": source_hash,
            "extraction_path": "src/persistent_cognition/self_model.py",
            "status": "retired", "extracted_sha256": None, "reason": "Removed operational identity",
        },
    }}
    assert verify(manifest, tmp_path) == {"unchanged": 0, "adapted": 1, "added": 0, "retired": 1}
    path.write_text("unexpected drift")
    with pytest.raises(SystemExit, match="modified extraction"):
        verify(manifest, tmp_path)
    path.write_text("adapted code")
    path.with_name("self_model.py").write_text("retired code restored")
    with pytest.raises(SystemExit, match="invalid retirement"):
        verify(manifest, tmp_path)


def test_source_verifier_detects_new_unrecorded_code_and_missing_addition(tmp_path):
    import runpy
    from pathlib import Path
    verify = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/verify_source_baseline.py"))["verify_manifest"]
    path = tmp_path / "src/persistent_cognition/new.py"
    path.parent.mkdir(parents=True)
    path.write_text("new code")
    with pytest.raises(SystemExit, match="unrecorded code"):
        verify({"files": {}}, tmp_path)
    manifest = {"files": {"missing.py": {
        "status": "added", "source_path": None, "source_sha256": None,
        "extraction_path": "missing.py", "extracted_sha256": None, "reason": "New runtime boundary",
    }}}
    with pytest.raises(SystemExit, match="missing current file"):
        verify(manifest, tmp_path)
