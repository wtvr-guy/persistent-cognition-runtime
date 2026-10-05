import pytest
from prometheist.contract_registry import STAGE_CONTRACTS, SEMANTIC_CONTRACTS, contract_manifest
from prometheist.percept_response_runtime import PERCEPT_STAGES, PerceptLLM
from prometheist.situation_runtime import SituationStage
from prometheist.self_reflection import SelfSchemaReview


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


def test_self_review_cannot_request_a_status():
    assert set(SelfSchemaReview.model_fields) == {"opposition_indices", "rationale"}
    with pytest.raises(ValueError):
        SelfSchemaReview(verdict="ESTABLISH", opposition_indices=[], rationale="model assertion")
    with pytest.raises(ValueError):
        SelfSchemaReview(opposition_indices=[1, 1], rationale="duplicate")


@pytest.mark.parametrize("kind", list(SEMANTIC_CONTRACTS))
def test_registered_prompts_and_schemas_are_resolvable(kind):
    contract = contract_manifest()["semantic_contracts"][kind]
    assert len(contract["schema_sha256"]) == 64
    assert len(contract["prompt_sha256"]) == 64
    assert SEMANTIC_CONTRACTS[kind].output_schema()["type"] == "object"
