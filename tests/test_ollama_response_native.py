"""Real-Ollama acceptance for current-only semantic response intent."""
import pytest

from persistent_cognition.percept_response_worker import UserPromptLLM
from persistent_cognition.response_policy import (
    HistoricalEvidenceScope, ResponseAnswerKind, ResponseSurfaceMode,
)
from tests._cli_helpers import ollama_available, print_transcript

pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(not ollama_available(), reason="Ollama is not reachable"),
]


@pytest.mark.parametrize(("prompt", "expected_scope", "expected_kind"), [
    (
        "Tell me the codename for Project Oriole from persistent memory.",
        HistoricalEvidenceScope.USER_AUTHORED,
        ResponseAnswerKind.EXTRACTIVE_VALUES,
    ),
    (
        "List the source event IDs in the last MEMORY_PACKET retrieval record.",
        HistoricalEvidenceScope.DERIVED_INTERNAL,
        ResponseAnswerKind.EXTRACTIVE_VALUES,
    ),
    (
        "Which approach conflicts with my established Kestrel rule, and what constraint "
        "profile did I give that rule? Answer in a short sentence.",
        HistoricalEvidenceScope.USER_AUTHORED,
        ResponseAnswerKind.EXTRACTIVE_VALUES,
    ),
    (
        "What nickname are we using for this plan, and which approach did you just rule "
        "out? Answer naturally and briefly.",
        HistoricalEvidenceScope.MIXED_CONVERSATION,
        ResponseAnswerKind.EXTRACTIVE_VALUES,
    ),
    (
        "The codename for Project Oriole is 2FF0372B.",
        HistoricalEvidenceScope.GENERAL_OR_CURRENT,
        ResponseAnswerKind.SYNTHESIS,
    ),
])
def test_real_ollama_policy_distinguishes_stored_facts_from_internal_records(
    prompt, expected_scope, expected_kind,
):
    policy = UserPromptLLM()._response_policy(prompt)
    print_transcript(f"\nSource-policy regression — User:\n{prompt}")
    print_transcript(f"Source-policy regression — Policy: {policy.model_dump_json()}")
    assert policy.evidence_scope is expected_scope
    assert policy.answer_kind is expected_kind
    assert policy.surface_mode is ResponseSurfaceMode.NATURAL_LANGUAGE
