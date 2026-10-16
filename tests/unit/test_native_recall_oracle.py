"""Provenance delivery must not masquerade as correct identifier recall."""
import pytest

from tests._native_artifact_assertions import assert_recalled_literal


@pytest.mark.parametrize("answer", [
    "The codename is BLUE-47A8.", "**BLUE-47A8**", "`BLUE-47A8`",
])
def test_natural_answers_can_retain_an_exact_identifier(answer):
    assert_recalled_literal(answer, "BLUE-47A8", label="recall")


@pytest.mark.parametrize("answer", [
    "I do not have the specific codename.",
    "The nickname is Kestrel-01.",
    "BLUE-47A8-OTHER", "OTHER-BLUE-47A8", "BLUE-47A8X", "XBLUE-47A8",
    "BLUE-47A80", "BLUE-47A8_OTHER",
])
def test_abstention_hallucination_and_partial_matches_are_not_recall(answer):
    with pytest.raises(AssertionError, match="did not reproduce"):
        assert_recalled_literal(answer, "BLUE-47A8", label="recall")
