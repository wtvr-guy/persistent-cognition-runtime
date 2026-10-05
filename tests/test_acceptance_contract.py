import inspect

import tests.test_acceptance_conversation_continuity as continuity


def test_continuity_acceptance_does_not_export_keyword_slot_parser():
    assert not hasattr(continuity, "_contains_answer_slots")


def test_continuity_acceptance_uses_artifact_receipts_and_human_review():
    source = inspect.getsource(continuity.test_stateless_four_turn_continuity_survives_sessions_and_distractors)

    assert source.count("assert_response_evidence_receipt(") == 4
    assert "HUMAN REVIEW REQUIRED" in source
    assert "assert answer1.strip()" in source
    assert "assert answer2.strip()" in source
    assert "assert answer3.strip()" in source
    assert "assert answer4.strip()" in source
    assert ".strip() ==" not in source
    assert "_contains_answer_slots" not in source
