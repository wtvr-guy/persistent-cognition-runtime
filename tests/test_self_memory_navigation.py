"""Replay the failed native run's frozen learning through production retrieval."""
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from psycopg.rows import dict_row

from prometheist import db, event_store
from prometheist.cognitive_store import put_record
from prometheist.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket
from prometheist.percept_response_runtime import retrieve_memory_package
from tests._retrieval_fixtures import load_person_fidelity_corpus
from prometheist.response_policy import HistoricalEvidenceScope, ResponseSurfaceMode
from prometheist.self_memory import activate_self_context
from prometheist.self_memory_navigation import expand_self_memory_roots

ROOT = Path(__file__).resolve().parents[1]
from tests._retrieval_fixtures import snapshot_digest  # noqa: E402
from tests._retrieval_fixtures import _restore_self_memory, _seed_life_record  # noqa: E402

SOURCE_TYPES = [EventType.USER_PROMPT, EventType.PERCEPT_OBSERVATION,
                EventType.SYSTEM_EVENT, EventType.TOOL_RESULT]


@pytest.fixture
def learned():
    corpus = load_person_fidelity_corpus(ROOT / "tests/fixtures/retrieval_corpus.json")
    frozen = json.loads((ROOT / "tests/fixtures/self_memory_navigation_snapshot.json").read_text())
    assert frozen["fixture_sha256"] == corpus.fixture_sha256
    assert snapshot_digest(frozen["snapshot"]) == frozen["snapshot_sha256"]
    with db.get_connection() as conn:
        event_ids = _seed_life_record(conn, corpus, form_situations=False)
        _restore_self_memory(conn, frozen["snapshot"])
        yield conn, corpus, event_ids, frozen["snapshot"]


def packet_for(conn, event_ids=()):
    items = []
    for event_id in event_ids:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM events WHERE event_id=%s", (event_id,))
            event = SimpleNamespace(**cur.fetchone())
        items.append(MemoryEvidence(
            source_event_id=event.event_id, event_type=event.event_type, source=event.source,
            created_at=event.created_at, conversation_id=event.conversation_id,
            conversation_seq=event.conversation_seq, global_seq=event.global_seq,
            content=event.payload_text,
        ))
    return MemoryPacket(memory_request_id=uuid4(), need=MemoryNeed(), supported=bool(items), items=items)


def cutoff(conn):
    return conn.execute("SELECT max(global_seq) + 1 FROM events").fetchone()[0]


def navigate(conn, query, *, packet=None, limit=20, types=SOURCE_TYPES, before=None):
    return expand_self_memory_roots(
        conn, packet if packet is not None else packet_for(conn), query_text=query,
        source_types=types, before_global_seq=before if before is not None else cutoff(conn),
        item_limit=limit, memory_request_id=uuid4(),
    )


@pytest.mark.parametrize("probe_id,initial_seed_ids", [
    ("pf-q005", ["pf-e004"]), ("pf-q006", []),
])
def test_frozen_failure_required_roots_reach_fixed_retrieval(
    learned, probe_id, initial_seed_ids,
):
    conn, corpus, event_ids, snapshot = learned
    probe = next(p for p in corpus.probes if p.probe_id == probe_id)
    required = {event_ids[key] for key in probe.required_event_ids}
    initial = packet_for(conn, [event_ids[key] for key in initial_seed_ids])
    assert not required.issubset({item.source_event_id for item in initial.items})

    conversation = event_store.start_conversation(conn)
    package = retrieve_memory_package(
        conn, SimpleNamespace(user_text=probe.prompt, before_global_seq=cutoff(conn),
                              interaction_id=uuid4(), conversation_id=conversation,
                              correlation_id=uuid4(), task_id=uuid4()), initial, SOURCE_TYPES,
        person_history_required=True,
    )
    assert required.issubset({item.source_event_id for item in package.memory_packet.items})
    assert package.retrieval_policy == "fixed-retrieval/v1"
    # Navigation must not promote any candidate or rewrite a learned assertion.
    for key, payload in snapshot["self_resolution"]:
        assert conn.execute("SELECT payload FROM cognitive_heads WHERE record_kind='self_resolution' "
                            "AND record_key=%s", (key,)).fetchone()[0] == payload


def test_frozen_message_prompt_recovers_style_and_observed_example(learned):
    conn, corpus, event_ids, _ = learned
    probe = next(p for p in corpus.probes if p.probe_id == "pf-q007")
    required = {event_ids[key] for key in probe.required_event_ids}
    conversation = event_store.start_conversation(conn)
    context = activate_self_context(conn, query_text=probe.prompt,
                                    evidence_scope=HistoricalEvidenceScope.SELF_MODEL,
                                    surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE)
    package = retrieve_memory_package(
        conn, SimpleNamespace(user_text=probe.prompt, before_global_seq=cutoff(conn),
            interaction_id=uuid4(), conversation_id=conversation, correlation_id=uuid4(), task_id=uuid4()),
        packet_for(conn), SOURCE_TYPES, context, person_history_required=True,
    )
    assert required.issubset({item.source_event_id for item in package.memory_packet.items})
    assert package.adaptive_recall_rounds == 4
    assert "self-memory-canonical-navigation-v1" in json.dumps(package.memory_packet.retrieval_trace)


def test_candidate_navigation_preserves_canonical_text_and_source_policy(learned):
    conn, corpus, event_ids, _ = learned
    result = navigate(conn, "surprises")
    assert [item.source_event_id for item in result.items] == [event_ids["pf-e011"]]
    assert result.items[0].content == next(e.text for e in corpus.life_events if e.event_id == "pf-e011")
    assert result.items[0].score is None
    assert result.items[0].retrieval_reasons == ["SELF_MEMORY_ROOT", "SUPPORTS"]
    assert result.retrieval_trace["root_routes"][0]["status"] == "CANDIDATE"
    assert not navigate(conn, "surprises", types=[EventType.SYSTEM_EVENT]).items
    assert not navigate(conn, "zzqvtx").items
    assert MemoryPacket.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("status", ["REJECTED", "SUPERSEDED"])
def test_rejected_and_superseded_hypotheses_do_not_route(learned, status):
    conn, _, _, snapshot = learned
    result = navigate(conn, "surprises")
    rep_id = result.retrieval_trace["candidate_representation_ids"][0]
    old = dict(snapshot["self_resolution"])[rep_id]
    put_record(conn, "self_resolution", rep_id, {**old, "status": status}, revision=status)
    assert not navigate(conn, "surprises").items


def test_navigation_cutoff_applies_to_learned_links_as_well_as_source_history(learned):
    conn, _, _, _ = learned
    result = navigate(conn, "surprises")
    link_id = result.retrieval_trace["root_routes"][0]["link_event_id"]
    link_sequence = conn.execute("SELECT global_seq FROM events WHERE event_id=%s", (link_id,)).fetchone()[0]
    assert not navigate(conn, "surprises", before=link_sequence).items


def test_navigation_preserves_original_seeds_and_obeys_saturation(learned):
    conn, _, event_ids, _ = learned
    initial = packet_for(conn, [event_ids["pf-e004"]])
    result = navigate(conn, "surprises", packet=initial, limit=2)
    assert result.items[0] == initial.items[0]
    assert len(result.items) == 2
    assert len(initial.items) == 1
    assert navigate(conn, "surprises", packet=result, limit=2) == result
    assert navigate(conn, "surprises", packet=result, limit=3).items == result.items
    with pytest.raises(ValueError, match="bounded packet"):
        navigate(conn, "surprises", packet=result, limit=1)


def test_compound_cue_can_find_a_learned_topic_tag(learned):
    conn, _, event_ids, _ = learned
    packet = navigate(conn, "grid-resilience")
    assert event_ids["pf-e002"] in {item.source_event_id for item in packet.items}, packet.retrieval_trace


def test_cue_in_linked_source_can_find_a_paraphrased_self_representation(learned):
    conn, _, event_ids, snapshot = learned
    # Learned statements/tags omit this canonical wording. Source cues must
    # still locate the representation's original evidence without a model query.
    assert all("messages" not in row[1]["statement"] for row in snapshot["self_representation"])
    packet = navigate(conn, "messages")
    assert event_ids["pf-e014"] in {item.source_event_id for item in packet.items}
    assert not navigate(conn, "messages", types=[EventType.SYSTEM_EVENT]).items


def test_counterevidence_is_returned_only_after_its_link_is_known(learned):
    conn, _, event_ids, snapshot = learned
    old = next(data for _, data in snapshot["self_evidence"]
               if data["root_event_id"] == str(event_ids["pf-e011"]))
    conversation = event_store.start_conversation(conn)
    correction = event_store.record_event(
        conn, conversation_id=conversation, correlation_id=uuid4(),
        event_type=EventType.USER_PROMPT, source="user",
        payload={"text": "I enjoyed being surprised at yesterday's party."},
        payload_text="I enjoyed being surprised at yesterday's party.",
    )
    link = {**old, "evidence_id": str(uuid4()), "root_event_id": str(correction.event_id),
            "relation": "OPPOSES", "observed_at": correction.created_at.isoformat(),
            "known_at": correction.created_at.isoformat()}
    link_event = put_record(conn, "self_evidence", f"{old['representation_id']}:counterexample",
                            link, revision="1")
    sequence = conn.execute("SELECT global_seq FROM events WHERE event_id=%s", (link_event,)).fetchone()[0]
    before = navigate(conn, "surprises", before=sequence)
    after = navigate(conn, "surprises")
    assert {item.source_event_id for item in before.items} == {event_ids["pf-e011"]}
    assert {item.source_event_id for item in after.items} == {event_ids["pf-e011"], correction.event_id}
    assert {route["relation"] for route in after.retrieval_trace["root_routes"]} == {"SUPPORTS", "OPPOSES"}


def test_disallowed_roots_cannot_consume_the_representation_candidate_budget(learned):
    conn, _, event_ids, snapshot = learned
    representation = snapshot["self_representation"][0][1]
    resolution = snapshot["self_resolution"][0][1]
    evidence = snapshot["self_evidence"][0][1]
    # More high-scoring disallowed representations than the candidate bound.
    from prometheist.self_memory_navigation import SELF_ROOT_CANDIDATE_LIMIT
    for index in range(SELF_ROOT_CANDIDATE_LIMIT + 1):
        rep_id = str(uuid4())
        allowed = index == SELF_ROOT_CANDIDATE_LIMIT
        put_record(conn, "self_representation", rep_id, {
            **representation, "representation_id": rep_id,
            "statement": "zirconium" if allowed else "zirconium zirconium zirconium",
            "context_tags": [],
        }, revision="1")
        put_record(conn, "self_resolution", rep_id, {
            **resolution, "representation_id": rep_id, "status": "CANDIDATE",
        }, revision="1")
        put_record(conn, "self_evidence", f"{rep_id}:root", {
            **evidence, "representation_id": rep_id, "evidence_id": str(uuid4()),
            "root_event_id": str(event_ids["pf-e011" if allowed else "pf-e015"]),
        }, revision="1")
    result = navigate(conn, "zirconium", types=[EventType.USER_PROMPT])
    assert {item.source_event_id for item in result.items} == {event_ids["pf-e011"]}
    assert len(result.retrieval_trace["candidate_representation_ids"]) == 1
