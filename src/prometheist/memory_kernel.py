"""Deterministic, LLM-independent core of the Prometheist memory kernel.

The kernel treats persisted events as evidence. It does not summarize or
rewrite them. Recall is a pure function of the evidence set, cue state, and an
explicit scoring policy so policy constants can be benchmarked without editing
production code between candidates.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import groupby
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

POLICY_VERSION = "deterministic-cues-v2"

# Named policy values are intentionally visible to the constraint auditor. They
# remain the current baseline until MEM-SCORE-001 selects evidence-backed values.
LEXICAL_WEIGHT = 0.78
ENTITY_WEIGHT = 0.14
TEMPORAL_WEIGHT = 0.05
CONVERSATION_WEIGHT = 0.03
PHRASE_BONUS = 0.15
TEMPORAL_SCALE_DAYS = 30.0

_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
        "did", "do", "does", "for", "from", "had", "has", "have", "he",
        "her", "hers", "him", "his", "i", "in", "is", "it", "its", "me",
        "my", "of", "on", "or", "our", "she", "so", "that", "the", "their",
        "them", "they", "this", "to", "us", "was", "we", "were", "what",
        "when", "where", "which", "who", "why", "with", "you", "your",
    }
)


@dataclass(frozen=True, slots=True)
class MemoryScoringPolicy:
    """Pure scoring knobs exposed for empirical calibration."""

    lexical_weight: float = LEXICAL_WEIGHT
    entity_weight: float = ENTITY_WEIGHT
    temporal_weight: float = TEMPORAL_WEIGHT
    conversation_weight: float = CONVERSATION_WEIGHT
    phrase_bonus: float = PHRASE_BONUS
    temporal_scale_days: float = TEMPORAL_SCALE_DAYS

    def __post_init__(self) -> None:
        weights = (
            self.lexical_weight,
            self.entity_weight,
            self.temporal_weight,
            self.conversation_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("memory scoring weights must be non-negative")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("memory scoring weights must sum to 1")
        if not 0.0 <= self.phrase_bonus <= 1.0:
            raise ValueError("phrase_bonus must be between 0 and 1")
        if self.temporal_scale_days <= 0.0:
            raise ValueError("temporal_scale_days must be positive")


DEFAULT_MEMORY_SCORING_POLICY = MemoryScoringPolicy()


def normalize_text(text: str) -> str:
    """Return stable case-folded text for deterministic matching."""
    return unicodedata.normalize("NFKC", text).casefold().strip()


def _light_stem(token: str) -> str:
    """Conservative deterministic morphology normalization."""
    if token.endswith("'s") and len(token) > 3:
        token = token[:-2]
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        base = token[:-3]
        if len(base) >= 3 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, *, drop_stop_words: bool = True) -> tuple[str, ...]:
    """Tokenize without an NLP model or external dependency."""
    tokens = tuple(
        _light_stem(normalize_text(match.group(0)))
        for match in _WORD_RE.finditer(text)
    )
    if not drop_stop_words:
        return tokens
    return tuple(token for token in tokens if len(token) > 1 and token not in _STOP_WORDS)


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    """Minimal canonical evidence shape consumed by the deterministic kernel."""

    event_id: str
    global_seq: int
    conversation_id: str
    conversation_seq: int
    event_type: str
    source: str
    created_at: datetime
    text: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CueState:
    """Present-state cues that may activate stored evidence."""

    query_text: str | None = None
    entities: tuple[str, ...] = ()
    reference_time: datetime | None = None
    conversation_id: str | None = None
    source_types: tuple[str, ...] = ()
    ignored_terms: tuple[str, ...] = ()
    limit: int = 5
    minimum_score: float = 0.15

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("limit must be >= 1")
        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError("minimum_score must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ScoreComponents:
    lexical: float = 0.0
    entity: float = 0.0
    temporal: float = 0.0
    conversation: float = 0.0
    total: float = 0.0


@dataclass(frozen=True, slots=True)
class RecallCandidate:
    event: MemoryEvent
    score: ScoreComponents
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecallTraceItem:
    event_id: str
    global_seq: int
    score: ScoreComponents
    reasons: tuple[str, ...]
    selected: bool


@dataclass(frozen=True, slots=True)
class RecallTrace:
    policy_version: str
    cue_tokens: tuple[str, ...]
    normalized_entities: tuple[str, ...]
    candidates_considered: int
    items: tuple[RecallTraceItem, ...]


@dataclass(frozen=True, slots=True)
class MemoryPacket:
    """Bounded evidence returned to a stateless consumer plus its audit trace."""

    items: tuple[MemoryEvent, ...]
    trace: RecallTrace


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalized_nonempty(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(normalize_text(value) for value in values)
    return tuple(value for value in normalized if value)


def _lexical_score(
    query_text: str | None,
    event_text: str,
    ignored_terms: Sequence[str] = (),
    *,
    policy: MemoryScoringPolicy = DEFAULT_MEMORY_SCORING_POLICY,
) -> float:
    if not query_text:
        return 0.0
    ignored = set(_normalized_nonempty(ignored_terms))
    query_tokens = tuple(token for token in tokenize(query_text) if token not in ignored)
    if not query_tokens:
        return 0.0
    event_tokens = set(tokenize(event_text))
    if not event_tokens:
        return 0.0

    unique_query = set(query_tokens)
    coverage = len(unique_query & event_tokens) / len(unique_query)
    normalized_query = normalize_text(query_text)
    normalized_event = normalize_text(event_text)
    phrase_bonus = (
        policy.phrase_bonus
        if normalized_query and normalized_query in normalized_event
        else 0.0
    )
    return _clamp01(coverage + phrase_bonus)


def _entity_score(
    entities: Sequence[str],
    event_text: str,
    payload: Mapping[str, Any],
) -> float:
    normalized_entities = _normalized_nonempty(entities)
    if not normalized_entities:
        return 0.0
    haystack = normalize_text(event_text)
    payload_entities = {
        normalized
        for value in payload.get("entities", [])
        if isinstance(value, (str, int, float))
        if (normalized := normalize_text(str(value)))
    }
    hits = sum(
        1
        for normalized in normalized_entities
        if normalized in haystack or normalized in payload_entities
    )
    return hits / len(normalized_entities)


def _temporal_score(
    reference_time: datetime | None,
    created_at: datetime,
    *,
    policy: MemoryScoringPolicy = DEFAULT_MEMORY_SCORING_POLICY,
) -> float:
    if reference_time is None:
        return 0.0
    ref = reference_time if reference_time.tzinfo else reference_time.replace(tzinfo=timezone.utc)
    event_time = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
    delta_days = abs((ref - event_time).total_seconds()) / 86400.0
    return 1.0 / (1.0 + delta_days / policy.temporal_scale_days)


def score_event(
    event: MemoryEvent,
    cue: CueState,
    *,
    policy: MemoryScoringPolicy = DEFAULT_MEMORY_SCORING_POLICY,
) -> ScoreComponents:
    """Score one event with explicit, inspectable deterministic components."""
    lexical = _lexical_score(cue.query_text, event.text, cue.ignored_terms, policy=policy)
    entity = _entity_score(cue.entities, event.text, event.payload)
    temporal = _temporal_score(cue.reference_time, event.created_at, policy=policy)
    conversation = 1.0 if cue.conversation_id and cue.conversation_id == event.conversation_id else 0.0
    total = _clamp01(
        policy.lexical_weight * lexical
        + policy.entity_weight * entity
        + policy.temporal_weight * temporal
        + policy.conversation_weight * conversation
    )
    return ScoreComponents(
        lexical=lexical,
        entity=entity,
        temporal=temporal,
        conversation=conversation,
        total=total,
    )


def _reasons(score: ScoreComponents) -> tuple[str, ...]:
    reasons: list[str] = []
    if score.lexical > 0:
        reasons.append("LEXICAL_CUE")
    if score.entity > 0:
        reasons.append("ENTITY_CUE")
    if score.temporal > 0:
        reasons.append("TEMPORAL_CUE")
    if score.conversation > 0:
        reasons.append("CONVERSATION_CUE")
    return tuple(reasons)


def _rank_with_temporal_diversity(
    candidates: list[RecallCandidate],
) -> list[RecallCandidate]:
    """Preserve chronology when relevance evidence cannot distinguish candidates.

    Equal total relevance scores carry no evidence that one chronology endpoint
    is semantically preferable to the other. Newest-only tie breaking therefore
    creates an arbitrary information-loss boundary for deep same-topic history.
    For each exact score tie this policy keeps the longstanding newest-first
    preference for the first item, then alternates oldest/newest endpoints.

    This is intentionally query-language agnostic. Words such as "originally"
    are not recognized here; the kernel simply avoids destroying temporal
    alternatives before a stateless consumer can interpret the question.
    """

    by_score = sorted(
        candidates,
        key=lambda candidate: (-candidate.score.total, candidate.event.event_id),
    )
    ranked: list[RecallCandidate] = []
    for _, tied_iter in groupby(by_score, key=lambda candidate: candidate.score.total):
        tied = deque(
            sorted(
                tied_iter,
                key=lambda candidate: (
                    candidate.event.global_seq,
                    candidate.event.event_id,
                ),
            )
        )
        take_newest = True
        while tied:
            ranked.append(tied.pop() if take_newest else tied.popleft())
            take_newest = not take_newest
    return ranked


def recall(
    events: Iterable[MemoryEvent],
    cue: CueState,
    *,
    scoring_policy: MemoryScoringPolicy = DEFAULT_MEMORY_SCORING_POLICY,
) -> MemoryPacket:
    """Return a bounded, deterministic evidence packet for the given cue state."""
    candidates: list[RecallCandidate] = []
    has_activation_cues = bool(cue.query_text or cue.entities or cue.reference_time or cue.conversation_id)
    allowed_types = set(cue.source_types)

    for event in events:
        if allowed_types and event.event_type not in allowed_types:
            continue
        score = score_event(event, cue, policy=scoring_policy)
        if has_activation_cues and score.total < cue.minimum_score:
            continue
        candidates.append(RecallCandidate(event=event, score=score, reasons=_reasons(score)))

    if has_activation_cues:
        candidates = _rank_with_temporal_diversity(candidates)
    else:
        candidates.sort(key=lambda candidate: (-candidate.event.global_seq, candidate.event.event_id))

    selected = candidates[: cue.limit]
    selected_ids = {candidate.event.event_id for candidate in selected}
    trace_items = tuple(
        RecallTraceItem(
            event_id=candidate.event.event_id,
            global_seq=candidate.event.global_seq,
            score=candidate.score,
            reasons=candidate.reasons,
            selected=candidate.event.event_id in selected_ids,
        )
        for candidate in candidates
    )
    ignored_terms = set(_normalized_nonempty(cue.ignored_terms))
    normalized_entities = _normalized_nonempty(cue.entities)
    trace = RecallTrace(
        policy_version=POLICY_VERSION,
        cue_tokens=tuple(token for token in tokenize(cue.query_text or "") if token not in ignored_terms),
        normalized_entities=normalized_entities,
        candidates_considered=len(candidates),
        items=trace_items,
    )
    return MemoryPacket(items=tuple(candidate.event for candidate in selected), trace=trace)


def reciprocal_rank(packet: MemoryPacket, relevant_event_ids: set[str]) -> float:
    """Convenience metric used by the synthetic-life benchmark."""
    for index, event in enumerate(packet.items, start=1):
        if event.event_id in relevant_event_ids:
            return 1.0 / index
    return 0.0
