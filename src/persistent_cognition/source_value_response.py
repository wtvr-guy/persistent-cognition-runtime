"""Copy selected canonical values into application-owned response templates.

Models select token ranges, never replacement values or response prose. Token
offsets are application-owned pointers into the original admitted source string;
copying a range preserves its original spelling and internal whitespace.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from pydantic import BaseModel, ConfigDict, Field


_TOKEN = re.compile(r"\w+(?:[-:/.'’]\w+)*|[^\w\s]")


class SourceValueSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_index: int = Field(ge=0)
    first_token: int = Field(ge=0)
    last_token: int = Field(ge=0)


class SourceValuePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    selections: list[SourceValueSelection]


@dataclass(frozen=True)
class IndexedValueSource:
    text: str
    authority_class: str
    tokens: tuple[tuple[int, int], ...]
    context: str = ""

    @classmethod
    def from_text(
        cls, text: str, authority_class: str, *, context: str = "",
    ) -> IndexedValueSource:
        return cls(text, authority_class,
                   tuple(match.span() for match in _TOKEN.finditer(text)), context)


def structured_value_sources(result: dict) -> tuple[IndexedValueSource, ...]:
    """Expose executor-owned scalar values without JSON string escaping.

    Field paths supply context only; they never become selectable value text.
    Input/result admission has already bounded and validated the result object.
    """

    sources = []

    def visit(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], (*path, str(key)))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, (*path, str(index)))
        else:
            text = value if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False, default=str,
            )
            sources.append(IndexedValueSource.from_text(
                text, "EXTERNAL_TOOL_EVIDENCE", context=".".join(path),
            ))

    visit(result, ())
    return tuple(sources)


def format_value_sources(sources: tuple[IndexedValueSource, ...]) -> str:
    blocks = []
    for index, source in enumerate(sources):
        tokens = "\n".join(
            f"{token_index}: {json.dumps(source.text[start:end], ensure_ascii=False)}"
            for token_index, (start, end) in enumerate(source.tokens)
        )
        blocks.append(
            f"source_index: {index}\nauthority_class: {source.authority_class}\n"
            f"source_context: {json.dumps(source.context, ensure_ascii=False)}\n"
            f"token_index: canonical_token\n{tokens}"
        )
    return "\n\n[Admitted sources: oldest historical event first]\n" + "\n\n".join(blocks)


def resolve_value_plan(
    sources: tuple[IndexedValueSource, ...], plan: SourceValuePlan,
) -> tuple[str, ...]:
    # The bound comes from the actual supplied candidate inventory, not a model
    # claim or an independent numerical limit.
    if len(plan.selections) > sum(len(source.tokens) for source in sources):
        raise ValueError("too many source-value selections")
    values = []
    for selection in plan.selections:
        if selection.source_index not in range(len(sources)):
            raise ValueError("source-value selection references an unknown source")
        source = sources[selection.source_index]
        if not 0 <= selection.first_token <= selection.last_token < len(source.tokens):
            raise ValueError("source-value selection references an invalid token range")
        start = source.tokens[selection.first_token][0]
        end = source.tokens[selection.last_token][1]
        values.append(source.text[start:end])
    return tuple(values)


def render_value_response(values: tuple[str, ...]) -> str:
    if not values:
        raise ValueError("a factual response requires selected source values")
    if len(values) == 1:
        sentence = f"The requested value is {values[0]}"
    else:
        sentence = "The requested values are " + "; ".join(values)
    return sentence if sentence.endswith((".", "?", "!")) else sentence + "."
