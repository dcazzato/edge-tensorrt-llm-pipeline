"""Defensive parsing of VLM replies.

The prompt ASKS for a compact JSON object; a language model, however, is not
an API — it adds fences, preambles, translations and Python-style literals.
This module's philosophy is "lose gracefully": every failure mode degrades to
``(raw text, checks_out=None)`` instead of raising, because

* the description the model DID write is still information worth returning;
* an unparsable reply is "no opinion" (None), NEVER "disagreement" (false) —
  inventing the difference from a garbled string would fake an alarm signal.

What it does NOT do, on purpose:

* guess ``checks_out`` from prose ("looks good to me!") — sentiment-guessing
  a multilingual chatty model is a bug generator;
* raise on malformed input — the provider wraps this in a 502 only when
  nothing at all could be recovered.
"""

import json
from typing import Any, NamedTuple


class ParsedAnswer(NamedTuple):
    """The only two facts worth extracting from a reply."""

    description: str
    checks_out: bool | None


def _find_json_object(text: str) -> str | None:
    """First balanced ``{...}`` substring, aware of strings and escapes.

    A naive regex breaks on braces inside quoted strings; this small scanner
    counts depth and skips string contents (and backslash escapes). If the
    first ``{`` never balances, later ones are tried — models sometimes
    quote broken JSON inside valid JSON.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            char = text[i]
            if escaped:
                escaped = False
                continue
            if in_string and char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _repair(candidate: str) -> str:
    """Cheap fixes for the classic wounds: Python literals, smart quotes."""
    fixed = candidate.replace("True", "true").replace("False", "false").replace("None", "null")
    for weird, plain in {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}.items():
        fixed = fixed.replace(weird, plain)
    return fixed


def _loads_lenient(candidate: str) -> dict[str, Any] | None:
    """json.loads with one repair round; only dicts count as answers."""
    for attempt in (candidate, _repair(candidate)):
        try:
            obj = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _coerce_checks_out(value: Any) -> bool | None:
    """Tri-state coercion: anything unclear becomes ``None`` — never a guess."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower().rstrip(".")
        if normalized in {"true", "yes"}:
            return True
        if normalized in {"false", "no"}:
            return False
    return None


def parse_reply(text: str) -> ParsedAnswer:
    """Extract ``(description, checks_out)`` from a raw VLM reply.

    Never raises. On any failure the description falls back to the raw text
    and ``checks_out`` to ``None`` — "the model said something we could not
    structure", which is exactly what happened.
    """
    raw = text.strip()
    candidate = _find_json_object(text)
    if candidate is not None:
        obj = _loads_lenient(candidate)
        if obj is not None:
            description = obj.get("description")
            if not isinstance(description, str) or not description.strip():
                description = raw
            return ParsedAnswer(
                description=description.strip(),
                checks_out=_coerce_checks_out(obj.get("checks_out")),
            )
    return ParsedAnswer(description=raw, checks_out=None)
