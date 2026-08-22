"""Splitting a rewrite into the labelled sections it is supposed to contain.

Three fields for the T2VA family, six for full-reference mode. A model that
ignored the output contract still has to yield something usable downstream, so
an unlabelled answer lands whole in the body field rather than nowhere.
"""

from __future__ import annotations

import re

from .constants import OUTPUT_FIELDS

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def answer_only(text: str) -> str:
    """The answer without the model's reasoning in front of it.

    Asking for no reasoning is not the same as getting none. Gemma-4's own
    decoder rewrites its thought channel into ``<think>``/``</think>``, and an
    E4B build here writes a full analysis into it even though the prompt primes
    that channel closed -- ComfyUI's own 'Generate Text' node shows the same
    thing on the same checkpoint, so it is the model, not the wiring. Qwen3-VL
    does it on the GGUF route too, where ``llama-mtmd-cli`` has no
    ``enable_thinking`` to forward. A caption is one line of a reference block,
    so the reasoning is cut rather than shipped: everything up to the last close
    tag goes, and an unclosed block -- which is what a truncated answer leaves
    behind -- goes with it.
    """
    if THINK_CLOSE in text:
        text = text.rsplit(THINK_CLOSE, 1)[-1]
    elif THINK_OPEN in text:
        text = text.split(THINK_OPEN, 1)[0]
    return text.strip()


_PATTERNS: dict[tuple[str, ...], re.Pattern] = {}


def _pattern(names: tuple[str, ...]) -> re.Pattern:
    """A label matcher for one set of field names, built once per set.

    The leading character class forgives the decoration a general-purpose model
    adds when it is halfway through obeying the contract: bold, a heading, a
    quote marker, a list bullet.
    """
    cached = _PATTERNS.get(names)
    if cached is None:
        cached = re.compile(
            r"^[ \t]*[*_#>\-\s]*(" + "|".join(re.escape(name) for name in names) + r")[*_ \t]*:[ \t]*",
            re.IGNORECASE | re.MULTILINE,
        )
        _PATTERNS[names] = cached
    return cached


def split_sections(
    text: str,
    names: tuple[str, ...] = OUTPUT_FIELDS,
    fallback: str = "",
) -> tuple[str, dict[str, str]]:
    """Return ``(text before the first label, {name: section})``.

    The head is where the alignment instruction of I2VA, FL2VA and L2VA lives:
    it belongs to the prompt but is not one of the fields.
    """
    result = {name: "" for name in names}
    matches = list(_pattern(names).finditer(text or ""))

    if not matches:
        result[fallback or names[0]] = (text or "").strip()
        return "", result

    head = (text[: matches[0].start()] or "").strip()
    for index, match in enumerate(matches):
        name = match.group(1).lower()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result[name] = text[match.end():end].strip()
    return head, result


def split_fields(
    text: str,
    names: tuple[str, ...] = OUTPUT_FIELDS,
    fallback: str = "",
) -> dict[str, str]:
    """The labelled sections of a rewrite, without the leading instruction."""
    return split_sections(text, names, fallback)[1]


def missing(sections: dict[str, str], names: tuple[str, ...] = OUTPUT_FIELDS) -> list[str]:
    """Which required fields the answer did not actually fill in."""
    return [name for name in names if not sections.get(name)]
