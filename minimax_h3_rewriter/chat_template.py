"""Rendering a GGUF's own chat template, the way Transformers would.

The upstream reference passes ``enable_thinking=False`` to
``apply_chat_template``; without it Qwen3.6 spends hundreds of tokens reasoning
before it starts the rewrite. llama-cpp-python's chat formatter has no way to
forward that flag, so the template is taken out of the GGUF metadata and
rendered here, in a sandbox set up like the one Transformers uses.
"""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

TEMPLATE_KEY = "tokenizer.chat_template"
BOS_KEY = "tokenizer.ggml.bos_token_id"
EOS_KEY = "tokenizer.ggml.eos_token_id"

#: A template that sets ``enable_thinking`` itself, before it reads ours.
#:
#: Gemma-4-31B's template opens with ``{%- set enable_thinking = true -%}``, so
#: the flag rendered here is overwritten by the first statement of the template
#: it is passed to and the model reasons anyway -- hundreds of tokens of English
#: analysis in front of a rewrite that was asked for three fields and nothing
#: else. The same template already knows how to be quiet: told thinking is off
#: it ends the prompt with an empty, already-closed thought channel
#: (``<|channel>thought\n<channel|>``) and the model starts on the answer. Only
#: the assignment stands between the two behaviours.
#:
#: So an explicit ``enable_thinking=False`` deletes it. A template that leaves
#: the flag to its caller -- which is nearly all of them -- has nothing to match
#: and is rendered exactly as before.
FORCED_THINKING = re.compile(
    r"\{%-?\s*set\s+enable_thinking\s*=\s*[Tt]rue\s*-?%\}\s*"
)


class TemplateError(RuntimeError):
    pass


def _environment():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise TemplateError(message)

    def tojson(value, indent=None, **_kwargs):
        return json.dumps(value, ensure_ascii=False, indent=indent)

    def strftime_now(fmt):
        # Templates that stamp a date are rare here; a fixed value keeps the
        # rendered prompt deterministic, which the seed contract depends on.
        return ""

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    env.filters["tojson"] = tojson
    return env


def _fold_system(messages: list[dict[str, str]]) -> list[dict[str, str]] | None:
    """Move a leading system message into the first user turn.

    Not every chat template accepts a system role -- Gemma's calls
    ``raise_exception`` on one, and it is not alone. Since the guided writers put
    a whole writing guide in that message, dropping it is not an option and
    refusing the model is a poor answer to "run this on anything". Prepending it
    to the first user turn is what those templates' own model cards prescribe.

    Returns ``None`` when there is nothing to fold, so the caller can re-raise
    the original error instead of pretending it tried something.
    """
    if not messages or messages[0].get("role") != "system":
        return None
    system = (messages[0].get("content") or "").strip()
    rest = [dict(message) for message in messages[1:]]
    for message in rest:
        if message.get("role") == "user":
            message["content"] = f"{system}\n\n{message.get('content') or ''}".strip()
            return rest
    return None


def render(
    template: str,
    messages: list[dict[str, str]],
    bos_token: str = "",
    eos_token: str = "",
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> str:
    if not template:
        raise TemplateError("the model file carries no chat template")

    if not enable_thinking:
        template, forced = FORCED_THINKING.subn("", template)
        if forced:
            log.debug(
                "[minimax_h3_rewriter.chat_template.render] template set enable_thinking "
                "itself; dropped %d assignment(s) so the flag passed in is the one that "
                "applies", forced,
            )

    def attempt(payload):
        return _environment().from_string(template).render(
            messages=payload,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            bos_token=bos_token,
            eos_token=eos_token,
            tools=None,
        )

    try:
        return attempt(messages)
    except Exception as error:
        folded = _fold_system(messages)
        if folded is None:
            raise
        log.debug(
            "[minimax_h3_rewriter.chat_template.render] template rejected the system role (%s), "
            "folding it into the first user turn", error,
        )
        try:
            return attempt(folded)
        except Exception:
            raise error


def from_metadata(metadata: dict, messages: list[dict[str, str]], **kwargs) -> str:
    """Render using the ``tokenizer.chat_template`` entry of a GGUF's metadata."""
    template = metadata.get(TEMPLATE_KEY) or metadata.get("chat_template")
    if not template:
        raise TemplateError(
            "this GGUF has no embedded chat template, so the rewriter cannot build its prompt"
        )
    return render(template, messages, **kwargs)
