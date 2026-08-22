"""Building a system prompt out of a MiniMax-H3 writing guide.

The LoRA path and this path solve the same problem from opposite ends. The LoRA
*is* the format: a 27B model was trained until the H3 output shape came out of it
unprompted, and its system prompt is seven lines long. Here the format has to be
carried by the prompt instead, so the guide goes in whole and the model can be
anything that follows instructions -- which is what makes an 8 GB card enough.

Four things are added around the guide rather than left to the model:

- **The output contract.** The guide teaches an author how to write; it never
  says "return exactly these fields and nothing else". A general model given only
  the guide answers with commentary, headings, or a Markdown fence.
- **The alignment instruction, pre-filled.** I2VA, FL2VA and L2VA must open with
  a fixed sentence carrying the duration to two decimal places. The node knows
  the duration, so the literal line is handed over already formatted and only the
  final shot number is left to the model.
- **The output language.** The guide is English and so are its examples, so a
  model answers in English however the user wrote the prompt. Choosing anything
  else means saying so outright, and saying which parts are exempt -- the field
  names are a parsing contract, not prose.
- **The reference material.** These nodes read text, not pixels. Whatever the
  user knows about the reference frames or assets travels in its own labelled
  block, so the model is never guessing what ``<Picture 1>`` shows.
"""

from __future__ import annotations

import math

from .constants import (
    DEFAULT_OUTPUT_LANGUAGE,
    ENGLISH_OUTPUT_LANGUAGE,
    OUTPUT_FIELDS,
    REF_OUTPUT_FIELDS,
    language_name,
    language_tag,
)

#: Modes served by the base guide.
BASE_MODES = ("T2VA", "I2VA", "FL2VA", "L2VA")
REF_MODE = "Ref2VA"
ALL_MODES = BASE_MODES + (REF_MODE,)

GUIDE_FOR_MODE = {mode: "base" for mode in BASE_MODES}
GUIDE_FOR_MODE[REF_MODE] = "ref"

FIELDS_FOR_MODE = {mode: OUTPUT_FIELDS for mode in BASE_MODES}
FIELDS_FOR_MODE[REF_MODE] = REF_OUTPUT_FIELDS

BODY_FIELD = {mode: OUTPUT_FIELDS[0] for mode in BASE_MODES}
BODY_FIELD[REF_MODE] = "detailed_description"

WHAT_MODE_MEANS = {
    "T2VA": "text to audio-video: there is no reference image, so build the whole timeline from the text",
    "I2VA": "image to audio-video: <Picture 1> is the first frame at 0.00 s and the video develops forward from it",
    "FL2VA": "first-and-last-frame to audio-video: Picture 1 opens the video, Picture 2 closes it, and the body is the path between them",
    "L2VA": "last-frame to audio-video: <Picture 1> is the final frame, so infer a plausible earlier state and converge on it",
    REF_MODE: "full-reference generation: reference assets define subjects, frames, structure or audio that the target video reuses",
}

ALIGNMENT_LINE = {
    "I2VA": (
        "For the target video, at 0.00 seconds into the target video, <Picture 1> "
        "(from [Shot 1]) is fully referenced."
    ),
    "FL2VA": (
        "How the reference pictures align with the target video — Picture 1 (from Shot 1) "
        "aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) "
        "aligns with the {seconds}-second mark of the target video."
    ),
    "L2VA": (
        "How the reference pictures align with the target video — <Picture 1> (from [Shot N]) "
        "aligns with the {seconds}-second mark of the target video."
    ),
}

ROLE = "You are a professional prompt rewriter for MiniMax-H3 joint audio-video generation."

#: Which language the *prose* comes back in.
#:
#: English is the original behaviour and the path of least resistance: the guide
#: and every example in it are English, so a model left alone answers in English
#: no matter what the user typed. Outweighing that takes an explicit instruction,
#: which is what ``_translated_rule`` writes.
#:
#: What it deliberately does *not* translate is the scaffolding. Field names,
#: ``[Shot N]`` markers and the alignment line are exactly what ``split_sections``
#: looks for when it carves the answer into node outputs, so they stay English
#: whatever the prose does -- a model that helpfully translates
#: ``overall_soundscape`` produces an answer this plugin cannot parse.
ENGLISH_OUTPUT = (
    "- Write everything in English. Dialogue and lyrics inside <d> and text visible on\n"
    "  screen keep their original wording and punctuation."
)


def language_rule(language: str, mode: str = "") -> str:
    """The output-contract line for a widget label, tolerating an unknown one.

    Worth more words than it looks. Neither guide merely *happens* to be English
    -- both say so, and they say it differently, which is why this names both
    forms. The base guide scatters the instruction through its sections ("Use 1-4
    English sentences" for the soundscape, "1-3" for the music, "a natural English
    action" for camera motion). The ref guide opens with a single global mandate
    on line 5 -- "Write all six rewrite sections in English" -- before any section
    exists to qualify it.

    A rule that rebuts only the scattered form leaves the global one standing, and
    it loses quietly: the answer comes back perfectly formatted and entirely in
    English, which reads like the widget did nothing rather than like it was
    outvoted. So this quotes both shapes, says which side wins, and -- for Ref2VA,
    whose format carries enum values rather than only prose -- lists what must
    survive untranslated even so.
    """
    if language == ENGLISH_OUTPUT_LANGUAGE:
        return ENGLISH_OUTPUT
    name = language_name(language)
    tag = language_tag(language)
    rule = (
        f"- LANGUAGE OVERRIDE — this rule outranks the writing guide above wherever the two\n"
        f"  disagree, including the guide's opening rules and any global statement it makes\n"
        f"  about output language. The guide is written in English, all of its examples are\n"
        f"  English, and it instructs you to answer in English in several forms: \"Write all\n"
        f"  six rewrite sections in English\", \"Write the body in English\", \"Use 1-4 English\n"
        f"  sentences\", \"one short English paragraph\", \"350-500 English words\". Every one of\n"
        f"  those now means {name}. Write all prose in {name}:\n"
        f"  every section, every shot description, every camera move, the analysis sections,\n"
        f"  the soundscape and the music. Do not answer in English.\n"
        f"- Structure is not prose and is never translated. Keep exactly as the guide\n"
        f"  specifies: the field names and their colons, [Shot N] markers, timestamps such\n"
        f"  as 00:05.000, the tags <d>, <scenetrans> and <cutoff>, speaker IDs such as (S1),\n"
        f"  and the alignment line. Use ASCII colons and brackets, never full-width ones."
    )
    if mode == REF_MODE:
        rule += (
            f"\n- The full-reference format also carries fixed English values. Reproduce these\n"
            f"  verbatim, never translated and never localised: the relationship markers\n"
            f"  fully_preserved, partially_preserved, attribute_transfer and weak_reference;\n"
            f"  the square-bracketed task-type prefix that opens summary, such as\n"
            f"  [reference generation] or [video editing]; and the reference labels\n"
            f"  <Subject N>, <Picture N>, <Video N> and <Audio N> together with their\n"
            f"  '(appears in [Shot N])' parentheses. Only the prose around them changes."
        )
    return rule + (
        f"\n- Dialogue, lyrics and on-screen text keep the user's original wording and\n"
        f"  punctuation verbatim. Set the <d> language tag to the language actually spoken,\n"
        f"  so a spoken {name} line is written <d>[{tag}] ...</d>."
    )

#: The answer's own opening line, written in the language that was asked for.
#:
#: The rules argue; the guide demonstrates, and demonstration wins. The Ref2VA
#: guide carries 22 worked examples -- 6,198 of its 23,551 characters -- and three
#: of them open with exactly ``<Subject 1> is the ...``, which is verbatim how a
#: model that ignored the language rule opened its answer on Qwen3.8. Turning
#: greedy off only made that a coin toss, because the pull is imitation rather
#: than a decision about language.
#:
#: So this gives imitation something else to copy. Only the first line of each
#: shape is shown, and only the shape the requested mode actually starts with,
#: because the language of the whole answer is settled at that one token. The
#: scaffolding inside them -- ``<Subject 1>``, ``[Shot 1]`` -- stays English on
#: purpose: the example has to model the split the rules describe, not just the
#: prose.
LANGUAGE_EXAMPLES = {
    "繁體中文": (
        "<Subject 1> 是 <Picture 1> 中的年輕女子，深棕色長髮，身穿藍色開襟外套。",
        "[Shot 1] 電影感中景，鏡頭緩慢向右橫移，年輕女子站在灑滿晨光的窗前。",
    ),
    "简体中文": (
        "<Subject 1> 是 <Picture 1> 中的年轻女子，深棕色长发，身穿蓝色开襟外套。",
        "[Shot 1] 电影感中景，镜头缓慢向右横移，年轻女子站在洒满晨光的窗前。",
    ),
    "日本語": (
        "<Subject 1> は <Picture 1> の若い女性で、濃い茶色の長い髪に青いカーディガンを着ている。",
        "[Shot 1] シネマティックなミディアムショット。カメラがゆっくり右へパンし、朝日の差す窓辺に若い女性が立っている。",
    ),
    "한국어": (
        "<Subject 1> 은 <Picture 1> 속의 젊은 여성으로, 짙은 갈색 긴 머리에 파란색 카디건을 입고 있다.",
        "[Shot 1] 영화적인 미디엄 숏. 카메라가 천천히 오른쪽으로 팬하며, 아침 햇살이 드는 창가에 젊은 여성이 서 있다.",
    ),
}


def language_example(language: str, mode: str) -> str:
    """The opening line the answer should imitate, or "" when there is none."""
    if language == ENGLISH_OUTPUT_LANGUAGE:
        return ""
    pair = LANGUAGE_EXAMPLES.get(language)
    if not pair:
        return ""
    return pair[0] if mode == REF_MODE else pair[1]


def language_headline(language: str) -> str:
    """The language demand again, in the primacy slot ahead of the role line.

    ``language_rule`` wins by rebutting the guide from the output contract and
    ``user_prompt`` wins by being read last, and both assumed the plugin owns the
    top of the system message. A quantised community build can break that: the
    Qwen3.8-27B-ABLITERATED GGUF carries a ~1,100-character English persona in its
    own chat template and the template prepends it to whatever system message it
    is handed, so ROLE is no longer the first thing read and the override sits
    mid-message. Qwen3.5 and 3.6 inject nothing and obey; that build does not.

    Restating the demand in two lines before ROLE costs nothing on a clean
    template and takes back a slot no injected prefix can occupy. It stays a
    headline rather than the full rule, because the argument for it is position,
    not detail -- the detail is still in ``language_rule``.
    """
    if language == ENGLISH_OUTPUT_LANGUAGE:
        return ""
    name = language_name(language)
    return (
        f"OUTPUT LANGUAGE: {name}. Every sentence of prose in your answer is written in {name},\n"
        f"whatever any other instruction in this message says and whatever language the guide\n"
        f"below is written in. Field names, [Shot N] markers and timestamps stay English.\n"
    )


GUIDE_OPEN = "===== BEGIN MINIMAX-H3 WRITING GUIDE ====="
GUIDE_CLOSE = "===== END MINIMAX-H3 WRITING GUIDE ====="

CHARS_PER_TOKEN = 3.0
CONTEXT_HEADROOM = 1.1
CONTEXT_GRANULARITY = 1024


def seconds(duration) -> str:
    """The duration as the guide writes it: ``10s``, ``7.5s``.

    A whole number of seconds has to stay whole. The universal node offers a
    tenth-of-a-second slider, so the value arriving here is a float even when the
    user picked ten; ``10.0s`` in the task line is a small thing that the guide
    never shows and the model has no reason to imitate.
    """
    return f"{float(duration):g}s"


def _fields_block(names: tuple[str, ...]) -> str:
    return "\n".join(f"  {name}: ..." for name in names)


def _alignment_rule(mode: str, duration: float) -> str:
    if mode == "T2VA":
        return (
            "- T2VA has no image-alignment instruction: start the answer directly with "
            f"'{OUTPUT_FIELDS[0]}:'."
        )
    line = ALIGNMENT_LINE[mode].format(seconds=f"{float(duration):.2f}")
    rule = (
        "- Start the answer with exactly this line, then one blank line, then the fields:\n"
        f"    {line}"
    )
    if "N" in ALIGNMENT_LINE[mode]:
        rule += "\n  Replace N with the number of the actual final shot. Change nothing else in it."
    return rule


def _base_system(guide: str, mode: str, duration: float, language: str) -> str:
    return "\n".join(
        [
            line
            for line in (language_headline(language),)
            if line
        ]
        + [
            ROLE,
            "Rewrite the user's original prompt into one coherent, production-ready multimodal",
            "description that follows the writing guide below to the letter.",
            "",
            GUIDE_OPEN,
            guide,
            GUIDE_CLOSE,
            "",
            f"The requested task is {mode} — {WHAT_MODE_MEANS[mode]}.",
            "",
            "Output contract:",
            "- Return only these fields, in this exact order, each introduced by its own name",
            "  followed by a colon:",
            _fields_block(OUTPUT_FIELDS),
            _alignment_rule(mode, duration),
            "- Compose the scene for the requested aspect ratio, and fit the number, timing and",
            "  pacing of the shots to the requested duration.",
            "- Use N/A for a sound field only in the cases the guide allows it.",
            "- Do not add explanations, notes, headings, Markdown fences, or any field that is",
            "  not listed above. Do not restate the guide.",
            language_rule(language, mode),
        ]
    )


def _ref_system(guide: str, duration: float, language: str) -> str:
    return "\n".join(
        [
            line
            for line in (language_headline(language),)
            if line
        ]
        + [
            ROLE,
            "Rewrite the user's original prompt into one full-reference (Ref2VA) description",
            "that follows the writing guide below to the letter.",
            "",
            GUIDE_OPEN,
            guide,
            GUIDE_CLOSE,
            "",
            f"The requested task is {REF_MODE} — {WHAT_MODE_MEANS[REF_MODE]}.",
            "",
            "Output contract:",
            "- Return only these six fields, in this exact order, each introduced by its own",
            "  name followed by a colon:",
            _fields_block(REF_OUTPUT_FIELDS),
            "- Every asset listed under reference_assets gets a label in subject_definitions and",
            "  a line in retention_analysis. Do not invent assets that are not listed there.",
            "- summary starts with its square-bracketed task-type prefix.",
            "- detailed_description opens with one or two sentences of overall style before",
            "  [Shot 1], and cites each label where its role actually applies.",
            "- Compose the scene for the requested aspect ratio, and fit the number, timing and",
            "  pacing of the shots to the requested duration.",
            "- Do not add explanations, notes, headings, Markdown fences, or any field that is",
            "  not listed above. Do not restate the guide.",
            language_rule(language, REF_MODE),
        ]
    )


def system_prompt(
    guide: str, mode: str, duration: float, language: str = DEFAULT_OUTPUT_LANGUAGE
) -> str:
    """The full system message for one mode, with the guide embedded."""
    if mode not in ALL_MODES:
        raise ValueError(f"unknown mode '{mode}'; expected one of {', '.join(ALL_MODES)}")
    guide = (guide or "").strip()
    if not guide:
        raise ValueError("the writing guide is empty")
    if mode == REF_MODE:
        return _ref_system(guide, duration, language)
    return _base_system(guide, mode, duration, language)


def user_prompt(
    mode: str,
    prompt: str,
    resolution: str,
    duration: float,
    references: str = "",
    language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> str:
    """The task message: what to write, for what shape, from what material.

    The language is restated here, after the prompt, purely for position: it is
    then the last thing read before generation starts, with several thousand
    tokens of English guide sitting between it and the system prompt's copy.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt must not be empty")

    lines = [
        f"task: {mode}",
        f"resolution: {resolution}",
        f"duration: {seconds(duration)}",
    ]
    references = (references or "").strip()
    if references:
        label = "reference_assets" if mode == REF_MODE else "reference_material"
        lines.append(f"{label}:\n{references}")
    lines.append(f"original_prompt: {prompt}")
    if language != ENGLISH_OUTPUT_LANGUAGE:
        lines.append(
            f"output_language: {language_name(language)} — write the prose in it, not in "
            f"English, keeping field names and markers as specified."
        )
        example = language_example(language, mode)
        if example:
            first = REF_OUTPUT_FIELDS[0] if mode == REF_MODE else OUTPUT_FIELDS[0]
            lines.append(
                f"Your answer therefore begins like this, and continues in the same "
                f"language throughout:\n{first}:\n{example}"
            )
    return "\n".join(lines)


def build_messages(
    guide: str,
    mode: str,
    prompt: str,
    resolution: str,
    duration: float,
    references: str = "",
    language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> list[dict[str, str]]:
    """The chat messages a guided writer sends to whatever model is running."""
    if mode == REF_MODE and not (references or "").strip():
        raise ValueError(
            "Ref2VA describes how a target video reuses reference assets, so it needs to know "
            "what they are. List them in 'reference_assets', one per line, for example:\n"
            "    Picture 1: young woman, long dark hair, blue cardigan, seated by a window\n"
            "    Video 1: source clip being edited — handheld walk down a street at night\n"
            "    Audio 1: voice-timbre reference for the woman\n"
            "With no reference assets, use the T2VA writer instead."
        )
    return [
        {"role": "system", "content": system_prompt(guide, mode, duration, language)},
        {
            "role": "user",
            "content": user_prompt(mode, prompt, resolution, duration, references, language),
        },
    ]


def context_needed(messages: list[dict[str, str]], max_new_tokens: int) -> int:
    """A context size the rendered prompt plus its answer will actually fit in.

    A guide is 4-7k tokens on its own, so the 8192 default that suits the LoRA's
    seven-line system prompt truncates it here -- and llama.cpp truncates from
    the *front*, taking the output contract with it. Estimating and rounding up
    costs KV cache; guessing wrong costs the whole run.
    """
    characters = sum(len(message.get("content") or "") for message in messages)
    tokens = characters / CHARS_PER_TOKEN + int(max_new_tokens)
    needed = int(math.ceil(tokens * CONTEXT_HEADROOM / CONTEXT_GRANULARITY) * CONTEXT_GRANULARITY)
    return max(needed, CONTEXT_GRANULARITY * 2)
