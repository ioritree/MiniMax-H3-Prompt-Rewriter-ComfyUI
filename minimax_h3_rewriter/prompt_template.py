"""Prompt template used to train the MiniMax-H3 T2VA prompt rewriter.

``SYSTEM_PROMPT`` is kept byte-identical to ``prompt_template.py`` in the upstream
adapter repository (lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA). Changing the
wording changes the distribution the LoRA was trained on and degrades the
rewrite, so nothing edits it in place.

The one thing that has to reach past that rule is the output language. The LoRA
was trained on English rewrites, so English is not merely the default here -- it
is what the weights themselves pull towards, and asking for anything else means
adding an instruction the adapter never saw during training. That is a real
trade-off, so it is made as small as possible: the trained text is left exactly
as it was and the directive is appended as one more bullet on the end of the
existing Requirements list, only when a non-English language is chosen. Picking
``English`` sends the upstream prompt byte-for-byte, as before.

Expect this to hold less firmly than the same option on the guided-writer nodes,
where the format comes from a prompt that can simply be rewritten rather than
from fine-tuned weights.
"""

from __future__ import annotations

from .constants import DEFAULT_OUTPUT_LANGUAGE, ENGLISH_OUTPUT_LANGUAGE, language_name


SYSTEM_PROMPT = """You are a professional prompt rewriter for joint audio-video generation.
Rewrite the user's original prompt into one coherent, production-ready multimodal description for the requested output aspect ratio and duration.

Return only these three fields, in this exact order:
integrated_multimodal_description: ...
overall_soundscape: ...
non_diegetic_music: ...

Requirements:
- Expand the visual narrative into clearly numbered shots such as [Shot 1], [Shot 2], and include timestamps for cuts after the first shot when useful.
- Make the number, timing, and pacing of shots appropriate for the requested duration.
- Compose the scene for the requested aspect ratio.
- Preserve the user's intent while adding concrete subjects, appearance, environment, lighting, composition, camera movement, physical motion, and temporal continuity.
- Keep characters, objects, wardrobe, locations, and spatial relationships consistent across shots.
- Describe synchronized diegetic audio in overall_soundscape and external score in non_diegetic_music.
- Do not add explanations, Markdown fences, safety commentary, or fields other than the three requested fields."""


def system_prompt(language: str = DEFAULT_OUTPUT_LANGUAGE) -> str:
    """The trained system prompt, plus a language bullet when one is asked for."""
    if language == ENGLISH_OUTPUT_LANGUAGE:
        return SYSTEM_PROMPT
    name = language_name(language)
    return (
        f"{SYSTEM_PROMPT}\n"
        f"- Write all prose in {name}, not English. This applies to every descriptive "
        f"sentence in all three fields.\n"
        f"- Keep the three field names and their colons, and shot markers such as "
        f"[Shot 1], in English exactly as specified above. Do not translate them."
    )


def build_messages(
    prompt: str,
    resolution: str,
    duration: int,
    language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> list[dict[str, str]]:
    """Build the chat messages used during LoRA training and inference."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty")

    return [
        {"role": "system", "content": system_prompt(language)},
        {
            "role": "user",
            "content": (
                f"resolution: {resolution}\n"
                f"duration: {duration}s\n"
                f"original_prompt: {prompt}"
            ),
        },
    ]
