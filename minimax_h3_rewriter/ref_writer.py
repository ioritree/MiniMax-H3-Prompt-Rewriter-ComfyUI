"""Seeing the references and writing the Ref2VA prompt in one node.

``Multi Reference Caption`` reads the pixels and ``Prompt Writer (Ref2VA)``
writes the six sections; between them runs a single ``reference_assets`` wire.
Once one model is good enough to do both halves -- a multimodal GGUF that also
follows the writing guide -- that wire is the only thing separating two nodes
that always run together, each with its own options block, its own seed and its
own output language to keep in step. This node is those two with the wire pulled
out.

What it does **not** buy is time. The captioner runs through ``llama-mtmd-cli``
and the writer through ``llama-completion``: two binaries, two loads of the
weights, even when both halves point at the same file. Folding them into one box
removes the wiring, not the second load.

Two deliberate differences from the pair it replaces:

- **One seed** drives both halves. That is the point of merging, but it does
  mean a graph rebuilt from the two-node version cannot reproduce both of its
  seeds.
- **No references, no rewrite.** If every asset comes back without a
  description, the writer is not run at all: a Ref2VA prompt whose whole premise
  is its references would otherwise be composed from none, after a minute of
  generation nobody asked for.

The caption half is the other node's, reused: the input groups, the label rules
and the encoder checks all come from ``multi_caption`` so the two cannot drift.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

from . import clip_caption, guide_prompt, media, mtmd_engine
from .constants import (
    DEFAULT_OUTPUT_LANGUAGE,
    DURATION_MAX,
    DURATION_MIN,
    OUTPUT_LANGUAGE_NAMES,
    REF_OUTPUT_FIELDS,
    RESOLUTIONS,
)
from .fields import split_sections
from .multi_caption import (
    GROUP_TOOLTIPS,
    GROUPS,
    MASK_TOOLTIP,
    _check_encoders,
    _collect,
    _disabled,
    _is_frames,
    _template_input,
)
from .nodes import (
    BYPASS_TOOLTIP,
    CAPTION_LANGUAGE_TOOLTIP,
    CAPTION_LENGTHS,
    CATEGORY,
    DEFAULT_OPTIONS,
    EMPTY_CAPTION_HINT,
    INSTRUCTIONS_TOOLTIP,
    OPTIONS_TYPE,
    OUTPUT_LANGUAGE_TOOLTIP,
    _ensure_pair,
    _guided_text,
    _report,
    _resolve_captioner_choice,
    caption_question,
    captioner_choices,
    cpu_settings,
    next_index,
    slot_instructions,
    writer_choices,
)
from .progress import NodeProgress

log = logging.getLogger(__name__)

DESCRIPTION = (
    "Describes every connected reference asset and writes the finished MiniMax-H3 "
    "full-reference (Ref2VA) prompt from them, in one node. The group an asset is plugged into "
    "decides its label, so the four labels the guide allows are the only four that can come "
    "out, and the reference block never has to travel down a wire. Two models are still two "
    "loads -- the captioner runs through llama-mtmd-cli and the writer through "
    "llama-completion -- so this saves wiring, not time. The block it built is an output too, "
    "for when you want to see what the writer was given."
)

SEED_TOOLTIP = (
    "Drives both halves: the captions and the rewrite. The two nodes this replaces had one "
    "seed each, so a graph rebuilt from that version reproduces only one of them."
)


class MiniMaxH3ReferenceWriterRef(io.ComfyNode):
    """Caption a set of reference assets and write the Ref2VA prompt from them."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceWriterRef",
            display_name="MiniMax-H3 Reference Writer (Ref2VA)",
            category=CATEGORY,
            description=DESCRIPTION,
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="What the target video should show, and how it uses the references.",
                ),
                *(
                    io.Autogrow.Input(
                        group.id,
                        optional=True,
                        tooltip=GROUP_TOOLTIPS[group.id],
                        template=io.Autogrow.TemplatePrefix(
                            input=_template_input(group),
                            prefix=group.prefix,
                            min=0,
                            max=group.maximum,
                        ),
                    )
                    for group in GROUPS
                ),
                io.String.Input(
                    "previous",
                    optional=True,
                    force_input=True,
                    tooltip=(
                        "reference_assets written earlier -- by a caption node, or by hand. The "
                        "assets here are appended to it and numbered on from it."
                    ),
                ),
                io.Clip.Input(
                    "clip",
                    optional=True,
                    tooltip=(
                        "A multimodal text encoder loaded by 'CLIPLoader' -- Qwen3-VL or "
                        "Gemma-4. Connect it and every asset here is described by that model "
                        "instead of by 'captioner': it stays loaded between assets and between "
                        "runs. Only Gemma-4 E2B, E4B and 12B can hear audio. Leave it "
                        "unconnected and nothing changes."
                    ),
                ),
                io.Custom(OPTIONS_TYPE).Input("options", optional=True),
                io.String.Input(
                    "enabled_mask",
                    default="{}",
                    optional=True,
                    tooltip=MASK_TOOLTIP,
                ),
                io.Combo.Input(
                    "captioner",
                    options=captioner_choices(),
                    tooltip=(
                        "A multimodal GGUF and its projector, for reading the assets. One model "
                        "reads every asset here, so it has to cover every kind you connected -- "
                        "a vision-only captioner cannot take the audio group. Ignored entirely "
                        "while 'clip' is connected."
                    ),
                ),
                io.Combo.Input(
                    "writer",
                    options=writer_choices(),
                    tooltip=(
                        "The language model that writes the six sections. Pick the same GGUF as "
                        "the captioner if it follows instructions well enough -- it is still a "
                        "second load either way. The full-reference guide is the longer of the "
                        "two, so a 4B will hold the format but a 9B keeps the labels consistent "
                        "across all six sections."
                    ),
                ),
                io.Combo.Input(
                    "length",
                    options=list(CAPTION_LENGTHS),
                    default="standard",
                    tooltip="How much the captioner is asked to write, for every asset here.",
                ),
                io.Combo.Input(
                    "resolution",
                    options=list(RESOLUTIONS),
                    default="16:9",
                    tooltip="Target aspect ratio the rewrite is composed for.",
                ),
                io.Int.Input(
                    "duration",
                    default=10,
                    min=DURATION_MIN,
                    max=DURATION_MAX,
                    step=1,
                    tooltip="Target clip length in seconds; drives shot count and pacing.",
                ),
                io.Boolean.Input(
                    "greedy",
                    default=True,
                    tooltip=(
                        "Deterministic decoding for the rewrite. Keep it on unless the result is "
                        "too plain. The captions are always greedy."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFF,
                    control_after_generate=True,
                    tooltip=SEED_TOOLTIP,
                ),
                io.Boolean.Input(
                    "keep_model_loaded",
                    default=False,
                    optional=True,
                    tooltip="Keep the writer in VRAM after the rewrite.",
                ),
                io.Int.Input(
                    "max_frames",
                    default=media.DEFAULT_MAX_FRAMES,
                    min=1,
                    max=64,
                    optional=True,
                    tooltip=(
                        "How many frames to take from an IMAGE batch or a video, spread evenly. "
                        "All of them would overflow the context and the wall clock."
                    ),
                ),
                io.Int.Input(
                    "context_size",
                    default=mtmd_engine.CONTEXT_FROM_MODEL,
                    min=0,
                    max=131072,
                    step=1024,
                    optional=True,
                    tooltip=(
                        "The captioner's context. 0 uses the model's own, which is what its "
                        "projector was sized against. Set a number only to cut the KV cache on a "
                        "small card. The writer's context is 'n_ctx' in the options node."
                    ),
                ),
                io.Boolean.Input(
                    "bypass",
                    default=False,
                    optional=True,
                    tooltip=BYPASS_TOOLTIP,
                ),
                io.Combo.Input(
                    "caption_language",
                    options=list(OUTPUT_LANGUAGE_NAMES),
                    default=DEFAULT_OUTPUT_LANGUAGE,
                    optional=True,
                    tooltip=CAPTION_LANGUAGE_TOOLTIP,
                ),
                io.Combo.Input(
                    "output_language",
                    options=list(OUTPUT_LANGUAGE_NAMES),
                    default=DEFAULT_OUTPUT_LANGUAGE,
                    optional=True,
                    tooltip=OUTPUT_LANGUAGE_TOOLTIP,
                ),
                # Last, like every widget added after a node shipped: ComfyUI
                # restores saved values by position. It is the strip's band that
                # writes here, the same one the caption node draws.
                io.String.Input(
                    "reference_instructions",
                    default="{}",
                    optional=True,
                    tooltip=INSTRUCTIONS_TOOLTIP,
                ),
            ],
            outputs=[
                io.String.Output(display_name="rewritten_prompt"),
                *(io.String.Output(display_name=name) for name in REF_OUTPUT_FIELDS),
                io.String.Output(display_name="reference_assets"),
                io.String.Output(display_name="captions"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        prompt,
        captioner,
        writer,
        length,
        resolution,
        duration,
        greedy,
        seed,
        subjects=None,
        pictures=None,
        videos=None,
        audios=None,
        previous="",
        clip=None,
        options=None,
        enabled_mask="{}",
        keep_model_loaded=False,
        max_frames=media.DEFAULT_MAX_FRAMES,
        context_size=mtmd_engine.CONTEXT_FROM_MODEL,
        bypass=False,
        caption_language=DEFAULT_OUTPUT_LANGUAGE,
        output_language=DEFAULT_OUTPUT_LANGUAGE,
        reference_instructions="{}",
    ) -> io.NodeOutput:
        progress = NodeProgress(cls.hidden.unique_id)
        block = (previous or "").strip()

        if bypass:
            progress.finish("bypassed")
            return io.NodeOutput((prompt or "").strip(), *("",) * len(REF_OUTPUT_FIELDS), block, "")

        supplied = {
            "subjects": subjects,
            "pictures": pictures,
            "videos": videos,
            "audios": audios,
        }
        assets, skipped = _collect(supplied, _disabled(enabled_mask))

        if not assets and not block:
            raise ValueError(
                "Nothing to describe, and nothing on 'previous'. Connect a reference to one of "
                "the four groups: an image to 'subjects' if it defines what something looks "
                "like, to 'pictures' if it is an actual frame of the target video, a clip to "
                "'videos', a sound to 'audios'."
            )

        settings = dict(DEFAULT_OPTIONS)
        if options:
            settings.update(options)

        kinds = {asset.kind for asset in assets}
        model_path = mmproj_path = None
        if assets and clip is None:
            choice = _resolve_captioner_choice(captioner)
            if choice.local:
                model_path, mmproj_path = choice.reference, choice.mmproj
            else:
                model_path, mmproj_path = _ensure_pair(
                    choice.reference, choice.file, choice.mmproj, "Captioner",
                    settings["auto_download"], progress,
                )
            _check_encoders(mmproj_path, kinds)
        elif assets:
            clip_caption.check(clip, kinds)

        asked_for = slot_instructions(reference_instructions)

        # One step per asset plus one for the rewrite, so the bar does not sit
        # at "done" through the longest part of the run.
        progress.set_total(len(assets) + 1)
        captions = []
        empty = 0
        for done, asset in enumerate(assets):
            index = next_index(block, asset.role)
            frames = asset.kind == "video" and _is_frames(asset.value)
            progress.update(
                done,
                f"{asset.role} {index}: reading {asset.slot} ({done + 1} of {len(assets)})",
            )

            asked = caption_question(
                asset.role, length, asked_for.get(asset.slot), caption_language
            )
            as_image = asset.value if asset.kind == "image" or frames else None
            as_audio = asset.value if asset.kind == "audio" else None
            as_video = None if frames else (asset.value if asset.kind == "video" else None)

            if clip is not None:
                caption = clip_caption.describe(
                    clip,
                    instruction=asked,
                    image=as_image,
                    audio=as_audio,
                    video=as_video,
                    max_frames=int(max_frames),
                    seed=int(seed),
                    settings=settings,
                    progress=progress,
                )
            else:
                caption = mtmd_engine.describe(
                    model_path=model_path,
                    mmproj_path=mmproj_path,
                    instruction=asked,
                    image=as_image,
                    audio=as_audio,
                    video=as_video,
                    max_frames=int(max_frames),
                    gpu_layers=int(settings["gpu_layers"]),
                    n_ctx=int(context_size),
                    seed=int(seed),
                    greedy=True,
                    max_new_tokens=min(int(settings["max_new_tokens"]), 1024),
                    temperature=float(settings["temperature"]),
                    top_p=float(settings["top_p"]),
                    top_k=int(settings["top_k"]),
                    device=settings["device"],
                    backend=settings["llama_backend"],
                    auto_download=settings["auto_download"],
                    **cpu_settings(settings),
                    progress=progress,
                )

            caption = " ".join(caption.split())
            if not caption:
                # Same rule as the caption-only node: the asset is lost, not the
                # run. The label goes with it -- and here it matters more, since
                # an empty 'Picture 2:' would go straight into the guide in this
                # same execution rather than sitting in a wire to be noticed.
                empty += 1
                log.warning("[minimax_h3_rewriter.ref_writer] %s: %s", asset.slot, EMPTY_CAPTION_HINT)
                continue
            captions.append(caption)
            block = f"{block}\n{asset.role} {index}: {caption}".strip()

        if not block:
            raise RuntimeError(
                f"No reference survived the captioner, so there is nothing for the Ref2VA guide "
                f"to write from and the writer was not run.\n\n{EMPTY_CAPTION_HINT}"
            )

        note = f"{len(assets) - empty} described"
        if empty:
            note += f", {empty} wrote only reasoning"
        if skipped:
            note += f", {skipped} switched off"
        progress.update(len(assets), f"{note} · writing\n{block}")

        text = _guided_text(
            guide_prompt.REF_MODE, writer, prompt, resolution, duration, block,
            greedy, int(seed), keep_model_loaded, settings, progress, output_language,
        )
        _head, sections = split_sections(text, REF_OUTPUT_FIELDS, fallback="detailed_description")
        _report(progress, text, sections, REF_OUTPUT_FIELDS)
        return io.NodeOutput(
            text,
            *(sections[name] for name in REF_OUTPUT_FIELDS),
            block,
            "\n".join(captions),
        )


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ReferenceWriterRef": MiniMaxH3ReferenceWriterRef,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ReferenceWriterRef": "MiniMax-H3 Reference Writer (Ref2VA)",
}
