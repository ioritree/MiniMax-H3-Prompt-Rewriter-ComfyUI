"""Describing a reference asset with ``llama-mtmd-cli``.

The writer nodes read text, not pixels. This is where the text comes from: an
image, an audio clip or a video goes to a multimodal model in a subprocess and
comes back as one or two sentences, which then become a line of
``reference_assets``.

It reuses the runtime the rewriter already fetches -- ``llama-mtmd-cli`` ships in
the same release archive as ``llama-completion``, so a machine that has run one
rewrite already has everything this needs.

Two differences from ``cli_engine`` are worth knowing:

- **The chat template is applied by the binary**, not here. ``-p`` is a plain
  instruction and mtmd wraps it, which is also what lets it splice the media
  tokens into the right place in the turn. Rendering the template ourselves, as
  the rewriter does to force ``enable_thinking=False``, would put the image
  markers in the wrong position.
- **The prompt goes on the command line**, not through ``--file``. There is no
  shell in the way -- the command is a list -- so quotes and newlines in an
  instruction need no escaping, and ``-f`` is not honoured alongside media.

Which models actually work here is a much shorter list than the set of
multimodal GGUFs on the Hub: the projector format has to be one ``mtmd``
understands. See ``models.json`` for the ones this pack has been run against.
"""

from __future__ import annotations

import logging
import os

from . import devices, llamacpp, media, runner
from .constants import normalize_seed
from .fields import answer_only
from .progress import NodeProgress

log = logging.getLogger(__name__)

PREVIEW_TAIL = 280
CHARS_PER_TOKEN = 4.0

ALL_LAYERS = 999

DEFAULT_MAX_TOKENS = 256

FLAG_FOR_KIND = {"image": "--image", "audio": "--audio"}

CONTEXT_FROM_MODEL = 0


def build_command(
    binary: str,
    model_path: str,
    mmproj_path: str,
    instruction: str,
    attachments: list[tuple[str, str]],
    gpu_layers: int,
    n_ctx: int = CONTEXT_FROM_MODEL,
    seed: int = 42,
    greedy: bool = True,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    device: str = devices.AUTO,
    adapter_path: str | None = None,
    system_prompt: str = "",
) -> list[str]:
    layers = devices.layers_for(device, gpu_layers)
    # Keep asking for every layer rather than letting llama.cpp fit the run to
    # the free VRAM itself: left to it, a big model keeps its full context and
    # spills half the weights to the CPU, which measures 20x slower than the
    # same model held whole with a small context. Trim context_size instead.
    layers = ALL_LAYERS if layers < 0 else layers
    command = [
        binary,
        "--model", model_path,
        "--mmproj", mmproj_path,
        *devices.llama_arguments(device),
        "--n-gpu-layers", str(layers),
        "--ctx-size", str(int(n_ctx)),
        "--predict", str(int(max_new_tokens)),
        "--seed", str(normalize_seed(seed)),
    ]
    if adapter_path:
        command += ["--lora", adapter_path]
    for kind, path in attachments:
        flag = FLAG_FOR_KIND.get(kind)
        if flag is None:
            raise ValueError(f"unknown attachment kind '{kind}'")
        command += [flag, path]
    if greedy:
        command += ["--temp", "0"]
    else:
        command += [
            "--temp", f"{float(temperature):g}",
            "--top-p", f"{float(top_p):g}",
            "--top-k", str(int(top_k)),
        ]
    if system_prompt:
        command += ["--system-prompt", system_prompt]
    # Last, so the instruction is never mistaken for a value of another flag.
    command += ["--prompt", instruction]
    return command


FALLBACK_CONTEXT = 8192


def failure_hint(stderr_text: str, n_ctx: int) -> str:
    """The advice that fits the way the child actually died."""
    if runner.out_of_memory(stderr_text):
        if int(n_ctx) == CONTEXT_FROM_MODEL:
            return (
                f"The card ran out of room. context_size is 0, which means the model's own "
                f"context, and on a large model that KV cache alone is gigabytes on top of the "
                f"weights. Set context_size on this node to {FALLBACK_CONTEXT} and run again; "
                f"lower gpu_layers in the options node only if that is still not enough."
            )
        return (
            f"The card ran out of room at context_size {int(n_ctx)}. Halve it, or lower "
            f"gpu_layers in the options node to keep part of the model on the CPU."
        )
    return (
        "Not every multimodal GGUF works here: llama.cpp's mtmd has to understand the "
        "projector format, and several current models abort while loading it. Pick an entry "
        "from the captioner list, which names the ones this pack has been run against."
    )


def clip_note(count: int, seconds: float) -> str:
    """Tell the model that these separate images are one clip, in order.

    Without it the frames read as an unrelated set and the answer comes back as
    "several photographs of ..." rather than a description of what happens.
    """
    span = f" spanning {seconds:.1f} seconds" if seconds > 0 else ""
    return (
        f"The {count} images are frames sampled evenly from a single video clip{span}, "
        f"in chronological order. Describe the clip, not the individual frames."
    )


def attachments_from(
    workspace: media.Workspace,
    image=None,
    audio=None,
    video=None,
    max_frames: int = media.DEFAULT_MAX_FRAMES,
) -> tuple[list[tuple[str, str]], list[str], str]:
    """Write the connected inputs to disk. Returns ``(attachments, notes, note)``."""
    attachments: list[tuple[str, str]] = []
    notes: list[str] = []
    note = ""

    if video is not None:
        paths, total, seconds = media.video_frames(video, workspace, max_frames)
        attachments.extend(("image", path) for path in paths)
        notes.append(f"{len(paths)} of {total} video frames" if total > len(paths)
                     else f"video, {len(paths)} frames")
        if len(paths) > 1:
            note = clip_note(len(paths), seconds)

    if image is not None:
        paths = media.image_files(image, workspace, max_frames)
        attachments.extend(("image", path) for path in paths)
        total = int(getattr(image, "shape", [len(paths)])[0])
        notes.append(
            f"{len(paths)} of {total} frames" if total > len(paths) else f"{len(paths)} image(s)"
        )

    if audio is not None:
        attachments.append(("audio", media.audio_file(audio, workspace)))
        notes.append("audio")

    return attachments, notes, note


def describe(
    model_path: str,
    mmproj_path: str,
    instruction: str,
    image=None,
    audio=None,
    video=None,
    attachments: list[tuple[str, str]] | None = None,
    max_frames: int = media.DEFAULT_MAX_FRAMES,
    gpu_layers: int = -1,
    n_ctx: int = CONTEXT_FROM_MODEL,
    seed: int = 42,
    greedy: bool = True,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    device: str = devices.AUTO,
    backend: str = "auto",
    auto_download: bool = True,
    adapter_path: str | None = None,
    system_prompt: str = "",
    progress: NodeProgress | None = None,
) -> str:
    """Fetch the runtime if needed, describe the attachments once, and return the text.

    ``adapter_path`` is here for the 8B rewriter rather than for captioning.
    That model reads the reference frames itself, so its LoRA has to be attached
    to the same process that holds the images -- there is no separate text pass
    to hang it on. A captioner passes nothing here and nothing changes for it.

    ``attachments`` is for the same caller, and for the same reason. Two
    reference frames are two ordered pictures with a role each, not a batch: the
    model is told which is the first frame and which is the last, and the two
    can be different sizes. So that caller writes its own files and states the
    order, rather than handing over one IMAGE for this to take apart.
    """
    device = devices.validate(device)
    if attachments is None and image is None and audio is None and video is None:
        raise ValueError(
            "nothing to describe: connect an image, an audio clip or a video, or type the "
            "description in by hand."
        )
    if attachments is not None and not attachments:
        raise ValueError("nothing to describe: 'attachments' is empty.")

    binary = llamacpp.ensure_mtmd(backend, auto_download, progress)

    with media.Workspace() as workspace:
        note = ""
        if attachments is None:
            attachments, notes, note = attachments_from(
                workspace, image, audio, video, max_frames
            )
        else:
            kinds = dict.fromkeys(kind for kind, _path in attachments)
            notes = [
                f"{sum(kind == wanted for kind, _path in attachments)} {wanted}(s)"
                for wanted in kinds
            ]
        if note:
            instruction = f"{note}\n\n{instruction}"
        command = build_command(
            binary, model_path, mmproj_path, instruction, attachments,
            gpu_layers, n_ctx, seed, greedy, max_new_tokens, temperature, top_p, top_k, device,
            adapter_path, system_prompt,
        )

        runner.free_comfy_vram(device)
        if progress is not None:
            where = "" if device == devices.AUTO else f" on {device}"
            lora = f" + {os.path.basename(adapter_path)}" if adapter_path else ""
            progress.set_total(max(int(max_new_tokens), 1))
            progress.text(
                f"Describing {' + '.join(notes)}\n{os.path.basename(model_path)}{lora}{where}",
                force=True,
            )

        def report(whole: str) -> None:
            if progress is not None:
                progress.update(
                    min(len(whole) / CHARS_PER_TOKEN, float(max_new_tokens)),
                    f"Describing · {len(whole)} chars\n{whole[-PREVIEW_TAIL:]}",
                )

        try:
            text, stderr_text = runner.run(command, binary, report)
        except runner.ChildFailed as error:
            raise RuntimeError(f"{error}\n\n{failure_hint(str(error), n_ctx)}") from error

    if progress is not None:
        progress.finish(f"Done · {len(text)} chars{runner.speed(stderr_text)}")

    # The reasoning is cut here rather than at the caller, so both caption nodes
    # get it, and after the progress line so the raw length still shows how much
    # of the budget the model spent thinking. mtmd-cli applies the chat template
    # itself and b10310 has no ``--chat-template-kwargs``, so there is no
    # ``enable_thinking=False`` to forward on this route -- a thinking build
    # writes the block whatever the instruction says.
    answer = answer_only(text.replace("\r\n", "\n"))
    if text and not answer:
        log.warning(
            "[minimax_h3_rewriter.mtmd_engine] %s spent the whole %d-token budget on an "
            "unfinished <think> block and never wrote the answer. Raise max_new_tokens in the "
            "options node, or pick a captioner that does not reason.",
            os.path.basename(model_path),
            int(max_new_tokens),
        )
    return answer
