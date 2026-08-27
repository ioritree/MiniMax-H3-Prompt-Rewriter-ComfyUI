"""Choosing which GPU the language model runs on.

Every node in this pack has carried the same warning: turn ``keep_model_loaded``
off, because the card is needed for video generation the moment the rewrite
finishes. That warning exists only because both models want the same device. On
a machine with two, they need not: the rewriter can live on the second card and
stay resident, and ComfyUI's diffusion model on the first never has to move.

So this is not only a placement knob. It is what makes ``keep_model_loaded``
worth switching on, and it is why :func:`shares_comfy_device` exists -- running
elsewhere means ComfyUI's own models must *not* be evicted first, which is what
every backend here does unconditionally today.

One spelling, three backends:

===========  =================================================================
``auto``     whatever each backend would have picked on its own
``cuda:N``   ``--device CUDA<N>`` for the binaries, ``main_gpu`` for the wheel,
             ``device_map={"": "cuda:N"}`` for Transformers
``cpu``      no offload at all
===========  =================================================================

The values are deliberately plain. A label carrying the card's model name would
read better and would break every saved workflow the day the card is replaced,
the same way a renamed model entry does.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

AUTO = "auto"
CPU = "cpu"
PREFIX = "cuda:"


def _torch():
    import torch

    return torch


def count() -> int:
    """How many CUDA devices this process can see.

    Note *this process*: ComfyUI started with ``--cuda-device 1`` sets
    ``CUDA_VISIBLE_DEVICES``, so there is exactly one visible device and it is
    numbered 0. The subprocess backends inherit that environment, so every
    spelling below stays consistent with what ComfyUI itself is using.
    """
    try:
        torch = _torch()
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        log.debug("[minimax_h3_rewriter.devices.count] no CUDA visible", exc_info=True)
        return 0


def describe(index: int) -> str:
    """``NVIDIA GeForce RTX 5090, 32.0 GB``, or "" when it cannot be read."""
    try:
        torch = _torch()
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory / 1024 ** 3
        return f"{name}, {total:.1f} GB"
    except Exception:
        log.debug("[minimax_h3_rewriter.devices.describe] %s unreadable", index, exc_info=True)
        return ""


def vram_bytes(spec: str = AUTO) -> int:
    """Total memory of the card a run is bound for, or 0 when it cannot be read.

    Total rather than free, deliberately. What is free at the moment of asking
    is mostly a statement about the diffusion model that is about to be evicted
    anyway, so it would size a context differently depending on what happened to
    be loaded when the node ran -- and the same graph would then fail only on
    the second pass. The headroom that covers the difference is the caller's.
    """
    if is_cpu(spec):
        return 0
    ordinal = index(spec) or 0
    try:
        torch = _torch()
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.get_device_properties(ordinal).total_memory)
    except Exception:
        log.debug("[minimax_h3_rewriter.devices.vram_bytes] %s unreadable", spec, exc_info=True)
        return 0


def choices() -> list[str]:
    """The values offered by the options node, in a stable spelling."""
    return [AUTO] + [f"{PREFIX}{index}" for index in range(count())] + [CPU]


def tooltip() -> str:
    lines = [
        "Which device the language model runs on. 'auto' behaves as before.",
        "",
        "Pick a second card and two things change: the model no longer competes "
        "with ComfyUI's own for VRAM, and 'keep_model_loaded' becomes worth "
        "turning on, because nothing has to be evicted to make room.",
    ]
    found = [f"  {PREFIX}{index} — {describe(index) or 'unreadable'}" for index in range(count())]
    if found:
        lines += ["", "Visible here:"] + found
    else:
        lines += ["", "No CUDA device is visible to ComfyUI, so only 'cpu' will do anything."]
    return "\n".join(lines)


def index(spec: str) -> int | None:
    """The CUDA ordinal of a device spec, or ``None`` for ``auto``/``cpu``."""
    spec = (spec or AUTO).strip().lower()
    if not spec.startswith(PREFIX):
        return None
    try:
        return int(spec[len(PREFIX):])
    except ValueError:
        return None


def is_cpu(spec: str) -> bool:
    return (spec or AUTO).strip().lower() == CPU


def validate(spec: str) -> str:
    """Return the spec, refusing one this machine cannot honour.

    Refused rather than quietly demoted to ``auto``: a workflow moved from a
    two-card machine asks for a card that is not there, and silently running on
    the wrong one is how somebody's video model gets evicted mid-batch.
    """
    spec = (spec or AUTO).strip() or AUTO
    ordinal = index(spec)
    if ordinal is None:
        if spec.lower() in (AUTO, CPU):
            return spec.lower()
        raise ValueError(
            f"'{spec}' is not a device. Use '{AUTO}', '{CPU}', or '{PREFIX}N' for a CUDA card."
        )
    visible = count()
    if ordinal >= visible:
        raise RuntimeError(
            f"This workflow asks for '{spec}', but ComfyUI can see "
            f"{visible if visible else 'no'} CUDA device{'' if visible == 1 else 's'} "
            f"({', '.join(f'{PREFIX}{i}' for i in range(visible)) or 'none'}). Pick one that "
            f"exists, or '{AUTO}'."
        )
    return spec.lower()


def shares_comfy_device(spec: str) -> bool:
    """Would this run compete with ComfyUI's own models for VRAM?

    ``True`` for ``auto`` — the safe assumption, and what every backend did
    before this existed. ``False`` only when the answer is definitely no, so an
    unreadable ComfyUI device leaves the eviction in place rather than skipping
    it on a guess.
    """
    ordinal = index(spec)
    if ordinal is None:
        # 'cpu' touches no VRAM at all; 'auto' lands wherever ComfyUI already is.
        return not is_cpu(spec)

    try:
        import comfy.model_management as mm

        current = mm.get_torch_device()
    except Exception:
        log.debug("[minimax_h3_rewriter.devices.shares_comfy_device] no comfy device", exc_info=True)
        return True

    if getattr(current, "type", None) != "cuda":
        return False
    return (current.index or 0) == ordinal


def llama_arguments(spec: str) -> list[str]:
    """Extra command-line arguments for a llama.cpp executable."""
    if is_cpu(spec):
        return ["--device", "none"]
    ordinal = index(spec)
    return ["--device", f"CUDA{ordinal}"] if ordinal is not None else []


#: KV cache types llama.cpp will accept, plus the entry that passes nothing.
#: "default" first so the widget opens on the behaviour every earlier version had.
KV_CACHE_TYPES = ("default", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "f16", "f32")

#: The three states of -fa, "default" being "do not pass the flag" rather than
#: llama.cpp's own "auto" -- they behave the same today and this one keeps saying
#: so if that changes.
FLASH_ATTENTION = ("default", "on", "off")


def memory_arguments(cache_type: str = "default", flash_attn: str = "default") -> list[str]:
    """``-ctk``/``-ctv`` and ``-fa``, or nothing at all.

    Both are for the run that does not quite fit. The KV cache is charged per
    token per layer, so at q8_0 it is half the bytes for a quality difference
    that does not show up in a rewrite; flash attention shrinks the compute
    buffer beside it. Neither touches the weights, which is the point -- on a
    card doing partial offload the freed VRAM becomes another layer or two that
    does not have to run on the CPU.

    "default" passes nothing, so a run that leaves these alone sends the command
    line it always sent.
    """
    arguments: list[str] = []
    if cache_type and cache_type != "default":
        arguments += ["--cache-type-k", cache_type, "--cache-type-v", cache_type]
    if flash_attn and flash_attn != "default":
        arguments += ["--flash-attn", flash_attn]
    return arguments


def cpu_arguments(threads: int = -1, poll: int = -1) -> list[str]:
    """``--threads`` and ``--poll``, or nothing at all.

    -1 means "do not pass the flag", so a run that never touches these sends the
    command line it always sent and llama.cpp picks as it always did: one thread
    per core, polling level 50. Both are worth setting the other way once every
    layer is on the GPU -- the threadpool is then waiting rather than working,
    and at the stock polling level it waits by spinning.
    """
    arguments: list[str] = []
    # 0 threads is not a thing llama.cpp accepts, so it reads as "unset" like -1
    # rather than being passed through to be refused.
    if int(threads) >= 1:
        arguments += ["--threads", str(int(threads))]
    if int(poll) >= 0:
        arguments += ["--poll", str(min(max(int(poll), 0), 100))]
    return arguments


def layers_for(spec: str, gpu_layers: int) -> int:
    """The offload count once the device has had its say."""
    return 0 if is_cpu(spec) else int(gpu_layers)


def llama_cpp_kwargs(spec: str) -> dict:
    """Keyword arguments for ``llama_cpp.Llama``."""
    if is_cpu(spec):
        return {"n_gpu_layers": 0}
    ordinal = index(spec)
    if ordinal is None:
        return {}
    kwargs = {"main_gpu": ordinal}
    try:
        import llama_cpp

        kwargs["split_mode"] = llama_cpp.LLAMA_SPLIT_MODE_NONE
    except Exception:
        log.debug("[minimax_h3_rewriter.devices] no LLAMA_SPLIT_MODE_NONE", exc_info=True)
    return kwargs


def device_map(spec: str):
    """``device_map`` for ``from_pretrained``."""
    if is_cpu(spec):
        return {"": "cpu"}
    ordinal = index(spec)
    return {"": f"cuda:{ordinal}"} if ordinal is not None else "auto"
