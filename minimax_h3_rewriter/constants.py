"""Repository coordinates and fixed choices shared by the rewriter nodes."""

from __future__ import annotations

import sys


BASE_MODEL_REPO = "Qwen/Qwen3.6-27B"
ADAPTER_REPO = "lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA"

ADAPTER_DIR_NAME = "MiniMax-H3-Prompt-Rewriter-LoRA"

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
BASE_SKIP_SUFFIXES = (".md", ".gitattributes")

RESOLUTIONS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
DURATION_MIN = 4
DURATION_MAX = 15

QUANTIZATIONS = ("nf4", "int8", "bfloat16", "float16")
ATTN_IMPLEMENTATIONS = ("sdpa", "eager", "flash_attention_2")

RUNTIME_AUTO = "auto"
RUNTIME_WHEEL = "llama-cpp-python"
RUNTIME_BINARY = "llama.cpp"
GGUF_RUNTIMES = (RUNTIME_AUTO, RUNTIME_WHEEL, RUNTIME_BINARY)

OUTPUT_FIELDS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)

REF_OUTPUT_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

#: Widget label -> how the language is named *to the model*, which is asked in
#: English because that is what the surrounding instruction is written in. Both
#: prompt paths read this so the two menus cannot drift apart, and insertion
#: order is menu order.
#:
#: ``ENGLISH_OUTPUT_LANGUAGE`` is the one entry with no directive attached: it is
#: the original behaviour, and both paths special-case it to emit their system
#: prompt untouched rather than adding "write in English" to a guide that is
#: already English.
ENGLISH_OUTPUT_LANGUAGE = "English"

OUTPUT_LANGUAGE_NAMES = {
    "繁體中文": "Traditional Chinese (繁體中文)",
    "简体中文": "Simplified Chinese (简体中文)",
    ENGLISH_OUTPUT_LANGUAGE: "English",
    "日本語": "Japanese (日本語)",
    "한국어": "Korean (한국어)",
}

#: What goes inside the guide's ``<d>[...]</d>`` dialogue tag, which is a *language*
#: tag driving how H3 pronounces the line -- not decoration. Chinese dialogue left
#: tagged ``[English]`` gets read aloud as English, so this is tracked separately
#: from the prose language: both Chinese entries share one tag, and the guide's own
#: examples show the script inside the tag staying whatever the user wrote.
OUTPUT_LANGUAGE_TAGS = {
    "繁體中文": "Chinese",
    "简体中文": "Chinese",
    ENGLISH_OUTPUT_LANGUAGE: "English",
    "日本語": "Japanese",
    "한국어": "Korean",
}

DEFAULT_OUTPUT_LANGUAGE = "繁體中文"


def language_name(language: str) -> str:
    """The model-facing name for a widget label, tolerating an unknown one."""
    return OUTPUT_LANGUAGE_NAMES.get(
        language or "", OUTPUT_LANGUAGE_NAMES[DEFAULT_OUTPUT_LANGUAGE]
    )


def language_tag(language: str) -> str:
    """The ``<d>[...]`` tag for a widget label, tolerating an unknown one."""
    return OUTPUT_LANGUAGE_TAGS.get(
        language or "", OUTPUT_LANGUAGE_TAGS[DEFAULT_OUTPUT_LANGUAGE]
    )


MODELS_SUBDIR = "LLM"

#: transformers' set_seed feeds numpy, whose legacy seeding refuses anything at
#: or above 2**32; llama.cpp takes a uint32 too. ComfyUI's "randomize" happily
#: produces 64-bit values, so every seed is folded into this range before use.
SEED_MODULUS = 2 ** 32


def normalize_seed(seed) -> int:
    return int(seed) % SEED_MODULUS


def install_command(package: str) -> str:
    """The pip line for *this* interpreter, ready to paste into a terminal.

    "pip install X" is not advice a ComfyUI user can follow: the portable build
    runs an embedded Python that is not on PATH and has no pip launcher, a
    manual install has a venv, and the desktop app has its own environment
    again. Typing the bare command installs the package into whichever Python
    the shell happens to find, and the node keeps reporting it as missing. So
    every "package X is missing" message names sys.executable instead.
    """
    executable = sys.executable or "python"
    if " " in executable:
        executable = f'"{executable}"'
    return f"{executable} -m pip install {package}"
