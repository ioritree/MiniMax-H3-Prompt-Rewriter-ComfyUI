"""Which GGUFs the rewriter LoRAs are offered to, and what is said about the rest.

Two things went wrong with the shape check alone (issue #18):

- An MTP build appends its draft head as an extra block and counts it in
  ``block_count``, so a real Qwen3.6-27B with the head read as 65 blocks and
  was refused as the wrong size. The adapter never touches that block, and
  llama.cpp does not load it unless asked to draft.
- Qwen3.8-27B has the same shape as Qwen3.6-27B, so the check let it through
  in silence. The LoRA attaches to it, but was never trained on it.

Nothing here needs ComfyUI or a real model: each GGUF is a header and padding.
"""

import importlib
import pathlib
import struct
import sys
import types

_PKG = "minimax_h3_rewriter"
ROOT = pathlib.Path(__file__).resolve().parent.parent

if _PKG not in sys.modules:
    _package = types.ModuleType(_PKG)
    _package.__path__ = [str(ROOT / _PKG)]
    sys.modules[_PKG] = _package

discovery = importlib.import_module(f"{_PKG}.discovery")
sections = importlib.import_module(f"{_PKG}.model_sections")

UINT32, STRING = 4, 8

TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"


def gguf(path: pathlib.Path, blocks: int = 64, draft: int | None = None, **names) -> pathlib.Path:
    """A qwen35 header of width 5120 with the given block count and names."""
    values = {
        "general.architecture": "qwen35",
        "qwen35.block_count": blocks,
        "qwen35.embedding_length": 5120,
        "tokenizer.chat_template": TEMPLATE,
    }
    if draft is not None:
        values["qwen35.nextn_predict_layers"] = draft
    parents = names.pop("base_models", ())
    for key, value in names.items():
        values[f"general.{key}"] = value
    if parents:
        values["general.base_model.count"] = len(parents)
        for index, (name, url) in enumerate(parents):
            values[f"general.base_model.{index}.name"] = name
            values[f"general.base_model.{index}.repo_url"] = url

    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(values)))
    for key, value in values.items():
        encoded = key.encode()
        out += struct.pack("<Q", len(encoded)) + encoded
        if isinstance(value, str):
            text = value.encode()
            out += struct.pack("<IQ", STRING, len(text)) + text
        else:
            out += struct.pack("<II", UINT32, value)
    out += b"\0" * 64
    path.write_bytes(bytes(out))
    return path


QWEN36 = {
    "name": "Qwen3.6-27B",
    "basename": "Qwen3.6-27B",
    "base_models": (("Qwen3.6 27B", "https://huggingface.co/Qwen/Qwen3.6-27B"),),
}

QWEN38 = {
    "name": "Huihui Qwen3.8 27B Abliterated",
    "basename": "Huihui-Qwen3.8",
    "size_label": "27B",
    "base_models": (("Qwen3.8 27B", "https://huggingface.co/Qwen/Qwen3.8-27B"),),
}


def test_the_draft_head_is_not_counted_as_a_block(tmp_path):
    path = gguf(tmp_path / "mtp.gguf", blocks=65, draft=1, **QWEN36)
    header = discovery.gguf_header(str(path))
    assert (header["blocks"], header["draft_blocks"]) == (64, 1)
    assert discovery.gguf_problem(str(path)) == ""


def test_a_65th_block_that_is_not_a_draft_head_is_still_the_wrong_size(tmp_path):
    path = gguf(tmp_path / "deeper.gguf", blocks=65, **QWEN36)
    assert "65 blocks" in discovery.gguf_problem(str(path))


def test_a_draft_head_does_not_rescue_the_wrong_trunk(tmp_path):
    path = gguf(tmp_path / "small.gguf", blocks=33, draft=1)
    assert "32 blocks" in discovery.gguf_problem(str(path))


def test_a_draft_count_as_large_as_the_model_is_ignored(tmp_path):
    path = gguf(tmp_path / "odd.gguf", blocks=64, draft=64)
    header = discovery.gguf_header(str(path))
    assert (header["blocks"], header["draft_blocks"]) == (64, 0)


def test_the_base_itself_gets_no_warning(tmp_path):
    path = gguf(tmp_path / "base.gguf", **QWEN36)
    assert discovery.gguf_base_note(str(path)) == ""


def test_the_converter_spelling_of_the_base_is_the_base(tmp_path):
    path = gguf(tmp_path / "converted.gguf", name="Qwen3.6 27B", basename="Qwen3.6", size_label="27B")
    assert discovery.gguf_base_note(str(path)) == ""


def test_a_fine_tune_that_names_the_base_gets_no_warning(tmp_path):
    path = gguf(tmp_path / "tuned.gguf", name="Huihui Qwen3.6 27B Abliterated")
    assert discovery.gguf_base_note(str(path)) == ""


def test_another_release_of_the_same_shape_is_warned_about(tmp_path):
    path = gguf(tmp_path / "later.gguf", blocks=65, draft=1, **QWEN38)
    assert discovery.gguf_problem(str(path)) == ""
    note = discovery.gguf_base_note(str(path))
    assert "Huihui Qwen3.8 27B Abliterated" in note
    assert "Qwen3.6-27B" in note


def test_a_header_that_names_nothing_gets_no_warning(tmp_path):
    path = gguf(tmp_path / "anonymous.gguf")
    assert discovery.gguf_base_note(str(path)) == ""


def test_the_warning_follows_the_rewriter_it_is_asked_for(tmp_path):
    path = gguf(tmp_path / "base.gguf", **QWEN36)
    assert discovery.gguf_base_note(str(path), discovery.BASE_NAME_OMNI) != ""


def check(tmp_path, file: str) -> dict:
    entry = {"name": "A", "format": "gguf", "repo": str(tmp_path), "file": file}
    return sections.check("models", entry)


def test_check_it_passes_an_mtp_build_of_the_base_and_says_what_the_head_is(tmp_path):
    gguf(tmp_path / "mtp.gguf", blocks=65, draft=1, **QWEN36)
    found = check(tmp_path, "mtp.gguf")
    assert found["verdict"] == "good"
    assert any("64 blocks" in line["text"] and "draft head" in line["text"] for line in found["lines"])


def test_check_it_warns_about_a_later_release(tmp_path):
    gguf(tmp_path / "later.gguf", blocks=65, draft=1, **QWEN38)
    found = check(tmp_path, "later.gguf")
    assert found["verdict"] == "warn"
    assert any(line["level"] == "warn" and "Qwen3.6-27B" in line["text"] for line in found["lines"])
