"""Letting go of a kept model: the button, 'repeat_last', and ComfyUI's own unload.

No model is loaded here. The two engines are stood in for by objects that say
what they hold and forget it when told to, which is all ``resident`` asks of
them. What is worth pinning down is *when* a release happens and when it must
not, because both mistakes are quiet: a release in the middle of a run closes a
llama.cpp context under the call reading from it, and a release the pack
triggers on itself unloads the very model it was making room for.

The package is registered by hand, as in test_memory: importing it for real
would run ``__init__.py`` and pull in ComfyUI.
"""

import importlib
import pathlib
import sys
import types

import pytest

_PKG = "minimax_h3_rewriter"
_ROOT = pathlib.Path(__file__).resolve().parent.parent

if _PKG not in sys.modules:
    package = types.ModuleType(_PKG)
    package.__path__ = [str(_ROOT / _PKG)]
    sys.modules[_PKG] = package

resident = importlib.import_module(f"{_PKG}.resident")
memory = importlib.import_module(f"{_PKG}.memory")
library = importlib.import_module(f"{_PKG}.library")
runner = importlib.import_module(f"{_PKG}.runner")
devices = importlib.import_module(f"{_PKG}.devices")
engine = importlib.import_module(f"{_PKG}.engine")
gguf_engine = importlib.import_module(f"{_PKG}.gguf_engine")
routes = importlib.import_module(f"{_PKG}.routes")

WRITER = "MiniMaxH3PromptRewriter"
ANSWER = ("a field at dawn", "a field at dawn", "wind", "N/A")


class Slot:
    """An engine, reduced to what ``resident`` reads: a name, and a way to drop it."""

    def __init__(self, name=""):
        self.name = name

    def holding(self):
        return self.name

    def unload(self):
        self.name = ""


@pytest.fixture
def slots(monkeypatch):
    """A Transformers model being kept, and an empty GGUF slot beside it."""
    pair = (Slot("Qwen3.6-27B"), Slot())
    monkeypatch.setattr(resident, "_engines", lambda: pair)
    yield pair
    resident._RUNS.update(active=0, owed=False)


@pytest.fixture
def comfy(monkeypatch):
    """ComfyUI's model management, as far as the hook is concerned."""
    management = types.ModuleType("comfy.model_management")
    management.calls = []

    def unload_all_models():
        management.calls.append("unload")
        return "stock"

    management.unload_all_models = unload_all_models
    management.soft_empty_cache = lambda force=False: management.calls.append("empty")
    parent = types.ModuleType("comfy")
    parent.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", parent)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    return management


@pytest.fixture
def kept_answer():
    memory.LAST.clear()
    memory.keep("7", WRITER, ANSWER, {"prompt": "a field"})
    yield
    memory.LAST.clear()


def test_a_release_names_what_went(slots):
    assert resident.release() == {"released": ["Qwen3.6-27B"], "waiting": False}
    assert resident.held() == []


def test_both_engines_go(slots):
    slots[1].name = "Qwen3.6-27B-Q4_K_M.gguf"
    assert resident.release()["released"] == ["Qwen3.6-27B", "Qwen3.6-27B-Q4_K_M.gguf"]


def test_nothing_held_is_not_an_error(slots):
    slots[0].name = ""
    assert resident.release() == {"released": [], "waiting": False}


def test_a_release_during_a_run_waits_for_it(slots):
    with resident.in_use():
        assert resident.release() == {"released": [], "waiting": True}
        assert resident.held() == ["Qwen3.6-27B"], "unloaded under the run that was using it"
    assert resident.held() == []


def test_a_run_that_fails_still_pays_what_is_owed(slots):
    with pytest.raises(RuntimeError):
        with resident.in_use():
            resident.release()
            raise RuntimeError("the model ran out of context")
    assert resident.held() == []


def test_the_debt_is_paid_once(slots):
    with resident.in_use():
        resident.release()
    slots[0].name = "Qwen3.6-27B"
    with resident.in_use():
        pass
    assert resident.held() == ["Qwen3.6-27B"], "a later run lost the model it was told to keep"


def test_a_run_inside_a_run_pays_at_the_last_way_out(slots):
    with resident.in_use():
        with resident.in_use():
            resident.release()
        assert resident.held() == ["Qwen3.6-27B"]
    assert resident.held() == []


def test_a_run_nobody_interrupted_keeps_its_model(slots):
    with resident.in_use():
        pass
    assert resident.held() == ["Qwen3.6-27B"]


@pytest.mark.parametrize("switch", [True, None])
def test_settling_leaves_a_kept_model_and_a_node_without_the_switch(slots, switch):
    assert resident.settle(switch) == []
    assert resident.held() == ["Qwen3.6-27B"]


def test_settling_with_the_switch_off_unloads(slots):
    assert resident.settle(False) == ["Qwen3.6-27B"]
    assert resident.held() == []


def test_the_caption_line_is_empty_when_nothing_went():
    assert resident.freed([]) == ""
    assert "Qwen3.6-27B" in resident.freed(["Qwen3.6-27B"])


def test_a_repeated_answer_reads_the_switch(slots, kept_answer):
    given = {"prompt": "a field", "keep_model_loaded": False}
    assert memory.repeat("7", WRITER, True, given) == ANSWER
    assert resident.held() == []


def test_a_repeated_answer_keeps_what_it_was_told_to_keep(slots, kept_answer):
    given = {"prompt": "a field", "keep_model_loaded": True}
    assert memory.repeat("7", WRITER, True, given) == ANSWER
    assert resident.held() == ["Qwen3.6-27B"]


def test_a_captioner_has_no_switch_to_read(slots, kept_answer):
    assert memory.repeat("7", WRITER, True, {"prompt": "a field"}) == ANSWER
    assert resident.held() == ["Qwen3.6-27B"]


def test_a_first_run_under_repeat_is_left_to_the_engine(slots):
    memory.LAST.clear()
    given = {"prompt": "a field", "keep_model_loaded": False}
    assert memory.repeat("7", WRITER, True, given) is None
    assert resident.held() == ["Qwen3.6-27B"], "unloaded just before the run that needs it"


def test_repeat_switched_off_touches_nothing(slots, kept_answer):
    given = {"prompt": "a field", "keep_model_loaded": False}
    assert memory.repeat("7", WRITER, False, given) is None
    assert resident.held() == ["Qwen3.6-27B"]


@pytest.fixture
def shelf(tmp_path, monkeypatch):
    """A prompt set with one record, and the pick a node would carry for it."""
    monkeypatch.setattr(library, "root", lambda: str(tmp_path))
    library.add(
        "global",
        {
            "name": "Dawn",
            "task": "T2VA",
            "node_class": WRITER,
            "text": ANSWER[0],
            "sections": list(ANSWER[1:]),
            "references": [],
        },
    )
    record = library.load("global")["records"][0]["id"]
    return '{"file": "global", "id": "%s"}' % record


def test_a_saved_prompt_handed_on_reads_the_switch(slots, shelf):
    chosen, _text = library.picked(shelf, True, WRITER, len(ANSWER), keep_loaded=False)
    assert chosen == ANSWER
    assert resident.held() == []


def test_a_saved_prompt_for_another_node_reads_it_too(slots, shelf):
    chosen, text = library.picked(shelf, True, "MiniMaxH3GuidedWriter", 7, keep_loaded=False)
    assert chosen is None and text == ANSWER[0]
    assert resident.held() == []


def test_a_saved_prompt_keeps_what_it_was_told_to_keep(slots, shelf):
    library.picked(shelf, True, WRITER, len(ANSWER), keep_loaded=True)
    assert resident.held() == ["Qwen3.6-27B"]


def test_a_pick_that_is_not_in_play_touches_nothing(slots, shelf):
    assert library.picked(shelf, False, WRITER, len(ANSWER), keep_loaded=False) == (None, None)
    assert library.picked("", True, WRITER, len(ANSWER), keep_loaded=False) == (None, None)
    assert resident.held() == ["Qwen3.6-27B"]


def test_the_route_answers_in_the_shape_the_button_reads(slots):
    assert routes._unload() == {"ok": True, "released": ["Qwen3.6-27B"], "waiting": False}


def test_comfyui_unloading_takes_ours_with_it(slots, comfy):
    assert resident.install()
    assert comfy.unload_all_models() == "stock"
    assert comfy.calls == ["unload"]
    assert resident.held() == []


def test_it_is_hooked_once(slots, comfy):
    assert resident.install()
    assert not resident.install()
    comfy.unload_all_models()
    assert comfy.calls == ["unload"]


def test_no_comfyui_is_not_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy", None)
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    assert not resident.install()


def test_a_failure_of_ours_never_reaches_comfyui(slots, comfy):
    def broken():
        raise OSError("the context would not close")

    slots[0].unload = broken
    resident.install()
    assert comfy.unload_all_models() == "stock"


def test_making_room_for_ourselves_leaves_ours_alone(slots, comfy, monkeypatch):
    """The pack evicts ComfyUI's models before loading; that must not come back round."""
    monkeypatch.setattr(devices, "shares_comfy_device", lambda device: True)
    resident.install()
    runner.free_comfy_vram("auto")
    assert comfy.calls == ["unload", "empty"]
    assert resident.held() == ["Qwen3.6-27B"]


def test_making_room_inside_a_run_leaves_no_debt(slots, comfy, monkeypatch):
    """What it would cost: the model being loaded, unloaded as the run ends."""
    monkeypatch.setattr(devices, "shares_comfy_device", lambda device: True)
    resident.install()
    with resident.in_use():
        runner.free_comfy_vram("auto")
    assert resident.held() == ["Qwen3.6-27B"]


def test_someone_elses_unload_inside_a_run_is_owed(slots, comfy):
    resident.install()
    with resident.in_use():
        comfy.unload_all_models()
        assert resident.held() == ["Qwen3.6-27B"]
    assert resident.held() == []


def test_the_transformers_engine_names_what_it_holds(monkeypatch):
    monkeypatch.setattr(engine, "_empty_cache", lambda: None)
    assert engine.holding() == ""
    engine._STATE.update(key=("somewhere",), model=object(), name="Qwen2.5-Omni-7B")
    try:
        assert engine.holding() == "Qwen2.5-Omni-7B"
        engine.unload()
        assert engine.holding() == "" and engine._STATE["name"] == ""
    finally:
        engine._STATE.update(key=None, tokenizer=None, model=None, processor=None, name="")


def test_the_gguf_engine_names_what_it_holds():
    closed = []
    llama = types.SimpleNamespace(close=lambda: closed.append(True))
    assert gguf_engine.holding() == ""
    gguf_engine._STATE.update(key=("somewhere",), llama=llama, name="Qwen3.6-27B-Q4_K_M.gguf")
    try:
        assert gguf_engine.holding() == "Qwen3.6-27B-Q4_K_M.gguf"
        gguf_engine.unload()
        assert closed == [True]
        assert gguf_engine.holding() == "" and gguf_engine._STATE["name"] == ""
    finally:
        gguf_engine._STATE.update(key=None, llama=None, lora=None, name="")


@pytest.fixture
def real_engines():
    """Both engines as they are, with nothing loaded before or after."""
    resident._RUNS.update(active=0, owed=False)
    yield
    engine._STATE.update(key=None, tokenizer=None, model=None, processor=None, name="")
    gguf_engine._STATE.update(key=None, llama=None, lora=None, name="")
    resident._RUNS.update(active=0, owed=False)


def transformers_run(monkeypatch, keep, during=lambda: None):
    """``engine.rewrite`` with the loading and the generating stood in for."""
    def load(base_dir, *args, **kwargs):
        engine._STATE.update(key=("k",), tokenizer=object(), model=object(), name="Qwen3.6-27B")
        return engine._STATE["tokenizer"], engine._STATE["model"]

    def generate(tokenizer, model, progress=None, **generation):
        during()
        return "text"

    monkeypatch.setattr(engine, "load", load)
    monkeypatch.setattr(engine, "generate", generate)
    monkeypatch.setattr(engine, "_empty_cache", lambda: None)
    return engine.rewrite("base", None, "bfloat16", "sdpa", keep_loaded=keep)


def gguf_run(monkeypatch, keep, during=lambda: None):
    """``gguf_engine.rewrite``, the same way."""
    def load(model_path, *args, **kwargs):
        llama = types.SimpleNamespace(close=lambda: None)
        gguf_engine._STATE.update(key=("k",), llama=llama, name="Qwen3.6-27B-Q4_K_M.gguf")
        return llama

    def generate(llama, progress=None, **generation):
        during()
        return "text"

    monkeypatch.setattr(gguf_engine, "load", load)
    monkeypatch.setattr(gguf_engine, "generate", generate)
    return gguf_engine.rewrite("model.gguf", None, -1, 4096, keep_loaded=keep)


@pytest.mark.parametrize("run", [transformers_run, gguf_run], ids=["transformers", "gguf"])
def test_a_run_holds_its_model_while_it_writes(real_engines, monkeypatch, run):
    seen = {}
    assert run(monkeypatch, True, lambda: seen.update(resident._RUNS)) == "text"
    assert seen["active"] == 1, "generating outside the guard that keeps an unload away"
    assert resident._RUNS["active"] == 0
    assert len(resident.held()) == 1


@pytest.mark.parametrize("run", [transformers_run, gguf_run], ids=["transformers", "gguf"])
def test_a_run_that_does_not_keep_unloads_as_before(real_engines, monkeypatch, run):
    assert run(monkeypatch, False) == "text"
    assert resident.held() == []


@pytest.mark.parametrize("run", [transformers_run, gguf_run], ids=["transformers", "gguf"])
def test_an_unload_asked_for_mid_run_lands_after_it(real_engines, monkeypatch, run):
    seen = {}

    def press():
        seen["answer"] = resident.release()
        seen["held"] = resident.held()

    assert run(monkeypatch, True, press) == "text"
    assert seen["answer"] == {"released": [], "waiting": True}
    assert len(seen["held"]) == 1, "the model went from under the run"
    assert resident.held() == []
