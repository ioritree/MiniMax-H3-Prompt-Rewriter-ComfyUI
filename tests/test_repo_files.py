"""Check it with a repository and no file: the GGUF files in it, to pick from.

Typing a repository is easy; typing the exact name of one file inside it is
where entries went wrong -- and the window refused the repository on its own
for having no file. Now that question is answered with what the repository
holds: the models smallest first, each with a name the dropdown could use, and
the projectors apart for the lists that need one.

The Hub is never asked here: ``download.list_repo_files`` is replaced with a
listing, which is also all the real call returns.
"""

import importlib
import pathlib
import sys
import types

import pytest

_PKG = "minimax_h3_rewriter"
ROOT = pathlib.Path(__file__).resolve().parent.parent

if _PKG not in sys.modules:
    _package = types.ModuleType(_PKG)
    _package.__path__ = [str(ROOT / _PKG)]
    sys.modules[_PKG] = _package

download = importlib.import_module(f"{_PKG}.download")
sections = importlib.import_module(f"{_PKG}.model_sections")
routes = importlib.import_module(f"{_PKG}.routes")

GB = 1_000_000_000

OMNI_REPO = [
    ("README.md", 2_000),
    (".gitattributes", 1_500),
    ("Qwen2.5-Omni-7B-Q8_0.gguf", int(8.1 * GB)),
    ("Qwen2.5-Omni-7B-Q4_K_M.gguf", int(4.68 * GB)),
    ("Qwen2.5-Omni-7B-f16.gguf", int(15.2 * GB)),
    ("mmproj-Qwen2.5-Omni-7B-Q8_0.gguf", int(1.5 * GB)),
    ("mmproj-Qwen2.5-Omni-7B-f16.gguf", int(2.6 * GB)),
    ("BF16/Qwen2.5-Omni-7B-BF16-00001-of-00002.gguf", int(9 * GB)),
    ("BF16/Qwen2.5-Omni-7B-BF16-00002-of-00002.gguf", int(6 * GB)),
]


@pytest.fixture
def hub(monkeypatch):
    """The Hub, answering with whatever ``hub.files`` holds; ``hub.asked`` counts calls."""
    state = types.SimpleNamespace(files=list(OMNI_REPO), asked=0, error=None)

    def listing(repo_id, revision="main", token=None):
        state.asked += 1
        if state.error:
            raise download.DownloadError(state.error)
        return [download.RepoFile(path=path, size=size) for path, size in state.files]

    monkeypatch.setattr(download, "list_repo_files", listing)
    monkeypatch.setattr(download, "access_token", lambda: None)
    return state


def levels(answer):
    return [line["level"] for line in answer["lines"]]


def said(answer):
    return " ".join(line["text"] for line in answer["lines"])


@pytest.mark.parametrize(
    "path, name",
    [
        ("Qwen2.5-Omni-7B-Q4_K_M.gguf", "Qwen2.5-Omni-7B Q4_K_M"),
        ("Qwen3.8-27B-UD-Q2_K_XL.gguf", "Qwen3.8-27B UD-Q2_K_XL"),
        ("gemma-3-12b-it-IQ4_XS.gguf", "gemma-3-12b-it IQ4_XS"),
        ("Qwen2.5-Omni-7B-f16.gguf", "Qwen2.5-Omni-7B f16"),
        ("model.BF16.gguf", "model BF16"),
        ("Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf", "Qwen3-30B-A3B Q4_K_M"),
        ("plain-model.gguf", "plain-model"),
    ],
)
def test_the_suggested_name_sets_the_quantisation_apart(path, name):
    assert sections.suggested_name(path) == name


def test_the_packaged_entries_are_named_the_same_way():
    """The shape of a suggestion is the shape of the names already in the list."""
    assert sections.suggested_name("Qwen3.8-27B-UD-Q2_K_XL.gguf").endswith("Q2_K_XL")
    assert sections.suggested_name("Qwen2.5-Omni-7B-Q4_K_M.gguf") == "Qwen2.5-Omni-7B Q4_K_M"


def test_a_list_with_a_projector_gets_models_and_projectors_apart(hub):
    answer = sections.repo_files("captioners", "ggml-org/Qwen2.5-Omni-7B-GGUF")
    assert [one["file"] for one in answer["files"]] == [
        "Qwen2.5-Omni-7B-Q4_K_M.gguf",
        "Qwen2.5-Omni-7B-Q8_0.gguf",
        "Qwen2.5-Omni-7B-f16.gguf",
    ]
    assert answer["files"][0] == {
        "file": "Qwen2.5-Omni-7B-Q4_K_M.gguf",
        "gb": 4.68,
        "name": "Qwen2.5-Omni-7B Q4_K_M",
    }
    assert [one["file"] for one in answer["projectors"]] == [
        "mmproj-Qwen2.5-Omni-7B-Q8_0.gguf",
        "mmproj-Qwen2.5-Omni-7B-f16.gguf",
    ]
    assert answer["verdict"] == "good"
    assert "3 GGUF models and 2 projectors" in said(answer)
    assert "then its projector" in said(answer)


def test_a_split_model_is_counted_and_left_out(hub):
    """An entry names one file and only that file is fetched, so half a model is no offer."""
    answer = sections.repo_files("captioners", "ggml-org/Qwen2.5-Omni-7B-GGUF")
    assert not any("-of-" in one["file"] for one in answer["files"])
    assert answer["split"] == 1
    assert "1 more is split into parts" in said(answer)


def test_a_list_without_a_projector_is_offered_none(hub):
    answer = sections.repo_files("writers", "ggml-org/Qwen2.5-Omni-7B-GGUF")
    assert answer["projectors"] == []
    assert len(answer["files"]) == 3
    assert "projector" not in said(answer)


def test_a_missing_projector_is_a_warning_where_one_is_needed(hub):
    hub.files = [("tiny-Q4_K_M.gguf", GB)]
    answer = sections.repo_files("captioners", "someone/tiny-GGUF")
    assert answer["verdict"] == "warn"
    assert "no projector" in said(answer)


def test_a_repository_without_gguf_says_so(hub):
    hub.files = [("model.safetensors", 5 * GB), ("config.json", 1_000)]
    answer = sections.repo_files("models", "Qwen/Some-Model")
    assert answer["verdict"] == "bad" and answer["files"] == []
    assert "'transformers' format" in said(answer)

    only_gguf = sections.repo_files("writers", "Qwen/Some-Model")
    assert "takes GGUF only" in said(only_gguf)


def test_a_repository_that_cannot_be_read_says_why(hub):
    hub.error = "'someone/private' was not found, or it is gated"
    answer = sections.repo_files("writers", "someone/private")
    assert answer["verdict"] == "bad"
    assert "gated" in said(answer)
    assert answer["files"] == [] and answer["projectors"] == []


def test_something_that_is_not_a_repository_is_not_asked_of_the_hub(hub):
    answer = sections.repo_files("writers", "just-a-name")
    assert answer["verdict"] == "bad" and hub.asked == 0


def test_a_network_path_is_refused_before_anything_is_read(hub):
    with pytest.raises(RuntimeError, match="network path"):
        sections.repo_files("writers", r"\\server\share\models")
    assert hub.asked == 0


def test_a_folder_on_this_machine_is_read_where_it_stands(hub, tmp_path):
    (tmp_path / "small-Q4_K_M.gguf").write_bytes(b"\0" * 300)
    (tmp_path / "small-Q8_0.gguf").write_bytes(b"\0" * 500)
    (tmp_path / "mmproj-small-f16.gguf").write_bytes(b"\0" * 100)
    (tmp_path / "notes.txt").write_text("not a model")
    answer = sections.repo_files("captioners", str(tmp_path))
    assert hub.asked == 0
    assert [one["file"] for one in answer["files"]] == ["small-Q4_K_M.gguf", "small-Q8_0.gguf"]
    assert [one["file"] for one in answer["projectors"]] == ["mmproj-small-f16.gguf"]
    assert said(answer).startswith("This folder holds 2 GGUF models")


def test_a_long_listing_is_cut_to_the_smallest(hub, monkeypatch):
    monkeypatch.setattr(sections, "SHOWN_REPO_FILES", 2)
    answer = sections.repo_files("writers", "ggml-org/Qwen2.5-Omni-7B-GGUF")
    assert [one["file"] for one in answer["files"]] == [
        "Qwen2.5-Omni-7B-Q4_K_M.gguf",
        "Qwen2.5-Omni-7B-Q8_0.gguf",
    ]
    assert "The 2 smallest" in said(answer)


def test_the_window_asks_with_no_file_and_no_name_yet(hub):
    """The question comes before the cleaning, which would refuse both blanks."""
    answer = routes._check_entry(
        "captioners", {"format": "gguf", "repo": "ggml-org/Qwen2.5-Omni-7B-GGUF"}
    )
    assert answer["ok"] and len(answer["files"]) == 3


def test_a_list_that_defaults_to_transformers_is_not_listed(hub):
    """A repository is the whole entry for a transformers model, so it is judged as one."""
    with pytest.raises(RuntimeError, match="name"):
        routes._check_entry("models", {"repo": "Qwen/Some-Model"})
    assert hub.asked == 0


@pytest.fixture
def nowhere_on_disk(monkeypatch):
    monkeypatch.setattr(sections, "on_disk", lambda *args, **kwargs: ())
    monkeypatch.setattr(sections, "_local_copy", lambda *args, **kwargs: "")


def test_a_misspelt_file_offers_what_the_repository_holds(hub, nowhere_on_disk):
    entry = sections.clean_entry("captioners", {
        "name": "Omni",
        "repo": "ggml-org/Qwen2.5-Omni-7B-GGUF",
        "file": "Qwen2.5-Omni-7B-Q4_K_S.gguf",
        "mmproj": "mmproj-Qwen2.5-Omni-7B-Q8_0.gguf",
    })
    answer = sections.check("captioners", entry)
    assert answer["verdict"] == "bad"
    assert "pick from what it holds below" in said(answer)
    assert len(answer["files"]) == 3 and len(answer["projectors"]) == 2


def test_a_right_entry_offers_nothing_and_sums_both_files(hub, nowhere_on_disk):
    entry = sections.clean_entry("captioners", {
        "name": "Omni",
        "repo": "ggml-org/Qwen2.5-Omni-7B-GGUF",
        "file": "Qwen2.5-Omni-7B-Q4_K_M.gguf",
        "mmproj": "mmproj-Qwen2.5-Omni-7B-Q8_0.gguf",
    })
    answer = sections.check("captioners", entry)
    assert answer["verdict"] == "good"
    assert answer["download_gb"] == 6.18
    assert "files" not in answer
