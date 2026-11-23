"""Resolution of local model directories under the ComfyUI models tree."""

from __future__ import annotations

import json
import logging
import os
import re

from .constants import ADAPTER_FILES, MODELS_SUBDIR

log = logging.getLogger(__name__)


def _models_root() -> str:
    import folder_paths

    try:
        registered = folder_paths.get_folder_paths(MODELS_SUBDIR)
    except KeyError:
        registered = []
    if registered:
        return registered[0]

    root = os.path.join(folder_paths.models_dir, MODELS_SUBDIR)
    try:
        folder_paths.add_model_folder_path(MODELS_SUBDIR, root)
    except Exception:
        log.debug("[minimax_h3_rewriter._models_root] could not register %s", MODELS_SUBDIR)
    return root


def models_root() -> str:
    """Return ``ComfyUI/models/LLM``, creating it on first use."""
    root = _models_root()
    os.makedirs(root, exist_ok=True)
    return root


def embeddings_root() -> str:
    """Return ComfyUI's own embeddings folder, creating it on first use.

    Not ``models_root()``: this one is not ours to place. ``embedding:`` reads
    the directory ComfyUI hands the tokenizer and nothing else, so a file
    anywhere else is a file the prompt cannot name. The folder is registered by
    ComfyUI itself, and the fallback is only for a build that renamed it.
    """
    import folder_paths

    try:
        registered = folder_paths.get_folder_paths("embeddings")
    except KeyError:
        registered = []

    root = registered[0] if registered else os.path.join(folder_paths.models_dir, "embeddings")
    os.makedirs(root, exist_ok=True)
    return root


def local_dir_for_repo(repo_id: str) -> str:
    return os.path.join(models_root(), repo_id.rstrip("/").split("/")[-1])


def looks_like_repo_id(value: str) -> bool:
    return "/" in value and not os.path.isabs(value) and "\\" not in value


_EXTENDED_LOCAL = re.compile(r"^\\\\[?.]\\[A-Za-z]:")


def is_network_path(value: str) -> bool:
    """Whether this string names a UNC share rather than something on this machine."""
    text = (value or "").strip()
    if os.name == "nt":
        text = text.replace("/", "\\")
    return text.startswith("\\\\") and not _EXTENDED_LOCAL.match(text)


def refuse_network_path(value: str, field: str, advice: str) -> None:
    """Refuse a network location that arrived from outside this machine.

    Paths written into ``models.json`` by hand are the user's own and may point
    wherever they like, a NAS included; that file never leaves the machine unless
    somebody sends it. A string that arrived over the wire is the other kind --
    inside a downloaded workflow, or in a request to the ComfyUI API, which has
    no CSRF token and is routinely served on ``--listen``. Merely *looking* at a
    UNC path is an authentication attempt against whatever host it names: it
    happens on the ``isfile`` call, long before anything judges the model.

    Nothing legitimate is lost. A share holding your models is reachable through
    a drive letter or a mount point like any other folder, and ``advice`` says so
    in the words that fit wherever this is being called from.
    """
    if not is_network_path(value):
        return
    raise RuntimeError(f"'{value}' is a network path, and {field} will not follow one.\n\n{advice}")


def catalog_file(repo: str, name: str) -> str:
    """Where one file of a catalog entry sits on disk, or ``""`` if it is on the Hub.

    A ``models.json`` entry names either a Hugging Face repository or something
    already on this machine, and there are two natural ways to write the second:
    ``repo`` as the folder with ``file`` a name inside it, or ``file`` as the
    whole path with ``repo`` left out. Both are accepted, because both are what
    people actually type -- and the alternative is a file that is right there on
    the disk and cannot be named.

    The answer is a path, not a promise: the caller still has to check it exists.
    """
    name = (name or "").strip()
    repo = (repo or "").strip()
    if not name:
        return ""
    if os.path.isabs(name):
        return os.path.normpath(name)
    if repo and os.path.isdir(repo):
        return os.path.normpath(os.path.join(repo, name))
    return ""


def llm_roots() -> list[str]:
    """Every folder ComfyUI knows as ``LLM``, ``models_root`` first.

    Downloads go to ``models_root``, the first registered one -- and an
    ``extra_model_paths.yaml`` that names an ``LLM:`` folder of its own becomes
    that first one. Everything fetched before the line was added stays where it
    was, in ``ComfyUI/models/LLM``, and a lookup that asked only the new first
    folder downloaded all of it again: 19.5 GB for the 27B, from a node whose
    file was sitting one folder over. So what is already here is looked for in
    all of them, and only where a new download lands is ``models_root``'s call.
    """
    found = [models_root()]
    try:
        import folder_paths
    except ImportError:
        return found
    try:
        registered = list(folder_paths.get_folder_paths(MODELS_SUBDIR))
    except KeyError:
        registered = []
    registered.append(os.path.join(folder_paths.models_dir, MODELS_SUBDIR))
    for path in registered:
        if os.path.isdir(path) and not any(same_path(path, one) for one in found):
            found.append(path)
    return found


def present_file(repo: str, name: str) -> str:
    """Where one file of an entry already is, so that running it downloads nothing.

    The first half of ``nodes._ensure_file``, which calls this: a file the entry
    names on this machine, the copy a previous run put flat in an ``LLM``
    folder, or one fetched by hand into a folder named after its repository.
    The model list uses the same answer to tell an entry from the scanned row
    for the file it downloaded -- one rule, so hiding that row never changes
    which file runs.
    """
    on_disk = catalog_file(repo, name)
    if on_disk:
        return on_disk if os.path.isfile(on_disk) else ""
    if not name or os.path.isabs(repo or ""):
        return ""
    folder = (repo or "").rstrip("/").split("/")[-1]
    for root in llm_roots():
        candidates = [os.path.join(root, name)]
        if folder:
            candidates.append(os.path.join(root, folder, name))
        for candidate in candidates:
            if _present(candidate):
                return candidate
    return ""


def present_pair(repo: str, file: str, mmproj: str) -> tuple[str, str] | None:
    """Where a model and its projector already are, or ``None``.

    The first half of ``nodes._ensure_pair``, and for the same reason as
    ``present_file``. Both files or nothing, and both from one place: a pair
    with its projector missing still has a download ahead of it.

    A pair is fetched into a folder named after its repository, but one put
    flat in an ``LLM`` folder -- by an older version, or by hand -- is the same
    two files. Asking only the folder downloaded 5 GB of Qwen3-VL-8B beside an
    exact copy of it.
    """
    on_disk = (catalog_file(repo, file), catalog_file(repo, mmproj))
    if all(on_disk):
        return on_disk if all(os.path.isfile(path) for path in on_disk) else None
    folder = (repo or "").rstrip("/").split("/")[-1]
    for root in llm_roots():
        for where in ((root, folder) if folder else (root,), (root,)):
            targets = (os.path.join(*where, file), os.path.join(*where, mmproj))
            if all(_present(path) for path in targets):
                return targets
    return None


def present_folder(value: str, default_repo: str = "", complete=None) -> str:
    """Where a checkpoint folder already is, complete, or an empty string.

    ``resolve_source`` names the folder in ``models_root``; the same name is
    looked for in every other ``LLM`` folder after it. ``complete`` is the test
    for "all here" -- a base model's by default, an adapter's where it is one.
    """
    complete = complete or base_model_is_complete
    value = (value or "").strip() or default_repo
    if os.path.isabs(value):
        return os.path.normpath(value) if complete(value) else ""
    name = value.rstrip("/").split("/")[-1] if looks_like_repo_id(value) else value
    for root in llm_roots():
        candidate = os.path.join(root, name)
        if complete(candidate):
            return candidate
    return ""


def same_path(one: str, other: str) -> bool:
    """Whether two paths name one file, however each was spelt."""
    if not one or not other:
        return False
    return os.path.normcase(os.path.abspath(one)) == os.path.normcase(os.path.abspath(other))


def resolve_source(value: str, default_repo: str) -> tuple[str, str]:
    """Map a user-entered model reference to ``(repo_id, local_dir)``.

    ``repo_id`` is empty when the reference points at a directory that is not
    managed by this package, in which case nothing is ever downloaded into it.
    """
    value = (value or "").strip()
    if not value:
        value = default_repo

    if os.path.isabs(value):
        return "", os.path.normpath(value)

    candidate = os.path.join(models_root(), value)
    if os.path.isdir(candidate) and not looks_like_repo_id(value):
        return "", candidate

    if looks_like_repo_id(value):
        return value, local_dir_for_repo(value)

    return "", candidate


def _shard_names(directory: str) -> list[str] | None:
    index = os.path.join(directory, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return None
    try:
        with open(index, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
    except (OSError, ValueError):
        return None
    return sorted(set(weight_map.values()))


def _present(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def base_model_is_complete(directory: str) -> bool:
    if not _present(os.path.join(directory, "config.json")):
        return False

    shards = _shard_names(directory)
    if shards is not None:
        return all(_present(os.path.join(directory, name)) for name in shards)

    try:
        entries = os.listdir(directory)
    except OSError:
        return False
    return any(name.endswith(".safetensors") for name in entries)


def adapter_is_complete(directory: str) -> bool:
    return all(_present(os.path.join(directory, name)) for name in ADAPTER_FILES)
