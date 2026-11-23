"""The MiniMax-H3 prompt-writing guides, fetched from MiniMax rather than bundled.

``docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md`` and its ``_ref_en`` companion are
what the writer nodes put in front of a general-purpose model in place of the
LoRA. They are deliberately *not* shipped inside this pack.

They are MiniMax's Documentation, and the MiniMax H3 Community License grants the
right to redistribute the Materials only within the "Applicable Territory" --
worldwide *excluding* the European Union, the United Kingdom, the Republic of
Korea and the United States -- and only alongside a copy of the agreement and a
NOTICE file. A node pack on the Comfy registry cannot honour a territory
boundary, so each installation fetches its own copy directly from MiniMax
instead. It is 16 and 24 KB, once, and it always matches upstream.

The copy lands in the ComfyUI user directory and is read from there afterwards,
which also makes it editable: trimming the guide is the cheapest way to fit a
small model's context window, and a fetch never overwrites a file already there.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

GUIDE_REPO = "MiniMaxAI/MiniMax-H3"
GUIDE_REVISION = "main"

GUIDE_FILES = {
    "base": "docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md",
    "ref": "docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md",
}

GUIDE_TITLES = {
    "base": "Video Prompt Writing Guide (T2VA / I2VA / FL2VA / L2VA)",
    "ref": "Full-Reference Mode Rewrite Output Format Guide (Ref2VA)",
}

USER_SUBDIR = "minimax_h3_rewriter"
GUIDES_DIRNAME = "guides"

MIN_BYTES = 2048

_TEXT_CACHE: dict[tuple[str, int, int], str] = {}


def root() -> str:
    """Where fetched guides live: ``<ComfyUI user>/minimax_h3_rewriter/guides``."""
    try:
        import folder_paths

        base = folder_paths.get_user_directory()
    except Exception:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_user")
    return os.path.join(base, USER_SUBDIR, GUIDES_DIRNAME)


def kinds() -> tuple[str, ...]:
    return tuple(GUIDE_FILES)


def path(kind: str) -> str:
    try:
        return os.path.join(root(), os.path.basename(GUIDE_FILES[kind]))
    except KeyError:
        raise ValueError(f"unknown guide '{kind}'; expected one of {', '.join(GUIDE_FILES)}") from None


def url(kind: str) -> str:
    from .download import endpoint

    return f"{endpoint()}/{GUIDE_REPO}/resolve/{GUIDE_REVISION}/{GUIDE_FILES[kind]}"


def present(kind: str) -> bool:
    destination = path(kind)
    try:
        return os.path.getsize(destination) >= MIN_BYTES
    except OSError:
        return False


def fetch(kind: str, progress=None) -> str:
    """Download one guide into :func:`root`, atomically. Returns its path."""
    from .download import (
        CONNECT_TIMEOUT,
        READ_TIMEOUT,
        DownloadError,
        _headers,
        access_token,
        http,
        unreachable,
    )

    import requests

    destination = path(kind)
    source = url(kind)
    if progress is not None:
        progress.text(f"Fetching the {GUIDE_TITLES[kind]}\nfrom {GUIDE_REPO}", force=True)

    try:
        response = http(
            "GET",
            source,
            headers=_headers(access_token()),
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=True,
        )
    except requests.RequestException as error:
        raise DownloadError(unreachable(source, error)) from error

    if response.status_code >= 400:
        raise DownloadError(
            f"HTTP {response.status_code} for {source}. Download it by hand and save it as "
            f"'{destination}'."
        )

    body = response.content
    if len(body) < MIN_BYTES:
        raise DownloadError(
            f"{source} returned only {len(body)} bytes, which is not the guide. Download it by "
            f"hand and save it as '{destination}'."
        )

    part = destination + ".part"
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(part, "wb") as handle:
            handle.write(body)
        os.replace(part, destination)
    except OSError as error:
        raise DownloadError(f"Could not write '{destination}': {error}") from error

    log.info("[minimax_h3_rewriter.guides.fetch] %s -> %s (%d bytes)", source, destination, len(body))
    return destination


def ensure(kind: str, auto_download: bool = True, progress=None) -> str:
    """Return the local path of a guide, fetching it once if it is missing."""
    destination = path(kind)
    if present(kind):
        return destination
    if not auto_download:
        raise RuntimeError(
            f"The {GUIDE_TITLES[kind]} is not in '{destination}' and auto_download is off.\n"
            f"Enable it, or save {url(kind)} at that path yourself."
        )
    return fetch(kind, progress)


def text(kind: str, auto_download: bool = True, progress=None) -> str:
    """The guide's Markdown, cached per file identity."""
    destination = ensure(kind, auto_download, progress)
    try:
        stat = os.stat(destination)
    except OSError as error:
        raise RuntimeError(f"Cannot read '{destination}': {error}") from error

    key = (os.path.normcase(destination), stat.st_size, int(stat.st_mtime))
    cached = _TEXT_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        with open(destination, "r", encoding="utf-8") as handle:
            body = handle.read().strip()
    except OSError as error:
        raise RuntimeError(f"Cannot read '{destination}': {error}") from error

    if len(body) < MIN_BYTES:
        raise RuntimeError(
            f"'{destination}' holds only {len(body)} characters, which is not the guide. "
            f"Delete it and let the node fetch it again."
        )

    _TEXT_CACHE[key] = body
    return body


def reveal() -> str:
    """Open the guide folder in the desktop file manager."""
    import subprocess
    import sys

    directory = root()
    os.makedirs(directory, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(directory)  # noqa: S606 - opening the user's own folder
    elif sys.platform == "darwin":
        subprocess.Popen(["open", directory])
    else:
        subprocess.Popen(["xdg-open", directory])
    return directory
