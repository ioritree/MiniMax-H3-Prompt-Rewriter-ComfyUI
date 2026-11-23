"""Resumable Hugging Face downloads with a byte-level progress callback.

``huggingface_hub`` is used only to discover an existing access token; the
transfer itself is a plain ranged HTTP download so progress can be reported per
chunk to the ComfyUI node that asked for it. Four details are load-bearing:

- **The token never reaches the CDN.** A Hub URL answers 302 with a pre-signed
  link, and an ``Authorization`` header on that hop is a 400 rather than an
  ignored header. ``requests`` strips the header itself when a redirect crosses
  hosts, which is exactly the behaviour wanted here.
- **A signed link expires**, so every retry re-requests the original Hub URL
  instead of a resolved one; resuming an hour later would otherwise download an
  error page.
- **``Accept-Encoding: identity``.** A transparently gzipped response makes the
  bytes written disagree with both ``Range`` offsets and the expected size.
- **Space is checked before the first byte**, because failing 50 GB into a 52 GB
  download costs the whole transfer.

Completion is tested by size. Bytes land in ``<name>.part`` and are renamed into
place only once whole, so a half-written shard is never mistaken for a model.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import ssl
import time
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter

log = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://huggingface.co"
CHUNK_SIZE = 1 << 22
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 60
MAX_ATTEMPTS = 5
BACKOFF_CAP = 30.0
USER_AGENT = "MiniMax-H3-Prompt-Rewriter-ComfyUI"
RETRY_STATUS = (408, 425, 429, 500, 502, 503, 504)
SPACE_MARGIN = 1.02


class DownloadError(RuntimeError):
    """The transfer cannot proceed; retrying the same call will not help."""


class _Retryable(Exception):
    """A transient failure that another attempt may clear."""


@dataclass
class RepoFile:
    path: str
    size: int


@dataclass
class FileTask:
    path: str
    url: str
    dest: str
    size: int
    already: int


def endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")


class _SystemTrust(HTTPAdapter):
    """Verify against what the operating system trusts, as well as certifi.

    ``requests`` checks a certificate against certifi's bundle and nothing else.
    An antivirus that inspects HTTPS re-signs the site's certificate --
    huggingface.co's and its CDN's alike -- with a root of its own and installs
    that root in the Windows store, so the browser accepts it. Python never
    looked there: every download stopped at "self-signed certificate in
    certificate chain" on a machine where the browser fetched the same file
    without a word.

    ``create_default_context`` loads the system store -- Windows' ROOT and CA
    stores, OpenSSL's default paths elsewhere -- and ``requests`` still adds
    certifi to it on every connection, so nothing that verified before stops
    verifying. Verification stays on. This widens whom it trusts to what the
    machine already trusts, which is what the browser does.
    """

    _context: ssl.SSLContext | None = None

    @classmethod
    def context(cls) -> ssl.SSLContext:
        if cls._context is None:
            cls._context = ssl.create_default_context()
        return cls._context

    def init_poolmanager(self, *args, **kwargs):
        kwargs.setdefault("ssl_context", self.context())
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        proxy_kwargs.setdefault("ssl_context", self.context())
        return super().proxy_manager_for(proxy, **proxy_kwargs)


def http(method: str, url: str, **kwargs) -> requests.Response:
    """``requests.request`` through ``_SystemTrust``; everything else as it was.

    A session per call, the way ``requests.get`` itself works: nothing shared
    between threads, and a streamed response keeps its connection when the
    session closes, because a connection in use is not in the pool it clears.
    """
    with requests.Session() as session:
        session.mount("https://", _SystemTrust())
        return session.request(method, url, **kwargs)


UNTRUSTED_HINT = (
    "The certificate presented for this site is trusted neither by this machine nor by "
    "certifi. Something between here and the site is inspecting HTTPS -- an antivirus "
    "scanning encrypted connections, or a company proxy -- with a root certificate that is "
    "not in the system store. Add the site to its exclusions, or install its root "
    "certificate into the system store."
)


def unreachable(where: str, error: Exception) -> str:
    """What to say when a request fails before any answer, with a hint for TLS."""
    message = f"Could not reach {where}: {error}"
    if isinstance(error, requests.exceptions.SSLError):
        message += f"\n\n{UNTRUSTED_HINT}"
    return message


def access_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value.strip()
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def human_size(size: float) -> str:
    for unit, scale in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if size >= scale:
            return f"{size / scale:.2f} {unit}"
    return f"{int(size)} B"


def _headers(token: str | None) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _interrupted() -> None:
    try:
        import comfy.model_management

        comfy.model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _safe_parts(repo_path: str) -> list[str]:
    """Split a repository path, refusing anything that escapes the destination."""
    parts = [part for part in re.split(r"[/\\]", repo_path) if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts) or ":" in repo_path:
        raise DownloadError(f"refusing unsafe repository path {repo_path!r}")
    return parts


def free_space(directory: str) -> int:
    probe = os.path.abspath(directory)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe).free


def check_space(directory: str, needed: int) -> None:
    """Refuse up front rather than filling the volume and failing at 99%."""
    if needed <= 0:
        return
    available = free_space(directory)
    if available < needed * SPACE_MARGIN:
        raise DownloadError(
            f"{human_size(needed)} is needed in {directory}, only {human_size(available)} is free."
        )


def _explain(status: int, what: str) -> str:
    if status in (401, 403):
        return (
            f"HTTP {status} for {what} — the repository is private or gated. Accept its licence "
            "on huggingface.co and set HF_TOKEN in the environment ComfyUI runs in."
        )
    if status == 404:
        return f"HTTP 404 for {what} — no such repository, revision, or file."
    return f"HTTP {status} for {what}"


def _pause(attempt: int) -> float:
    return min(2.0 ** attempt, BACKOFF_CAP)


def list_repo_files(repo_id: str, revision: str = "main", token: str | None = None) -> list[RepoFile]:
    url = f"{endpoint()}/api/models/{repo_id}/tree/{revision}?recursive=1"
    try:
        response = http("GET", url, headers=_headers(token), timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except requests.RequestException as error:
        raise DownloadError(unreachable(endpoint(), error)) from error

    if response.status_code >= 400:
        raise DownloadError(_explain(response.status_code, f"'{repo_id}' ({revision})"))

    files = []
    for entry in response.json():
        if entry.get("type") != "file":
            continue
        lfs = entry.get("lfs") or {}
        files.append(RepoFile(path=entry["path"], size=int(lfs.get("size") or entry.get("size") or 0)))
    return files


def select_files(
    files: list[RepoFile],
    allow: tuple[str, ...] | None = None,
    skip_suffixes: tuple[str, ...] = (),
) -> list[RepoFile]:
    selected = []
    for item in files:
        name = item.path.split("/")[-1]
        if allow is not None:
            if item.path in allow or name in allow:
                selected.append(item)
            continue
        if any(name.lower().endswith(suffix) for suffix in skip_suffixes):
            continue
        selected.append(item)
    return selected


def build_tasks(repo_id: str, dest_dir: str, files: list[RepoFile], revision: str = "main") -> list[FileTask]:
    tasks = []
    for item in files:
        dest = os.path.join(dest_dir, *_safe_parts(item.path))
        part = dest + ".part"
        already = 0
        if os.path.isfile(dest):
            local = os.path.getsize(dest)
            if local == item.size or (not item.size and local > 0):
                already = local
            else:
                os.remove(dest)
        elif os.path.isfile(part):
            # A leftover at or past the full size is corrupt, not finished. Counting it
            # as complete would skip the file entirely, so leave the task pending and
            # let the transfer finalize or restart it.
            partial = os.path.getsize(part)
            already = partial if not item.size or partial < item.size else 0
        tasks.append(
            FileTask(
                path=item.path,
                url=f"{endpoint()}/{repo_id}/resolve/{revision}/{item.path}",
                dest=dest,
                size=item.size,
                already=already,
            )
        )
    return tasks


def _finalize(part: str, dest: str) -> int:
    final = os.path.getsize(part)
    if os.path.isfile(dest):
        os.remove(dest)
    os.replace(part, dest)
    return final


def _stream_to_part(task: FileTask, token: str | None, base: int, on_progress) -> int:
    """Fetch one file, resuming its ``.part`` remainder when the server allows it.

    ``base`` is the absolute byte count of every *other* task, so the reported
    position stays correct whether ``Range`` is honoured or the transfer restarts
    from zero. Returns the final size of the file.
    """
    part = task.dest + ".part"
    name = task.path.split("/")[-1]
    os.makedirs(os.path.dirname(task.dest) or ".", exist_ok=True)

    offset = os.path.getsize(part) if os.path.isfile(part) else 0
    if task.size and offset >= task.size:
        if offset == task.size:
            return _finalize(part, task.dest)
        os.remove(part)
        offset = 0

    headers = _headers(token)
    if offset:
        headers["Range"] = f"bytes={offset}-"

    try:
        response = http(
            "GET",
            task.url,
            headers=headers,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=True,
        )
    except requests.exceptions.SSLError as error:
        raise DownloadError(unreachable(task.url.split("/resolve/")[0], error)) from error
    except requests.RequestException as error:
        raise _Retryable(str(error)) from error

    with response:
        if response.status_code in RETRY_STATUS:
            raise _Retryable(f"HTTP {response.status_code}")
        if response.status_code >= 400:
            raise DownloadError(_explain(response.status_code, task.path))

        mode = "ab"
        if offset and response.status_code != 206:
            log.warning("[minimax_h3_rewriter._stream_to_part] %s: range ignored, restarting", task.path)
            offset = 0
            mode = "wb"

        written = offset
        on_progress(base + written, name)
        try:
            with open(part, mode) as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    _interrupted()
                    handle.write(chunk)
                    written += len(chunk)
                    on_progress(base + written, name)
        except requests.RequestException as error:
            raise _Retryable(str(error)) from error

    if task.size and os.path.getsize(part) != task.size:
        raise _Retryable(f"{task.path}: {os.path.getsize(part)} bytes written, expected {task.size}")

    return _finalize(part, task.dest)


def download_task(task: FileTask, token: str | None, base: int, on_progress) -> int:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _stream_to_part(task, token, base, on_progress)
        except _Retryable as error:
            last_error = error
            log.warning(
                "[minimax_h3_rewriter.download_task] %s failed on attempt %d/%d: %s",
                task.path, attempt, MAX_ATTEMPTS, error,
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(_pause(attempt))
        except OSError as error:
            raise DownloadError(f"Could not write '{task.dest}': {error}") from error
    raise DownloadError(f"Gave up on '{task.path}' after {MAX_ATTEMPTS} attempts: {last_error}")


def repo_size(
    repo_id: str,
    allow: tuple[str, ...] | None = None,
    skip_suffixes: tuple[str, ...] = (),
    revision: str = "main",
) -> tuple[int, int]:
    """``(file count, total bytes)`` for the selected files, without downloading."""
    files = select_files(list_repo_files(repo_id, revision, access_token()), allow, skip_suffixes)
    return len(files), sum(item.size for item in files)


def sync_repo(
    repo_id: str,
    dest_dir: str,
    allow: tuple[str, ...] | None = None,
    skip_suffixes: tuple[str, ...] = (),
    revision: str = "main",
    on_progress=None,
    on_status=None,
    on_total=None,
) -> dict:
    """Mirror the selected files of ``repo_id`` into ``dest_dir``.

    Files already present at the expected size are left alone; interrupted files
    resume from their ``.part`` remainder.
    """
    token = access_token()
    if on_status is not None:
        on_status(f"Listing {repo_id}")

    files = select_files(list_repo_files(repo_id, revision, token), allow, skip_suffixes)
    if not files:
        raise DownloadError(f"No matching files found in '{repo_id}'.")

    os.makedirs(dest_dir, exist_ok=True)
    tasks = build_tasks(repo_id, dest_dir, files, revision)

    total = sum(task.size for task in tasks)
    transferred = sum(task.already for task in tasks)
    pending = [task for task in tasks if not (task.size and task.already == task.size)]
    check_space(dest_dir, total - transferred)

    if on_total is not None:
        on_total(total)
    if on_progress is None:
        on_progress = lambda *_: None
    on_progress(transferred, "")

    for task in pending:
        _interrupted()
        base = transferred - task.already
        transferred = base + download_task(task, token, base, on_progress)

    on_progress(total, "")
    log.info(
        "[minimax_h3_rewriter.download] %s: %d of %d files fetched into %s",
        repo_id, len(pending), len(tasks), dest_dir,
    )
    return {
        "repo_id": repo_id,
        "dir": dest_dir,
        "files": len(tasks),
        "downloaded_files": len(pending),
        "total_bytes": total,
    }
