"""Running a llama.cpp executable once and reading its output as it arrives.

Shared by every subprocess backend in the pack: the rewriter drives
``llama-completion`` and the captioner drives ``llama-mtmd-cli``, but the part
that matters is identical and is the part that is easy to get wrong.

Four things are load-bearing:

- **stdin is closed, not inherited.** A child that can never block waiting for a
  key is one failure mode fewer, and ComfyUI's own stdin is not ours to read.
- **Decoding carries state.** Token pieces arrive as raw bytes and a multi-byte
  character can straddle two reads, so a per-chunk ``bytes.decode`` mangles any
  non-ASCII output.
- **Silence is timed.** Generous before the first byte, because loading tens of
  gigabytes from a cold disk legitimately takes minutes; short afterwards,
  because a model that has begun emitting and then stops for three minutes is
  not going to resume. Either way the node fails with a message instead of
  wedging the ComfyUI queue, which is what a subprocess that never exits does.
- **No child outlives the interpreter.** Every process started here is
  registered, and an ``atexit`` hook kills whatever is still holding VRAM.
"""

from __future__ import annotations

import atexit
import codecs
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time

log = logging.getLogger(__name__)

READ_CHUNK = 4096
POLL_SECONDS = 0.25
STDERR_TAIL = 40

FIRST_BYTE_SECONDS = 900.0
STALL_SECONDS = 180.0

_PERF = re.compile(r"eval time =.*?\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)")

_END_MARKER = re.compile(r"\s*\[end of text\]\s*$")

_LIVE: set = set()


def _kill_all() -> None:
    for process in list(_LIVE):
        if process.poll() is None:
            try:
                process.kill()
            except Exception:
                pass


atexit.register(_kill_all)


class ChildFailed(RuntimeError):
    """The subprocess stalled, crashed, or exited non-zero."""


#: Phrases llama.cpp prints when the card runs out of room, in any of the
#: three places it can happen: the weights, the compute buffer, the KV cache.
OUT_OF_MEMORY_MARKERS = (
    "out of memory",
    "cudamalloc failed",
    "failed to allocate buffer for kv cache",
    "alloc_tensor_range: failed to allocate",
    "vk::device::allocatememory",
)


def out_of_memory(text: str) -> bool:
    """True when the child died for want of VRAM rather than a bad model."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in OUT_OF_MEMORY_MARKERS)


def free_comfy_vram(device: str = "auto") -> None:
    """Evict ComfyUI's models, unless this run is going somewhere else entirely.

    Making room is right when both models want the same card and actively
    harmful when they do not: on a second GPU, unloading the diffusion model
    costs a full reload after the rewrite and buys nothing.
    """
    from . import devices

    if not devices.shares_comfy_device(device):
        log.info(
            "[minimax_h3_rewriter.runner.free_comfy_vram] running on %s, leaving ComfyUI's "
            "models where they are", device,
        )
        return
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache(force=True)
    except Exception:
        log.debug("[minimax_h3_rewriter.runner.free_comfy_vram] skipped", exc_info=True)


def interrupted() -> bool:
    try:
        import comfy.model_management as mm

        return bool(mm.processing_interrupted())
    except Exception:
        return False


def spawn(command: list[str], binary: str) -> subprocess.Popen:
    environment = dict(os.environ)
    directory = os.path.dirname(os.path.abspath(binary))
    if sys.platform != "win32":
        # The tar releases put the shared libraries beside the executable.
        existing = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = f"{directory}{os.pathsep}{existing}" if existing else directory

    creation = 0
    if sys.platform == "win32":
        # Otherwise a console window flashes over the ComfyUI browser tab.
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=directory,
        env=environment,
        creationflags=creation,
        bufsize=0,
    )
    _LIVE.add(process)
    return process


def _pump(stream, sink: queue.Queue) -> None:
    try:
        while True:
            chunk = stream.read(READ_CHUNK)
            if not chunk:
                break
            sink.put(chunk)
    except Exception:
        log.debug("[minimax_h3_rewriter.runner._pump] reader stopped", exc_info=True)
    finally:
        sink.put(None)


def _drain(sink: queue.Queue) -> str:
    chunks = []
    while True:
        try:
            chunk = sink.get_nowait()
        except queue.Empty:
            break
        if chunk is None:
            continue
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def speed(stderr_text: str) -> str:
    match = _PERF.search(stderr_text)
    return f" · {match.group(2)} tok/s" if match else ""


def run(
    command: list[str],
    binary: str,
    on_text=None,
    first_byte_seconds: float = FIRST_BYTE_SECONDS,
    stall_seconds: float = STALL_SECONDS,
) -> tuple[str, str]:
    """Run to completion, streaming stdout. Returns ``(stdout text, stderr text)``.

    ``on_text`` is called with the whole text so far every time more arrives, so
    a caller can drive a progress bar and a preview without re-implementing the
    incremental decode.
    """
    log.info("[minimax_h3_rewriter.runner.run] %s", " ".join(command))

    try:
        process = spawn(command, binary)
    except OSError as error:
        raise ChildFailed(f"Could not start '{binary}': {error}") from error

    output: queue.Queue = queue.Queue()
    errors: queue.Queue = queue.Queue()
    threading.Thread(target=_pump, args=(process.stdout, output), daemon=True).start()
    threading.Thread(target=_pump, args=(process.stderr, errors), daemon=True).start()

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pieces: list[str] = []
    was_interrupted = False
    finished = False
    stalled = ""
    last_output = time.monotonic()

    try:
        while not finished:
            if interrupted():
                was_interrupted = True
                break
            try:
                chunk = output.get(timeout=POLL_SECONDS)
            except queue.Empty:
                limit = stall_seconds if pieces else first_byte_seconds
                waited = time.monotonic() - last_output
                if waited > limit:
                    stalled = (
                        f"produced nothing for {waited:.0f} s"
                        if not pieces
                        else f"stopped mid-generation for {waited:.0f} s"
                    )
                    break
                continue
            if chunk is None:
                finished = True
                break
            last_output = time.monotonic()
            text = decoder.decode(chunk)
            if not text:
                continue
            pieces.append(text)
            if on_text is not None:
                on_text("".join(pieces))
        pieces.append(decoder.decode(b"", final=True))
    finally:
        if was_interrupted or stalled or process.poll() is None:
            process.kill()
        process.wait()
        _LIVE.discard(process)

    stderr_text = _drain(errors)

    if was_interrupted:
        import comfy.model_management as mm

        raise mm.InterruptProcessingException()

    if stalled:
        tail = "\n".join(stderr_text.splitlines()[-STDERR_TAIL:])
        raise ChildFailed(
            f"{os.path.basename(binary)} {stalled} and was stopped, so it could not wedge "
            f"the queue. Last output from it:\n{tail}"
        )

    if process.returncode != 0:
        tail = "\n".join(stderr_text.splitlines()[-STDERR_TAIL:])
        raise ChildFailed(
            f"{os.path.basename(binary)} exited with code {process.returncode}.\n{tail}"
        )

    text = _END_MARKER.sub("", "".join(pieces).strip()).strip()
    return text, stderr_text
