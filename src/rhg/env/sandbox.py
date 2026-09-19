"""Subprocess sandbox for model-written code (DESIGN §2.4, §8 item 11).

``run_python(script, ...)`` runs ``script`` in a fresh interpreter (``sys.executable -I -B -X utf8``)
with

* a temporary working directory (removed afterwards unless ``workdir`` is given) and the script
  itself stored in a *separate* temporary directory,
* a minimal, cleared environment: nothing is inherited, so API keys and other secrets of the
  parent are never visible to model code,
* stdout/stderr drained through pipes and capped at 64 KB each,
* stdin fed from ``stdin_data`` and then closed (this is how the grader hands the harness its
  payload and nonce; the payload never appears in the script file or the command line),
* a wall-clock timeout after which the whole process tree is killed (POSIX: own session +
  ``killpg``; Windows: ``taskkill /T /F``),
* POSIX only: ``RLIMIT_AS`` (``mem_mb``), ``RLIMIT_CPU``, ``RLIMIT_FSIZE`` and ``RLIMIT_CORE``
  via ``preexec_fn``. ``RLIMIT_NPROC`` is deliberately not set: it is a per-user limit compared
  with *all* of the user's processes, so any value small enough to matter can also break the
  parent's own workers. On Windows only the timeout is enforced (one-time warning).

Status: ``ok`` (exit code 0), ``timeout`` (killed by us, or by ``RLIMIT_CPU``), ``oom``
(SIGKILL we did not send, or ``MemoryError`` on stderr with a non-zero exit), ``crash`` (any other
non-zero exit, e.g. a syntax error).

Result channel (see ``rhg.env.grader``): the *result is never taken from stdout/stderr*. The
grader's harness receives ``{nonce, result paths, payload}`` on stdin, closes stdin, keeps
those only in locals, runs the model code, and writes one JSON object containing the nonce per
stage to a file with ``O_EXCL``. The parent accepts a file only if it parses as exactly one
JSON object with the right nonce and stage, so printing a fake result or writing a file at a
guessed path cannot change a graded result.

RESIDUAL RISK: model code runs in the *same interpreter* as the harness. Code that deliberately
introspects frames (``sys._getframe``, ``gc``), monkeypatches the interpreter, or returns
objects that compare equal to everything can in principle find the channel or fool the tests.
A 1.7B model is unlikely to do this on purpose, but it is not excluded; such behaviour would show
up as ``visible_pass``/``heldout_pass`` anomalies (``GAP_OTHER``) and is audited through the
judge (DESIGN §8 item 11). Processes are not network-isolated (no namespaces without root); the
cleared environment and lack of credentials are the mitigation.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

OUTPUT_CAP_BYTES = 64 * 1024
_IS_WINDOWS = os.name == "nt"
_log = logging.getLogger(__name__)
_windows_warned = False
_warn_lock = threading.Lock()


@dataclass(frozen=True)
class SandboxResult:
    status: str  # ok | timeout | crash | oom
    stdout: str
    stderr: str
    returncode: int | None
    wall_s: float
    stdout_truncated: bool = False
    stderr_truncated: bool = False


def _minimal_env(workdir: str) -> dict[str, str]:
    if _IS_WINDOWS:
        env = {"TEMP": workdir, "TMP": workdir, "USERPROFILE": workdir, "HOME": workdir}
        root = os.environ.get("SYSTEMROOT") or os.environ.get("SystemRoot") or r"C:\Windows"
        env["SYSTEMROOT"] = root
        return env
    return {"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir, "LANG": "C.UTF-8"}


def _warn_windows_once() -> None:
    global _windows_warned
    with _warn_lock:
        if _windows_warned:
            return
        _windows_warned = True
    _log.warning(
        "rhg.env.sandbox: on Windows only the wall-clock timeout is enforced (no memory/CPU/file "
        "rlimits). Use Linux for real runs."
    )


def _posix_limits(mem_mb: int, timeout_s: float):
    import math
    import resource

    mem = int(mem_mb) * 1024 * 1024
    cpu = max(1, math.ceil(timeout_s)) + 1
    fsize = 32 * 1024 * 1024

    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


def _kill_tree(proc: subprocess.Popen) -> None:
    if _IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            proc.kill()
        except OSError:
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass


def _drain(stream, cap: int, out: list) -> None:
    """Read ``stream`` to EOF, keeping at most ``cap`` bytes; ``out`` = [bytes, truncated]."""
    kept = bytearray()
    truncated = False
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            room = cap - len(kept)
            if room > 0:
                kept += chunk[:room]
            if len(chunk) > max(room, 0):
                truncated = True
    except (OSError, ValueError):
        pass
    out.append(bytes(kept))
    out.append(truncated)


def _feed(stream, data: bytes) -> None:
    try:
        stream.write(data)
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def run_python(
    script: str,
    *,
    timeout_s: float,
    mem_mb: int,
    workdir: str | os.PathLike | None = None,
    stdin_data: str | None = None,
) -> SandboxResult:
    """Run ``script`` in a fresh, isolated interpreter and return a :class:`SandboxResult`."""
    if not _IS_WINDOWS:
        limits = _posix_limits(mem_mb, timeout_s)
    else:
        limits = None
        _warn_windows_once()

    script_dir = tempfile.mkdtemp(prefix="rhg-sbx-script-")
    own_workdir = workdir is None
    cwd = tempfile.mkdtemp(prefix="rhg-sbx-cwd-") if own_workdir else str(workdir)
    proc = None
    threads: list[threading.Thread] = []
    out_buf: list = []
    err_buf: list = []
    timed_out = False
    start = time.monotonic()
    try:
        script_path = Path(script_dir) / "sandbox_main.py"
        script_path.write_text(script, encoding="utf-8", errors="surrogatepass", newline="\n")
        cmd = [sys.executable, "-I", "-B", "-X", "utf8", str(script_path)]
        kwargs: dict = {}
        if _IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
            kwargs["preexec_fn"] = limits
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=_minimal_env(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )
        payload = (stdin_data or "").encode("utf-8", errors="surrogatepass")
        threads = [
            threading.Thread(target=_drain, args=(proc.stdout, OUTPUT_CAP_BYTES, out_buf), daemon=True),
            threading.Thread(target=_drain, args=(proc.stderr, OUTPUT_CAP_BYTES, err_buf), daemon=True),
            threading.Thread(target=_feed, args=(proc.stdin, payload), daemon=True),
        ]
        for t in threads:
            t.start()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            proc.wait()
        if not _IS_WINDOWS:
            _kill_tree(proc)  # stragglers left in the process group after a normal exit
        for t in threads:
            t.join(timeout=5)
    finally:
        if proc is not None and proc.poll() is None:
            _kill_tree(proc)
            proc.wait()
        wall = time.monotonic() - start
        if proc is not None:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    stream.close()
                except (OSError, ValueError, AttributeError):
                    pass
        shutil.rmtree(script_dir, ignore_errors=True)
        if own_workdir:
            shutil.rmtree(cwd, ignore_errors=True)

    stdout_b, stdout_trunc = (out_buf + [b"", False])[:2]
    stderr_b, stderr_trunc = (err_buf + [b"", False])[:2]
    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    rc = proc.returncode
    if timed_out:
        status = "timeout"
    elif rc == 0:
        status = "ok"
    elif not _IS_WINDOWS and rc == -signal.SIGXCPU:
        status = "timeout"
    elif not _IS_WINDOWS and rc == -signal.SIGKILL:
        status = "oom"
    elif "MemoryError" in stderr:
        status = "oom"
    else:
        status = "crash"
    return SandboxResult(status, stdout, stderr, rc, wall, bool(stdout_trunc), bool(stderr_trunc))
