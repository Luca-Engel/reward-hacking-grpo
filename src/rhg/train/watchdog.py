"""Stall watchdog: kill a run that stops making progress (exit code 75 = infrastructure stall).

Guards against the known TRL + vLLM colocate hang reports, which would otherwise burn GPU rental
silently. A daemon thread compares ``now - last heartbeat`` with ``timeout_s``; on a stall it dumps
every thread's stack (``faulthandler``) to the log, calls ``on_stall(reason)`` (the run driver writes
status ``failed`` / reason ``stall`` and the ledger entry there) and terminates the process with
``os._exit(75)`` (``exit_fn`` is injectable for tests). The heartbeat is beaten after every training step
and during evals; the clock starts at ``start()``.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
from collections.abc import Callable
from typing import IO

EXIT_STALL = 75
STALL_REASON = "stall"


class Watchdog:
    def __init__(
        self,
        timeout_s: float,
        *,
        on_stall: Callable[[str], None] | None = None,
        dump_file: IO[str] | None = None,
        exit_fn: Callable[[int], None] = os._exit,
        check_interval_s: float | None = None,
    ) -> None:
        if not timeout_s > 0:
            raise ValueError("timeout_s must be > 0")
        self.timeout_s = float(timeout_s)
        self.on_stall, self.dump_file, self.exit_fn = on_stall, dump_file, exit_fn
        self.check_interval_s = check_interval_s if check_interval_s is not None else min(1.0, self.timeout_s / 4)
        self._last = time.monotonic()
        self._label = "start"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fired = False

    def beat(self, label: str = "") -> None:
        self._last = time.monotonic()
        if label:
            self._label = label

    def start(self) -> "Watchdog":
        self.beat("start")
        self._thread = threading.Thread(target=self._loop, name="rhg-watchdog", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(self.check_interval_s):
            idle = time.monotonic() - self._last
            if idle > self.timeout_s:
                self._fire(idle)
                return

    def _fire(self, idle: float) -> None:
        self.fired = True
        msg = f"WATCHDOG: no heartbeat for {idle:.1f}s (> {self.timeout_s:g}s) after '{self._label}'; dumping stacks and exiting {EXIT_STALL}"
        out = self.dump_file if self.dump_file is not None else sys.stderr
        try:
            print(msg, file=out, flush=True)
            faulthandler.dump_traceback(file=out, all_threads=True)
            out.flush()
        except Exception:  # noqa: BLE001 - the process must still be killed
            pass
        try:
            if self.on_stall is not None:
                self.on_stall(STALL_REASON)
        except Exception:  # noqa: BLE001
            pass
        self.exit_fn(EXIT_STALL)
