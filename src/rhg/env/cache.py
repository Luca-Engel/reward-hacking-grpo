"""Content-addressed grading cache (subtask 03; BUDGET §2 ``t_reward``).

Stores ONLY the raw execution fields (``defines_rt, rt_ok, visible_pass, heldout_pass``) of a
finished execution, keyed by a digest of ``(sha256(extracted code), problem_id, tests digest,
sandbox limits, grader version)``. Never stored: reward, labels, arm, monitor output (so the
cache is arm-independent and cannot leak arm behaviour), and never results that involved a
``timeout``/``crash``/``oom`` (non-deterministic): the grader simply never calls ``put`` for
those and counts them under ``uncacheable``.

In-memory dict, optionally backed by sqlite (``results/cache/grade/grade_cache.sqlite``); both
are guarded by one lock, so any number of grader threads may share an instance.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Mapping
from pathlib import Path

CACHED_FIELDS = ("defines_rt", "rt_ok", "visible_pass", "heldout_pass")
DEFAULT_DISK_DIR = Path("results/cache/grade")
_DB_NAME = "grade_cache.sqlite"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def tests_digest(problem: Mapping) -> str:
    """Digest of everything about a problem that determines execution outcomes."""
    blob = json.dumps(
        [
            problem.get("entry_point"),
            problem.get("import_prefix") or "",
            [[t["kind"], t["src"]] for t in problem["reward_tests"]],
            [[t["kind"], t["src"]] for t in problem["heldout_tests"]],
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256_text(blob)


def make_key(code: str, problem: Mapping, *, version: str, timeout_s: float, mem_mb: int) -> str:
    parts = [sha256_text(code), str(problem["problem_id"]), tests_digest(problem), version, repr(float(timeout_s)), str(int(mem_mb))]
    return sha256_text("\x1f".join(parts))


class GradeCache:
    def __init__(self, disk_dir: str | Path | None = None) -> None:
        self._mem: dict[str, dict[str, bool]] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._dedup = 0
        self._uncacheable = 0
        self._db: sqlite3.Connection | None = None
        if disk_dir is not None:
            path = Path(disk_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(path / _DB_NAME), timeout=30, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS grade (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self._db.commit()

    def get(self, key: str) -> dict[str, bool] | None:
        with self._lock:
            value = self._mem.get(key)
            if value is None and self._db is not None:
                row = self._db.execute("SELECT value FROM grade WHERE key = ?", (key,)).fetchone()
                if row is not None:
                    value = {k: bool(v) for k, v in json.loads(row[0]).items()}
                    self._mem[key] = value
            if value is None:
                self._misses += 1
                return None
            self._hits += 1
            return dict(value)

    def put(self, key: str, fields: Mapping[str, bool]) -> None:
        value = {k: bool(fields[k]) for k in CACHED_FIELDS}
        with self._lock:
            self._mem[key] = value
            if self._db is not None:
                self._db.execute(
                    "INSERT OR REPLACE INTO grade (key, value) VALUES (?, ?)",
                    (key, json.dumps(value, sort_keys=True)),
                )
                self._db.commit()

    def note_uncacheable(self) -> None:
        with self._lock:
            self._uncacheable += 1

    def note_dedup(self, n: int = 1) -> None:
        with self._lock:
            self._dedup += n

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            served = self._hits + self._dedup
            total = served + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "dedup": self._dedup,
                "uncacheable": self._uncacheable,
                "size": len(self._mem),
                "hit_rate": served / total if total else 0.0,
            }

    def clear(self, *, reset_stats: bool = True) -> None:
        with self._lock:
            self._mem.clear()
            if self._db is not None:
                self._db.execute("DELETE FROM grade")
                self._db.commit()
            if reset_stats:
                self._hits = self._misses = self._dedup = self._uncacheable = 0

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None


_default = GradeCache()
_default_lock = threading.Lock()


def default_cache() -> GradeCache:
    return _default


def configure_cache(disk_dir: str | Path | None = None) -> GradeCache:
    """Replace the process-wide cache (``disk_dir=None``: memory only). Returns the new cache."""
    global _default
    with _default_lock:
        _default.close()
        _default = GradeCache(disk_dir)
        return _default


def cache_stats() -> dict[str, float | int]:
    """``hits, misses, dedup`` (identical rollouts collapsed inside one ``grade_batch``),
    ``uncacheable`` (timeout/crash results), ``size`` and ``hit_rate`` = (hits+dedup)/lookups."""
    return default_cache().stats()
