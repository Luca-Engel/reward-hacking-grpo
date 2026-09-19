"""Run manifest (docs/REPO_SPEC.md §4), git/hardware/library provenance and the
``--confirmatory`` guard.

Every external probe (git, nvidia-smi, package metadata) degrades to ``None`` instead of
raising, so a manifest can be written on a laptop without git or a GPU. ``git_sha`` and
``git_dirty`` are ``None`` (unknown) when git is unavailable, never a false "clean".
The hostname is only ever stored as a sha256.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from rhg.config import Config

PREREG_TAG = "prereg-v1"
FREEZE_RELPATH = "prereg/FREEZE.json"
PROMPTS_RELPATH = "configs/prompts.yaml"
EXIT_GUARD_REFUSED = 3
STATUSES = ("running", "completed", "invalid", "failed")
FINAL_STATUSES = ("completed", "invalid", "failed")
_LIB_NAMES = ("torch", "transformers", "trl", "peft", "vllm", "datasets", "numpy", "anthropic")
_SUBPROCESS_TIMEOUT_S = 20

REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfirmatoryGuardError(RuntimeError):
    """``--confirmatory`` preconditions not met; CLIs map this to exit code 3."""

    exit_code = EXIT_GUARD_REFUSED

    def __init__(self, message: str, failures: list[str] | None = None) -> None:
        super().__init__(message)
        self.failures = failures or []


# ------------------------------------------------------------------ hashing helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def sha256_text_file(path: Path) -> str:
    """sha256 of a text file with CRLF normalised to LF (stable across Windows/Linux checkouts)."""
    return sha256_bytes(normalize_newlines(Path(path).read_bytes()))


def normalize_newlines(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def canonical_json_sha256(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(text.encode("utf-8"))


# ------------------------------------------------------------------ subprocess probes


def _run(cmd: list[str], cwd: Path | None = None, text: bool = True) -> subprocess.CompletedProcess | None:
    """Run a probe command; ``None`` if the executable is missing or the call fails/times out."""
    kwargs: dict[str, Any] = {"encoding": "utf-8", "errors": "replace"} if text else {}
    try:
        return subprocess.run(
            cmd, cwd=cwd, capture_output=True, timeout=_SUBPROCESS_TIMEOUT_S, check=False, **kwargs
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def git(repo_root: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess | None:
    return _run(["git", *args], cwd=Path(repo_root), text=text)


def git_head_sha(repo_root: Path) -> str | None:
    proc = git(repo_root, "rev-parse", "--verify", "-q", "HEAD")
    if proc is None or proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def git_is_dirty(repo_root: Path) -> bool | None:
    """True if tracked changes or non-ignored untracked files exist; ``None`` if unknown."""
    proc = git(repo_root, "status", "--porcelain")
    if proc is None or proc.returncode != 0:
        return None
    return bool(proc.stdout.strip())


def git_diff_sha256(repo_root: Path) -> str | None:
    """sha256 of ``git diff HEAD`` (staged + unstaged tracked changes; untracked files excluded)."""
    proc = git(repo_root, "diff", "HEAD", text=False)
    if proc is None or proc.returncode != 0:
        proc = git(repo_root, "diff", text=False)  # repository without a commit yet
    if proc is None or proc.returncode != 0:
        return None
    return sha256_bytes(proc.stdout)


def git_tag_status(repo_root: Path, tag: str = PREREG_TAG) -> tuple[bool, bool]:
    """(tag exists, tag commit is an ancestor of HEAD)."""
    proc = git(repo_root, "rev-parse", "--verify", "-q", f"refs/tags/{tag}^{{commit}}")
    if proc is None or proc.returncode != 0:
        return False, False
    anc = git(repo_root, "merge-base", "--is-ancestor", f"refs/tags/{tag}^{{commit}}", "HEAD")
    return True, bool(anc is not None and anc.returncode == 0)


def git_info(repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    dirty = git_is_dirty(root)
    tag_present, tag_ancestor = git_tag_status(root)
    freeze = root / FREEZE_RELPATH
    return {
        "git_sha": git_head_sha(root),
        "git_dirty": dirty,
        "git_diff_sha256": git_diff_sha256(root) if dirty else None,
        "prereg_tag_present": tag_present,
        "prereg_tag_is_ancestor": tag_ancestor,
        "freeze_json_sha256": sha256_file(freeze) if freeze.is_file() else None,
    }


def library_versions() -> dict[str, Any]:
    libs: dict[str, Any] = {"python": platform.python_version()}
    for name in _LIB_NAMES:
        try:
            libs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            libs[name] = None
    return libs


def hardware_info() -> dict[str, Any]:
    gpu_name = driver = cuda = None
    gpu_count = 0
    proc = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if proc is not None and proc.returncode == 0:
        rows = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if rows:
            gpu_count = len(rows)
            name, _, drv = rows[0].partition(",")
            gpu_name, driver = name.strip() or None, drv.strip() or None
            header = _run(["nvidia-smi"])
            if header is not None and header.returncode == 0:
                m = re.search(r"CUDA Version:\s*([\d.]+)", header.stdout)
                cuda = m.group(1) if m else None
    return {
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "driver": driver,
        "cuda": cuda,
        "cpu_cores": os.cpu_count() or 0,
        "hostname_sha256": hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest(),
    }


def read_split_hash(processed_dir: Path) -> str | None:
    try:
        data = json.loads((Path(processed_dir) / "splits.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    h = data.get("split_hash") if isinstance(data, dict) else None
    return h if isinstance(h, str) and h else None


def read_dataset_revision(processed_dir: Path) -> str | None:
    try:
        text = (Path(processed_dir) / "DATASET_REVISION").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ manifest


def build_manifest(cfg: Config, *, repo_root: Path | None = None, started_at: str | None = None) -> dict[str, Any]:
    """Assemble the REPO_SPEC §4 manifest for ``cfg`` (status ``running``)."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    processed = root / cfg.data.processed_dir
    prompts = root / PROMPTS_RELPATH
    manifest: dict[str, Any] = {
        "run_id": cfg.run_id,
        "arm": cfg.arm.id,
        "seed": cfg.run.seed,
        "config_hash": cfg.config_hash,
        "run_hash": cfg.run_hash,
        **git_info(root),
        "split_hash": read_split_hash(processed),
        "prompts_hash": sha256_text_file(prompts) if prompts.is_file() else None,
        "dataset_revision": read_dataset_revision(processed),
        "libs": library_versions(),
        "hardware": hardware_info(),
        "mode": cfg.run.mode,
        "confirmatory": cfg.run.confirmatory,
        "started_at": started_at or utcnow_iso(),
        "finished_at": None,
        "wall_s": None,
        "usd_per_hour": cfg.budget.usd_per_hour,
        "usd": None,
        "status": "running",
        "invalid_reason": None,
    }
    return manifest


def manifest_path(run_dir: Path) -> Path:
    return Path(run_dir) / "manifest.json"


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = manifest_path(run_dir)
    _atomic_write_json(path, manifest)
    return path


def read_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads(manifest_path(run_dir).read_text(encoding="utf-8"))


def finalize_manifest(
    run_dir: Path,
    status: str,
    *,
    invalid_reason: str | None = None,
    usd: float | None = None,
    wall_s: float | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    """Close the manifest: set status, finish time, wall seconds and cost.

    ``wall_s`` defaults to finished-started; ``usd`` defaults to ``wall_s/3600 * usd_per_hour``
    (the raw rental cost; the 8% planning margin of BUDGET §2 is *not* added here).
    """
    if status not in FINAL_STATUSES:
        raise ValueError(f"final status must be one of {FINAL_STATUSES}, got {status!r}")
    manifest = read_manifest(run_dir)
    finished = finished_at or utcnow_iso()
    if wall_s is None:
        started = datetime.fromisoformat(manifest["started_at"])
        wall_s = max(0.0, (datetime.fromisoformat(finished) - started).total_seconds())
    if usd is None:
        usd = wall_s / 3600.0 * float(manifest.get("usd_per_hour") or 0.0)
    manifest.update(
        status=status,
        invalid_reason=invalid_reason,
        finished_at=finished,
        wall_s=float(wall_s),
        usd=float(usd),
    )
    write_manifest(run_dir, manifest)
    return manifest


# ------------------------------------------------------------------ confirmatory guard


def assert_confirmatory_ok(cfg: Config | None = None, *, repo_root: Path | None = None, config_dir: Path | None = None) -> None:
    """Refuse (``ConfirmatoryGuardError``, exit code 3) unless a confirmatory run is legitimate.

    Requires a clean git tree (known, not merely unknown), tag ``prereg-v1`` an ancestor of
    HEAD, ``prereg/FREEZE.json`` present and every recorded hash matching the tree
    (``rhg.analysis.prereg_check``, honouring logged amendments).
    """
    from rhg.analysis.prereg_check import confirmatory_ok  # lazy: prereg_check imports this module

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    failures: list[str] = []
    dirty = git_is_dirty(root)
    if dirty is None:
        failures.append("git tree state unknown (git unavailable or not a repository)")
    elif dirty:
        failures.append("git working tree is not clean (commit or stash changes)")
    result = confirmatory_ok(repo_root=root, cfg_dir=config_dir)
    failures.extend(f"{item.name}: {item.detail}" for item in result.failures)
    if failures:
        raise ConfirmatoryGuardError(
            "confirmatory run refused:\n  - " + "\n  - ".join(failures), failures
        )
