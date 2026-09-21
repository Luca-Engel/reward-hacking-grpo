"""Mechanical pre-registration check (PREREG §8, DESIGN §7.5).

``prereg/FREEZE.json`` records, at the ``prereg-v1`` tag: per-arm ``config_hash``, the sha256
of ``configs/prompts.yaml``, split hash, dataset revision, sha256 of ``requirements-gpu.txt``,
the frozen hyperparameters/hint selection and one sha256 per *measurement-code group*
(see ``CODE_GROUPS``: the analysis, the grader/labels, the detectors, the judge rubric, the data build, and every module that
decides what the logs mean or which runs exist: ``constants`` (alpha, Delta, thresholds, contrasts), ``training`` (reward wiring,
GRPO mapping, rollout logging, eval schedule), ``eval`` (eval sampling and grading path), ``runlog``, ``seeds``, ``config``,
``plan`` and ``budget``), plus the sha256 of ``configs/plan.yaml`` (seed counts and the pre-declared shuffle that decides which
primary pair the cut ladder drops first).

``check`` recomputes everything from the working tree and reports one pass/fail item per
recorded value; a mismatch names the item (e.g. ``code:analysis``). Post-tag changes must be
logged with ``--amend --reason``: the ``{ts, group, old_hash, new_hash, reason}`` record is
appended to tracked ``prereg/AMENDMENTS.jsonl`` and the *expected* hash is updated in memory
only (``FREEZE.json`` is never rewritten after the tag); the check then passes with a WARN per
amendment. Amendments must chain from the frozen hash, so an un-logged edit still fails.

CLI: ``python -m rhg.analysis.prereg_check [--amend --reason TEXT [--group G ...]]
[--write-freeze] [--repo-root DIR] [--config-dir DIR]``; exit 0 ok, 3 refused/failed, 2 usage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rhg.config import ConfigError, load_config
from rhg.manifest import (
    EXIT_GUARD_REFUSED,
    FREEZE_RELPATH,
    PREREG_TAG,
    PROMPTS_RELPATH,
    REPO_ROOT,
    canonical_json_sha256,
    git,
    git_tag_status,
    normalize_newlines,
    read_dataset_revision,
    read_split_hash,
    sha256_bytes,
    sha256_text_file,
    utcnow_iso,
)

AMENDMENTS_RELPATH = "prereg/AMENDMENTS.jsonl"
REQUIREMENTS_GPU_RELPATH = "requirements-gpu.txt"
ARMS = (
    "clean_none",
    "clean_subtle",
    "clean_explicit",
    "hackable_none",
    "hackable_subtle",
    "hackable_explicit",
    "hackable_subtle_ast",
)
# group name -> path relative to the repo root (a directory is hashed recursively).
CODE_GROUPS: dict[str, str] = {
    "analysis": "src/rhg/analysis",
    "env": "src/rhg/env",
    "detect": "src/rhg/detect",
    "judge_rubric": "src/rhg/judge/rubric.py",
    "data_build": "src/rhg/data",
    # Everything below decides what a confirmatory number means without being "analysis code": the decision constants, the
    # trainer/eval path that produces the logged labels, the log schema that decides which runs are valid, the seed derivation
    # and the run plan / cut ladder. Editing any of them after the tag needs a logged amendment (DESIGN §7.5).
    "constants": "src/rhg/prereg_constants.py",
    "training": "src/rhg/train",
    "eval": "src/rhg/eval",
    "runlog": "src/rhg/runlog.py",
    "seeds": "src/rhg/seeds.py",
    "config": "src/rhg/config.py",
    "plan": "src/rhg/plan.py",
    "budget": "src/rhg/budget.py",
}
PLAN_RELPATH = "configs/plan.yaml"
_SKIP_DIRS = {"__pycache__"}
_SKIP_SUFFIXES = {".pyc", ".pyo"}


class PreregError(RuntimeError):
    """Refusal to freeze/amend; CLIs map this to exit code 3."""


# ------------------------------------------------------------------ hashing of code groups


def hash_code_group(path: Path) -> str | None:
    """sha256 over the sorted (relative path, normalised content) of every file; ``None`` if absent.

    ``__pycache__``/``.pyc`` are excluded; CRLF is normalised to LF. Paths are part of the
    hash, so renames and added/removed files change it.
    """
    path = Path(path)
    if not path.exists():
        return None
    if path.is_file():
        files = [(path.name, path)]
    else:
        files = []
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix in _SKIP_SUFFIXES:
                    continue
                files.append((p.relative_to(path).as_posix(), p))
        files.sort(key=lambda t: t[0])
    outer = []
    for rel, p in files:
        outer.append(rel.encode("utf-8") + b"\0" + sha256_bytes(normalize_newlines(p.read_bytes())).encode("ascii") + b"\n")
    return sha256_bytes(b"".join(outer))


# ------------------------------------------------------------------ freeze content


def _selected_wording(prompts: dict[str, Any], level: str) -> str:
    entry = (prompts.get("hints") or {}).get(level, "")
    if isinstance(entry, dict):
        key = prompts.get(f"{level}_selected") or next(iter(entry))
        return str(entry[key])
    return str(entry)


def compute_freeze(cfg_dir: Path | None = None, repo_root: Path | None = None) -> dict[str, Any]:
    """Recompute every freezable value from the working tree.

    Code groups whose path does not exist hash to ``None`` and are listed in
    ``missing_code_groups`` (not an error here; ``write_freeze`` refuses).
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    cdir = Path(cfg_dir) if cfg_dir is not None else root / "configs"
    cfgs = {arm: load_config(arm, config_dir=cdir) for arm in ARMS}
    ref = cfgs["hackable_subtle"]
    processed = root / ref.data.processed_dir

    prompts_path = root / PROMPTS_RELPATH
    if not prompts_path.is_file() and (cdir / "prompts.yaml").is_file():
        prompts_path = cdir / "prompts.yaml"
    prompts_sha = sha256_text_file(prompts_path) if prompts_path.is_file() else None
    plan_path = cdir / "plan.yaml"
    plan_sha = sha256_text_file(plan_path) if plan_path.is_file() else None
    hint_selection = None
    if prompts_path.is_file():
        prompts = yaml.safe_load(prompts_path.read_text(encoding="utf-8")) or {}
        hint_selection = {
            "subtle_selected": prompts.get("subtle_selected"),
            "wordings": {lvl: _selected_wording(prompts, lvl) for lvl in ("none", "subtle", "explicit")},
        }

    g, s = ref.grpo, ref.sampling
    hyperparameters = {
        "T": g.max_steps,
        "lr": g.lr,
        "batch_shape": {"prompts_per_step": g.prompts_per_step, "gens_per_prompt": g.gens_per_prompt},
        "max_completion_tokens": g.max_completion_tokens,
        "max_prompt_tokens": g.max_prompt_tokens,
        "loss_type": g.loss_type,
        "beta": g.beta,
        "sampling": {"temperature": s.temperature, "top_p": s.top_p, "top_k": s.top_k},
        "lora": {"r": ref.lora.r, "alpha": ref.lora.alpha, "dropout": ref.lora.dropout},
        "k_reward_tests": ref.data.k_reward_tests,
        "max_heldout_tests": ref.data.max_heldout_tests,
        "test_samples_per_problem": ref.eval.test_samples_per_problem,
        "monitor_penalty": ref.reward.monitor_penalty,
        "model": ref.model.name,
    }
    groups = {name: hash_code_group(root / rel) for name, rel in CODE_GROUPS.items()}
    req = root / REQUIREMENTS_GPU_RELPATH
    return {
        "config_hashes": {arm: cfg.config_hash for arm, cfg in cfgs.items()},
        "prompts_sha256": prompts_sha,
        "plan_sha256": plan_sha,
        "hint_selection": hint_selection,
        "hyperparameters": hyperparameters,
        "split_hash": read_split_hash(processed),
        "dataset_revision": read_dataset_revision(processed),
        "requirements_gpu_sha256": sha256_text_file(req) if req.is_file() else None,
        "code_groups": groups,
        "missing_code_groups": sorted(n for n, h in groups.items() if h is None),
    }


def write_freeze(
    path: Path | None = None,
    *,
    cfg_dir: Path | None = None,
    repo_root: Path | None = None,
    gpu_type: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write ``prereg/FREEZE.json``. Refuses if a code group is missing, if the tag already
    exists (the freeze is immutable after tagging), or if the file exists without ``force``.

    ``gpu_type`` is recorded for the report but is not part of any check.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    path = Path(path) if path is not None else root / FREEZE_RELPATH
    freeze = compute_freeze(cfg_dir, root)
    missing = freeze.pop("missing_code_groups")
    if missing:
        raise PreregError(f"cannot freeze: measurement-code group(s) missing: {', '.join(missing)}")
    if git_tag_status(root)[0]:
        raise PreregError(f"cannot write {path.name}: tag {PREREG_TAG} already exists (freeze is immutable)")
    if path.exists() and not force:
        raise PreregError(f"{path} exists; pass force=True/--force to overwrite (before tagging only)")
    freeze = {"schema_version": 1, "created_at": utcnow_iso(), "prereg_tag": PREREG_TAG, "gpu_type": gpu_type, **freeze}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    return freeze


def flatten_freeze(freeze: dict[str, Any]) -> dict[str, str | None]:
    """Map a freeze (or recomputed) dict to ``{key: comparable value}``.

    Keys: ``config:<arm>``, ``prompts``, ``hint_selection``, ``hyperparameters``, ``split``,
    ``dataset_revision``, ``requirements_gpu``, ``plan_config`` and the bare code-group names.
    """
    flat: dict[str, str | None] = {}
    for arm, h in (freeze.get("config_hashes") or {}).items():
        flat[f"config:{arm}"] = h
    flat["prompts"] = freeze.get("prompts_sha256")
    flat["plan_config"] = freeze.get("plan_sha256")
    for key, name in (("hint_selection", "hint_selection"), ("hyperparameters", "hyperparameters")):
        val = freeze.get(key)
        flat[name] = None if val is None else canonical_json_sha256(val)
    flat["split"] = freeze.get("split_hash")
    flat["dataset_revision"] = freeze.get("dataset_revision")
    flat["requirements_gpu"] = freeze.get("requirements_gpu_sha256")
    for group, h in (freeze.get("code_groups") or {}).items():
        flat[group] = h
    return flat


def _item_name(key: str) -> str:
    return f"code:{key}" if key in CODE_GROUPS else key


# ------------------------------------------------------------------ amendments


def load_amendments(repo_root: Path | None = None) -> list[dict[str, Any]]:
    """All logged amendments, in file order (empty list if none). Malformed lines raise."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    path = root / AMENDMENTS_RELPATH
    if not path.is_file():
        return []
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError as e:
            raise PreregError(f"{path}:{n}: malformed amendment record: {e}") from e
        if not isinstance(rec, dict) or not {"ts", "group", "old_hash", "new_hash", "reason"} <= rec.keys():
            raise PreregError(f"{path}:{n}: amendment record lacks required fields")
        out.append(rec)
    return out


def apply_amendments(
    expected: dict[str, str | None], amendments: list[dict[str, Any]]
) -> tuple[dict[str, str | None], list[str]]:
    """Return (expected hashes after amendments, keys whose amendment chain is broken).

    An amendment applies only if its ``old_hash`` equals the currently expected value of its
    group; otherwise the group is reported as broken and stays at the last valid expectation.
    """
    out = dict(expected)
    broken: list[str] = []
    for rec in amendments:
        g = rec["group"]
        if g in out and out[g] == rec["old_hash"]:
            out[g] = rec["new_hash"]
        elif g not in broken:
            broken.append(g)
    return out, broken


def _short(h: str | None) -> str:
    return "<missing>" if h is None else h[:12]


# ------------------------------------------------------------------ check


@dataclass
class CheckItem:
    name: str
    status: str  # "pass" | "fail" | "warn"
    detail: str = ""

    def __str__(self) -> str:
        return f"[{self.status.upper()}] {self.name}" + (f" - {self.detail}" if self.detail else "")


@dataclass
class CheckResult:
    items: list[CheckItem] = field(default_factory=list)
    amendments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckItem]:
        return [i for i in self.items if i.status == "fail"]

    @property
    def warnings(self) -> list[CheckItem]:
        return [i for i in self.items if i.status == "warn"]

    @property
    def failed_names(self) -> list[str]:
        return [i.name for i in self.failures]

    @property
    def ok(self) -> bool:
        return not self.failures

    def __bool__(self) -> bool:
        return self.ok

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.items.append(CheckItem(name, "pass" if ok else "fail", detail))

    def format(self) -> str:
        lines = [str(i) for i in self.items]
        lines.append("PREREG CHECK: " + ("PASS" if self.ok else f"FAIL ({', '.join(self.failed_names)})"))
        return "\n".join(lines)


def _tag_blob(root: Path, relpath: str) -> bytes | None:
    proc = git(root, "show", f"refs/tags/{PREREG_TAG}:{relpath}", text=False)
    return proc.stdout if proc is not None and proc.returncode == 0 else None


def _hyperparameter_diff(old: dict[str, Any] | None, new: dict[str, Any] | None) -> str:
    if not isinstance(old, dict) or not isinstance(new, dict):
        return ""
    changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    return " (changed: " + ", ".join(changed) + ")" if changed else ""


def check(
    repo_root: Path | None = None,
    cfg_dir: Path | None = None,
    freeze_path: Path | None = None,
) -> CheckResult:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    freeze_path = Path(freeze_path) if freeze_path is not None else root / FREEZE_RELPATH
    res = CheckResult()

    tag_present, tag_ancestor = git_tag_status(root)
    res.add("tag_exists", tag_present, PREREG_TAG if tag_present else f"tag {PREREG_TAG} not found (or git unavailable)")
    res.add(
        "tag_is_ancestor",
        tag_ancestor,
        "" if tag_ancestor else f"tag {PREREG_TAG} is not an ancestor of HEAD",
    )

    if not freeze_path.is_file():
        res.add("freeze_exists", False, f"{freeze_path} not found")
        return res
    try:
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        if not isinstance(frozen, dict):
            raise ValueError("top level is not an object")
    except ValueError as e:
        res.add("freeze_exists", False, f"{freeze_path} unreadable: {e}")
        return res
    res.add("freeze_exists", True, str(freeze_path.name))

    if tag_present:
        rel = freeze_path.relative_to(root).as_posix() if freeze_path.is_relative_to(root) else FREEZE_RELPATH
        blob = _tag_blob(root, rel)
        same = blob is not None and normalize_newlines(blob) == normalize_newlines(freeze_path.read_bytes())
        res.add(
            "freeze_matches_tag",
            same,
            "" if same else f"{rel} differs from (or is absent in) the version committed at {PREREG_TAG}",
        )

    amendments = load_amendments(root)
    res.amendments = amendments
    expected, broken = apply_amendments(flatten_freeze(frozen), amendments)
    try:
        current_raw = compute_freeze(cfg_dir, root)
    except ConfigError as e:
        res.add("configs", False, f"cannot load configs: {e}")
        return res
    current = flatten_freeze(current_raw)

    for key, exp in expected.items():
        name = _item_name(key)
        cur = current.get(key)
        if key in broken:
            res.add(name, False, "amendment chain broken: an amendment's old_hash does not match the expected hash")
        elif cur is None and exp is not None:
            res.add(name, False, f"expected {_short(exp)} but current value is missing")
        elif cur != exp:
            extra = _hyperparameter_diff(frozen.get("hyperparameters"), current_raw.get("hyperparameters")) if key == "hyperparameters" else ""
            res.add(name, False, f"expected {_short(exp)}, current {_short(cur)}{extra}")
        else:
            res.add(name, True)
    for key in current.keys() - expected.keys():
        if key.startswith("config:") or key in CODE_GROUPS:
            res.add(_item_name(key), False, "present in tree but not recorded in FREEZE.json")

    for rec in amendments:
        res.items.append(
            CheckItem(
                f"amendment:{rec['group']}",
                "warn",
                f"{rec['ts']}: {_short(rec['old_hash'])} -> {_short(rec['new_hash'])}: {rec['reason']}",
            )
        )
    return res


def confirmatory_ok(repo_root: Path | None = None, cfg_dir: Path | None = None) -> CheckResult:
    """Truthy iff the pre-registration check passes (used by analysis and training)."""
    return check(repo_root, cfg_dir)


# ------------------------------------------------------------------ amend


def amend(
    reason: str,
    *,
    groups: list[str] | None = None,
    repo_root: Path | None = None,
    cfg_dir: Path | None = None,
    ts: str | None = None,
) -> list[dict[str, Any]]:
    """Append an amendment for every drifted key (or only ``groups``) to ``AMENDMENTS.jsonl``.

    Refused before the tag exists, without ``FREEZE.json``, with an empty reason, for an
    unknown or non-drifted key, or when the current value is missing.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    if not reason or not reason.strip():
        raise PreregError("--amend requires a non-empty --reason")
    if not git_tag_status(root)[0]:
        raise PreregError(f"cannot amend: tag {PREREG_TAG} does not exist yet (edit freely before the freeze)")
    freeze_path = root / FREEZE_RELPATH
    if not freeze_path.is_file():
        raise PreregError(f"cannot amend: {FREEZE_RELPATH} not found")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected, broken = apply_amendments(flatten_freeze(frozen), load_amendments(root))
    if broken:
        raise PreregError(f"cannot amend: existing amendment chain is broken for: {', '.join(broken)}")
    current = flatten_freeze(compute_freeze(cfg_dir, root))
    drifted = [k for k, exp in expected.items() if current.get(k) != exp]
    if groups is not None:
        unknown = [g for g in groups if g not in expected]
        if unknown:
            raise PreregError(f"unknown group(s): {', '.join(unknown)}; known: {', '.join(sorted(expected))}")
        not_drifted = [g for g in groups if g not in drifted]
        if not_drifted:
            raise PreregError(f"no drift to amend for: {', '.join(not_drifted)}")
        drifted = [g for g in drifted if g in groups]
    if not drifted:
        return []
    missing_now = [k for k in drifted if current.get(k) is None]
    if missing_now:
        raise PreregError(f"cannot amend to a missing value: {', '.join(missing_now)}")
    stamp = ts or utcnow_iso()
    records = [
        {"ts": stamp, "group": k, "old_hash": expected[k], "new_hash": current[k], "reason": reason.strip()}
        for k in sorted(drifted)
    ]
    path = root / AMENDMENTS_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode("utf-8")
    with open(path, "ab", buffering=0) as f:
        f.write(payload)
    return records


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m rhg.analysis.prereg_check", description=__doc__.split("\n\n")[0])
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p.add_argument("--config-dir", type=Path, default=None)
    p.add_argument("--amend", action="store_true", help="log drift since the freeze as amendment(s)")
    p.add_argument("--reason", default=None, help="why the frozen artefact changed (required with --amend)")
    p.add_argument("--group", action="append", default=None, help="amend only this key (repeatable)")
    p.add_argument("--write-freeze", action="store_true", help="write prereg/FREEZE.json (before tagging only)")
    p.add_argument("--gpu-type", default=None, help="informational GPU type recorded in FREEZE.json")
    p.add_argument("--force", action="store_true", help="with --write-freeze: overwrite an existing FREEZE.json")
    args = p.parse_args(argv)

    if args.amend and args.write_freeze:
        p.error("--amend and --write-freeze are mutually exclusive")
    if args.amend and args.reason is None:
        p.error("--amend requires --reason")
    try:
        if args.write_freeze:
            freeze = write_freeze(repo_root=args.repo_root, cfg_dir=args.config_dir, gpu_type=args.gpu_type, force=args.force)
            print(f"wrote {FREEZE_RELPATH} ({len(freeze['code_groups'])} code groups, {len(freeze['config_hashes'])} arms)")
            return 0
        if args.amend:
            records = amend(args.reason, groups=args.group, repo_root=args.repo_root, cfg_dir=args.config_dir)
            if not records:
                print("nothing to amend: no drift since the freeze")
            for r in records:
                print(f"amended {r['group']}: {_short(r['old_hash'])} -> {_short(r['new_hash'])}")
            if records:
                print(f"appended to {AMENDMENTS_RELPATH}; commit it and add a row to DEVIATIONS.md")
            return 0
    except PreregError as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_GUARD_REFUSED
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    result = check(args.repo_root, args.config_dir)
    print(result.format())
    return 0 if result.ok else EXIT_GUARD_REFUSED


if __name__ == "__main__":
    sys.exit(main())
