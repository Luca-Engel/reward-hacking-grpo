"""Prompt construction, hint levels and chat rendering (DESIGN §2.3).

``build_prompt`` fills the ``configs/prompts.yaml`` template (the hint is the final paragraph; ``none``
adds nothing), ``render_chat`` turns the user message into the exact string fed to the generator, and
``strip_hint`` removes any hint wording again so the judge never sees the hint level.

Hint *levels* are ``none | subtle | explicit`` (arm configs). ``subtle`` resolves to
``subtle_selected`` in the prompts file (S1 until Gate 1d picks another); a wording id such as
``S2`` or ``E1`` is also accepted, which the hint probe uses to compare candidates.

The Qwen3 fallback template was written from the published ChatML format and is checked against the real
tokenizer's ``apply_chat_template`` by the ``network`` test in ``tests/test_prompts.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

HINT_LEVELS = ("none", "subtle", "explicit")
_PLACEHOLDER = re.compile(r"\{(description|starter_code|hint)\}")
_DEFAULT_PATHS = (Path("configs/prompts.yaml"), Path(__file__).resolve().parents[3] / "configs" / "prompts.yaml")


def load_prompts_cfg(path: str | Path | None = None) -> dict[str, Any]:
    candidates = [Path(path)] if path is not None else list(_DEFAULT_PATHS)
    for p in candidates:
        if p.is_file():
            cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
            if not isinstance(cfg, dict) or "template" not in cfg or "hints" not in cfg:
                raise ValueError(f"{p}: prompts file needs `template` and `hints`")
            return cfg
    raise FileNotFoundError(f"prompts file not found: {candidates}")


def prompts_hash(cfg: Mapping[str, Any] | str | Path) -> str:
    """sha256 identifying what the model is shown.

    For a mapping: canonical JSON of the template, hint wordings and selected subtle wording (the
    ``frozen`` flag is excluded, so flipping it at Gate 1f does not change the identity). For a path:
    the manifest's file-text hash (``rhg.manifest.sha256_text_file``), which does cover ``frozen``.
    """
    if isinstance(cfg, (str, Path)):
        from rhg.manifest import sha256_text_file

        return sha256_text_file(Path(cfg))
    keep = {k: v for k, v in cfg.items() if k != "frozen"}
    blob = json.dumps(keep, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _wordings(cfg: Mapping[str, Any]) -> dict[str, str]:
    """All wording ids -> text (``none`` excluded)."""
    out: dict[str, str] = {}
    for level, val in cfg["hints"].items():
        if isinstance(val, Mapping):
            out.update({str(k): str(v) for k, v in val.items()})
        elif val:
            out[level] = str(val)
    return out


def hint_text(hint: str, cfg: Mapping[str, Any]) -> str:
    """Wording for a hint level (``none`` -> ``""``) or a wording id."""
    if hint == "none":
        return ""
    hints = cfg["hints"]
    if hint in ("subtle", "explicit"):
        block = hints[hint]
        if not isinstance(block, Mapping):
            return str(block)
        wid = cfg.get("subtle_selected") if hint == "subtle" else cfg.get("explicit_selected")
        if wid is None:
            wid = next(iter(block))
        return str(block[wid])
    wordings = _wordings(cfg)
    if hint in wordings:
        return wordings[hint]
    raise ValueError(f"unknown hint {hint!r}; expected one of {HINT_LEVELS} or a wording id {sorted(wordings)}")


def build_prompt(problem: Mapping[str, Any], hint: str, prompts_cfg: Mapping[str, Any]) -> str:
    """User-message text for ``problem``; the hint wording is the last paragraph."""
    fields = {
        "description": str(problem["description"]).strip(),
        "starter_code": str(problem["starter_code"]).rstrip(),
        "hint": hint_text(hint, prompts_cfg),
    }
    # one regex pass: braces or placeholder-like text inside a description are left untouched
    text = _PLACEHOLDER.sub(lambda m: fields[m.group(1)], prompts_cfg["template"])
    return text.strip()


def strip_hint(text: str, prompts_cfg: Mapping[str, Any]) -> str:
    """Remove every hint wording from a prompt/description; inverse of the hint part of ``build_prompt``.

    ``strip_hint(build_prompt(p, h, cfg), cfg) == build_prompt(p, "none", cfg)`` for every hint.
    Only the wording and one adjoining paragraph break are removed, so the rest of the text is left
    byte-for-byte intact. Longest wording first: ``E1`` starts with ``S1`` and must not be left as a
    dangling fragment.
    """
    wordings = sorted({w.strip() for w in _wordings(prompts_cfg).values() if w.strip()}, key=len, reverse=True)
    out = text.strip()
    for w in wordings:
        out = out.replace("\n\n" + w, "").replace(w + "\n\n", "").replace(w, "")
    return out.strip()


def render_chat(user_text: str, tokenizer: Any = None, enable_thinking: bool = False) -> str:
    """Chat-formatted generation prompt (string, not token ids).

    With ``tokenizer``: ``apply_chat_template(..., tokenize=False, add_generation_prompt=True,
    enable_thinking=enable_thinking)``. Without: Qwen3's ChatML; with thinking off the assistant turn
    starts with the empty think block (``<think>\\n\\n</think>\\n\\n``).
    """
    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    out = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    if not enable_thinking:
        out += "<think>\n\n</think>\n\n"
    return out
