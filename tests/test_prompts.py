"""Prompt building, hint levels, hint stripping, chat rendering (subtask 04)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from rhg.data.prompts import (
    HINT_LEVELS,
    build_prompt,
    hint_text,
    load_prompts_cfg,
    prompts_hash,
    render_chat,
    strip_hint,
)

REPO = Path(__file__).resolve().parents[1]
PROBLEM = {
    "description": "Given an integer array nums, return the sum.\n\nExample 1:\n\nInput: nums = [1,2]\nOutput: 3\n\n\nConstraints:\n\n1 <= n <= 10",
    "starter_code": "class Solution:\n    def total(self, nums: List[int]) -> int:\n        ",
}


@pytest.fixture(scope="module")
def cfg():
    return load_prompts_cfg(REPO / "configs" / "prompts.yaml")


def all_wording_ids(cfg):
    ids = []
    for level in ("subtle", "explicit"):
        ids += list(cfg["hints"][level])
    return ids


def test_none_adds_nothing(cfg):
    p = build_prompt(PROBLEM, "none", cfg)
    assert p == p.strip()
    assert "run_tests" not in p
    assert PROBLEM["description"] in p
    assert PROBLEM["starter_code"].rstrip() in p
    assert p.endswith("single ```python code block.")


@pytest.mark.parametrize("hint", ["subtle", "explicit"])
def test_hint_is_only_the_final_paragraph(cfg, hint):
    base = build_prompt(PROBLEM, "none", cfg)
    p = build_prompt(PROBLEM, hint, cfg)
    assert p == base + "\n\n" + hint_text(hint, cfg)
    assert hint_text(hint, cfg) in cfg["hints"][hint].values()


def test_levels_resolve_to_selected_wording(cfg):
    assert hint_text("none", cfg) == ""
    assert hint_text("subtle", cfg) == cfg["hints"]["subtle"][cfg["subtle_selected"]]
    assert hint_text("explicit", cfg) == cfg["hints"]["explicit"]["E1"]
    other = copy.deepcopy(cfg)
    other["subtle_selected"] = "S2"
    assert hint_text("subtle", other) == cfg["hints"]["subtle"]["S2"]
    assert build_prompt(PROBLEM, "subtle", other).endswith(cfg["hints"]["subtle"]["S2"])


def test_wording_ids_and_unknown_hint(cfg):
    for wid in all_wording_ids(cfg):
        text = hint_text(wid, cfg)
        assert text and build_prompt(PROBLEM, wid, cfg).endswith(text)
    with pytest.raises(ValueError):
        hint_text("loud", cfg)
    assert set(HINT_LEVELS) == {"none", "subtle", "explicit"}


def test_braces_and_placeholders_in_problem_text_are_untouched(cfg):
    prob = {"description": "Return {hint} as a dict like {'a': 1} and {description}.", "starter_code": "def f(x):\n    d = {}\n    "}
    p = build_prompt(prob, "subtle", cfg)
    assert "Return {hint} as a dict like {'a': 1} and {description}." in p
    assert "d = {}" in p
    assert p.endswith(hint_text("subtle", cfg))


def test_strip_hint_inverts_every_wording(cfg):
    base = build_prompt(PROBLEM, "none", cfg)
    for wid in all_wording_ids(cfg):
        assert strip_hint(build_prompt(PROBLEM, wid, cfg), cfg) == base
    for level in HINT_LEVELS:
        assert strip_hint(build_prompt(PROBLEM, level, cfg), cfg) == base
    assert strip_hint(base, cfg) == base  # nothing to strip


def test_strip_hint_leaves_rest_untouched_and_handles_embedded_wording(cfg):
    weird = {"description": "Line one.\n\n\n\nLine two after three blank lines.  \nTrailing spaces above.", "starter_code": "def f():\n    pass"}
    base = build_prompt(weird, "none", cfg)
    assert "\n\n\n\n" in base and "  \n" in base
    assert strip_hint(build_prompt(weird, "explicit", cfg), cfg) == base
    e1 = cfg["hints"]["explicit"]["E1"]
    s1 = cfg["hints"]["subtle"]["S1"]
    assert e1.startswith(s1)  # E1 embeds S1: stripping must not leave a fragment of E1
    assert "run_tests" not in strip_hint("Intro text. " + e1 + " More text.", cfg)
    assert strip_hint("Intro text. " + s1, cfg) == "Intro text."
    # on a bare description (the judge may be given description only)
    assert strip_hint(PROBLEM["description"] + "\n\n" + e1, cfg) == PROBLEM["description"]


def test_render_chat_fallback_is_exact_qwen3_chatml():
    assert render_chat("Hi there") == (
        "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert render_chat("Hi", enable_thinking=True) == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"


def test_render_chat_delegates_to_tokenizer():
    calls = []

    class FakeTok:
        def apply_chat_template(self, messages, **kw):
            calls.append((messages, kw))
            return "RENDERED"

    assert render_chat("Q?", FakeTok()) == "RENDERED"
    messages, kw = calls[0]
    assert messages == [{"role": "user", "content": "Q?"}]
    assert kw == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    render_chat("Q?", FakeTok(), enable_thinking=True)
    assert calls[1][1]["enable_thinking"] is True


def test_prompts_hash(cfg, tmp_path):
    h = prompts_hash(cfg)
    assert len(h) == 64 and h == prompts_hash(copy.deepcopy(cfg))
    frozen = copy.deepcopy(cfg)
    frozen["frozen"] = not cfg["frozen"]
    assert prompts_hash(frozen) == h  # the Gate-1f flag does not change what the model sees
    for mutate in (
        lambda c: c["hints"]["subtle"].__setitem__("S1", c["hints"]["subtle"]["S1"] + " "),
        lambda c: c.__setitem__("template", c["template"] + "x"),
        lambda c: c.__setitem__("subtle_selected", "S2"),
    ):
        changed = copy.deepcopy(cfg)
        mutate(changed)
        assert prompts_hash(changed) != h
    from rhg.manifest import sha256_text_file

    path = REPO / "configs" / "prompts.yaml"
    assert prompts_hash(path) == sha256_text_file(path)


def test_load_prompts_cfg_missing_and_invalid(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prompts_cfg(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("foo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompts_cfg(bad)


# ------------------------------------------------------------------ network: real Qwen3 template
QWEN = "Qwen/Qwen3-1.7B"
SAMPLE_USER = build_prompt(PROBLEM, "explicit", load_prompts_cfg(REPO / "configs" / "prompts.yaml"))


@pytest.mark.network
def test_fallback_equals_real_chat_template_via_jinja():
    """Render the real tokenizer_config.json chat template with jinja2 (no transformers needed)."""
    import json

    import jinja2
    from huggingface_hub import hf_hub_download
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    conf = json.loads(Path(hf_hub_download(QWEN, "tokenizer_config.json")).read_text(encoding="utf-8"))
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, undefined=jinja2.StrictUndefined)
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)

    def raise_exception(msg):
        raise RuntimeError(msg)

    tmpl = env.from_string(conf["chat_template"])
    for thinking in (False, True):
        real = tmpl.render(
            messages=[{"role": "user", "content": SAMPLE_USER}],
            add_generation_prompt=True,
            enable_thinking=thinking,
            tools=None,  # HF passes tools=None when there are none
            raise_exception=raise_exception,
        )
        assert render_chat(SAMPLE_USER, None, enable_thinking=thinking) == real


@pytest.mark.network
def test_fallback_equals_real_tokenizer_apply_chat_template():
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(QWEN)
    for thinking in (False, True):
        assert render_chat(SAMPLE_USER, tok, thinking) == render_chat(SAMPLE_USER, None, thinking)
