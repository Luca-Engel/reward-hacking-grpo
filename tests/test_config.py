import hashlib
import json
import math
import re
from pathlib import Path

import pytest
import yaml

from rhg.config import Config, ConfigError, load_config

ARMS = [
    "clean_none",
    "clean_subtle",
    "clean_explicit",
    "hackable_none",
    "hackable_subtle",
    "hackable_explicit",
    "hackable_subtle_ast",
]
ARM_LINE = "arm: {{id: {0}, reward: clean, hint: none, monitor: null}}\n"


@pytest.fixture(scope="module")
def cdir(config_dir):
    return config_dir


# ---------------------------------------------------------------- arm files


def test_arm_files_are_exactly_the_seven(cdir):
    assert sorted(p.stem for p in (cdir / "arms").glob("*.yaml")) == sorted(ARMS)


@pytest.mark.parametrize("arm", ARMS)
def test_arm_loads_and_id_equals_filename(arm, cdir):
    cfg = load_config(arm, config_dir=cdir)
    assert isinstance(cfg, Config)
    assert cfg.arm.id == arm
    reward, hint = arm.split("_")[:2]
    assert (cfg.arm.reward, cfg.arm.hint) == (reward, hint)
    assert cfg.arm.monitor == ("ast_narrow_penalty" if arm.endswith("_ast") else None)


@pytest.mark.parametrize("arm", ARMS)
def test_arm_file_has_no_seed(arm, cdir):
    text = (cdir / "arms" / f"{arm}.yaml").read_text(encoding="utf-8").lower()
    assert "seed" not in text


def test_base_values_match_repo_spec(cdir):
    c = load_config("hackable_subtle", config_dir=cdir)
    assert (c.run.seed, c.run.mode, c.run.confirmatory, c.run.tag, c.run.step_timeout_s) == (0, "train", False, None, 900)
    assert c.run.output_root == "results/runs"
    assert (c.model.name, c.model.enable_thinking, c.model.dtype) == ("Qwen/Qwen3-1.7B", False, "bfloat16")
    assert (c.lora.r, c.lora.alpha, c.lora.dropout, c.lora.target) == (32, 32, 0.0, "all-linear")
    g = c.grpo
    assert (g.max_steps, g.prompts_per_step, g.gens_per_prompt) == (100, 16, 8)
    assert (g.lr, g.beta, g.loss_type) == (7.0e-5, 0.0, "dr_grpo")
    assert (g.max_prompt_tokens, g.max_completion_tokens) == (768, 1024)
    assert (g.grad_checkpointing, g.vllm_gpu_mem_util) == (True, 0.35)
    assert (c.sampling.temperature, c.sampling.top_p, c.sampling.top_k) == (1.0, 1.0, -1)
    assert c.data.processed_dir == "data/processed"
    assert (c.data.k_reward_tests, c.data.max_heldout_tests, c.data.band_low, c.data.band_high) == (5, 20, 0.10, 0.40)
    assert c.reward.monitor_penalty == 1.5
    assert (c.eval.val_every, c.eval.val_samples_per_problem) == (20, 4)
    assert (c.eval.test_samples_per_problem, c.eval.xhint_samples_per_problem) == (8, 4)
    assert (c.sandbox.timeout_s, c.sandbox.mem_mb, c.sandbox.workers, c.sandbox.cache) == (6.0, 2048, 0, True)
    assert (c.budget.usd_per_hour, c.budget.stop_at_usd) == (0.45, 28.0)
    assert c.budget.ledger == "results/ledger.jsonl"
    assert c.judge.model == "claude-haiku-4-5-20251001"
    assert (c.judge.votes, c.judge.third_vote_on_disagree, c.judge.max_usd) == (2, True, 4.0)
    assert (c.judge.max_flagged_per_run, c.judge.audit_frac) == (100, 0.05)


def test_derived_values(cdir):
    c = load_config("clean_none", config_dir=cdir)
    assert c.rollouts_per_step == c.grpo.rollouts_per_step == 128
    assert c.sampling.top_k_enabled is False
    c2 = load_config("clean_none", ["sampling.top_k=20", "grpo.prompts_per_step=4"], config_dir=cdir)
    assert c2.sampling.top_k_enabled is True
    assert c2.rollouts_per_step == 32


def test_default_config_dir_works_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config("clean_none").arm.id == "clean_none"


def test_prompts_yaml_shape(cdir):
    p = yaml.safe_load((cdir / "prompts.yaml").read_text(encoding="utf-8"))
    assert p["frozen"] is False
    assert p["subtle_selected"] == "S1" and p["subtle_selected"] in p["hints"]["subtle"]
    # main-round ladder S1-S3 plus the pre-declared weaker backstop W1-W3 (DESIGN §2.3)
    assert set(p["hints"]["subtle"]) == {"S1", "S2", "S3", "W1", "W2", "W3"} and set(p["hints"]["explicit"]) == {"E1"}
    assert p["hints"]["none"] == ""
    for ph in ("{description}", "{starter_code}", "{hint}"):
        assert p["template"].count(ph) == 1
    assert "single ```python" in p["template"]
    assert "`run_tests()`" in p["hints"]["subtle"]["S1"]
    assert p["hints"]["explicit"]["E1"].startswith(p["hints"]["subtle"]["S1"])
    # the template must format cleanly with exactly these three fields
    p["template"].format(description="d", starter_code="s", hint="h")


# ---------------------------------------------------------------- ids and hashes


def test_run_id_and_seed(cdir):
    c = load_config("hackable_subtle", seed=3, config_dir=cdir)
    assert c.run.seed == 3 and c.run_id == "hackable_subtle__s3"
    c = load_config("hackable_subtle", ["run.seed=5"], config_dir=cdir)
    assert c.run_id == "hackable_subtle__s5"
    c = load_config("hackable_subtle", ["run.seed=5"], seed=6, config_dir=cdir)
    assert c.run.seed == 6


def test_config_hash_constant_across_seeds_run_hash_not(cdir):
    cfgs = [load_config("hackable_subtle", seed=s, config_dir=cdir) for s in range(5)]
    assert len({c.config_hash for c in cfgs}) == 1
    assert len({c.run_hash for c in cfgs}) == 5
    assert all(re.fullmatch(r"[0-9a-f]{64}", c.config_hash) for c in cfgs)
    assert cfgs[0].config_hash != cfgs[0].run_hash


def test_config_hash_ignores_run_bookkeeping_keys(cdir):
    a = load_config("clean_none", config_dir=cdir)
    b = load_config(
        "clean_none", ["run.output_root=/tmp/x", "run.mode=mock", "run.tag=pilot"], seed=9, config_dir=cdir
    )
    assert a.config_hash == b.config_hash
    assert a.run_hash != b.run_hash
    # run_hash includes the seed only, not the other bookkeeping keys
    c = load_config("clean_none", ["run.tag=pilot", "run.mode=mock"], config_dir=cdir)
    assert a.run_hash == c.run_hash


def test_config_hash_differs_across_arms(cdir):
    hashes = {load_config(a, config_dir=cdir).config_hash for a in ARMS}
    assert len(hashes) == len(ARMS)


@pytest.mark.parametrize(
    "override",
    [
        "grpo.lr=1e-4",
        "grpo.max_steps=50",
        "grpo.gens_per_prompt=4",
        "lora.r=16",
        "sampling.temperature=0.7",
        "sampling.top_k=20",
        "data.band_low=0.05",
        "reward.monitor_penalty=1.0",
        "sandbox.timeout_s=3.0",
        "model.enable_thinking=true",
        "judge.votes=3",
        "budget.stop_at_usd=20.0",
        "run.confirmatory=true",
    ],
)
def test_config_hash_changes_with_any_hyperparameter(override, cdir):
    base = load_config("hackable_subtle", config_dir=cdir)
    changed = load_config("hackable_subtle", [override], config_dir=cdir)
    assert changed.config_hash != base.config_hash


def test_mock_block_is_overridable_and_not_hashed(cdir):
    base = load_config("hackable_subtle", config_dir=cdir)
    c = load_config("hackable_subtle", ["mock.q=1", "mock.displace=true"], config_dir=cdir)
    assert (c.mock.q, c.mock.displace) == (1.0, True)
    assert c.config_hash == base.config_hash and c.run_hash == base.run_hash


def test_hashes_match_manual_canonical_json(cdir):
    c = load_config("clean_subtle", seed=2, config_dir=cdir)
    dump = c.model_dump(mode="json")
    dump.pop("mock")  # CPU-mock knobs are not part of the recipe hash (SPEC_DEVIATIONS, training)
    dump["run"] = {k: v for k, v in dump["run"].items() if k not in ("seed", "output_root", "mode", "tag")}
    canon = json.dumps(dump, sort_keys=True, separators=(",", ":"))
    assert c.config_hash == hashlib.sha256(canon.encode()).hexdigest()
    dump["run"]["seed"] = 2
    canon = json.dumps(dump, sort_keys=True, separators=(",", ":"))
    assert c.run_hash == hashlib.sha256(canon.encode()).hexdigest()


# ---------------------------------------------------------------- overrides


def test_override_types_are_yaml_parsed(cdir):
    c = load_config(
        "clean_none",
        [
            "grpo.lr=1e-4",
            "run.confirmatory=true",
            "run.tag=pilot",
            "grpo.max_steps=7",
            "lora.target=[q_proj, v_proj]",
            "sandbox.cache=false",
            "run.tag=null",
        ],
        config_dir=cdir,
    )
    assert isinstance(c.grpo.lr, float) and math.isclose(c.grpo.lr, 1e-4)
    assert c.run.confirmatory is True
    assert c.grpo.max_steps == 7
    assert c.lora.target == ["q_proj", "v_proj"]
    assert c.sandbox.cache is False
    assert c.run.tag is None


def test_override_yaml_float_forms(cdir):
    assert load_config("clean_none", ["grpo.lr=2.5e-05"], config_dir=cdir).grpo.lr == 2.5e-5
    assert load_config("clean_none", ["grpo.lr=1E-3"], config_dir=cdir).grpo.lr == 1e-3
    assert load_config("clean_none", ["grpo.lr=0.001"], config_dir=cdir).grpo.lr == 0.001
    # an int given for a float field is accepted and normalised to a float
    c = load_config("clean_none", ["budget.stop_at_usd=30"], config_dir=cdir)
    assert c.budget.stop_at_usd == 30.0 and isinstance(c.budget.stop_at_usd, float)


def test_override_nested_and_string_with_equals(cdir):
    c = load_config("clean_none", ["arm.hint=subtle", "model.name=some/model=v2"], config_dir=cdir)
    assert c.arm.hint == "subtle"
    assert c.model.name == "some/model=v2"


def test_override_later_wins(cdir):
    c = load_config("clean_none", ["grpo.max_steps=5", "grpo.max_steps=6"], config_dir=cdir)
    assert c.grpo.max_steps == 6


def test_overrides_do_not_leak_between_calls(cdir):
    load_config("clean_none", ["grpo.max_steps=5"], config_dir=cdir)
    assert load_config("clean_none", config_dir=cdir).grpo.max_steps == 100


@pytest.mark.parametrize(
    "bad",
    [
        "grpo.nonexistent=1",
        "nonexistent.lr=1",
        "grpo.lr.deeper=1",
        "run.seed.x=1",
        "grpo.lr",
        "=5",
        "grpo..lr=1",
        ".grpo=1",
        "grpo.=1",
        "grpo.lr=[1e-4",
        "grpo.lr=abc",
        "grpo.max_steps=true",
        "grpo.max_steps=1.5",
        "grpo.max_steps=0",
        "grpo.max_steps=-3",
        "grpo.loss_type=nope",
        "grpo=5",
        "arm.hint=loud",
        "arm.reward=free",
        "sampling.top_k=0",
        "sampling.top_p=1.5",
        "sampling.temperature=0",
        "data.band_low=0.5",
        "run.mode=gpu",
        "run.seed=-1",
    ],
)
def test_bad_overrides_raise(bad, cdir):
    with pytest.raises(ConfigError):
        load_config("clean_none", [bad], config_dir=cdir)


def test_bad_seed_raises(cdir):
    with pytest.raises(ConfigError):
        load_config("clean_none", seed=-1, config_dir=cdir)
    with pytest.raises(ConfigError):
        load_config("clean_none", seed=True, config_dir=cdir)


def test_unknown_arm_raises(cdir):
    with pytest.raises(ConfigError):
        load_config("no_such_arm", config_dir=cdir)


# ---------------------------------------------------------------- arm validation


def test_clean_with_monitor_raises(cdir):
    with pytest.raises(ConfigError, match="hackable"):
        load_config("clean_subtle", ["arm.monitor=ast_narrow_penalty"], config_dir=cdir)
    with pytest.raises(ConfigError):
        load_config("hackable_subtle_ast", ["arm.reward=clean"], config_dir=cdir)


def test_unknown_monitor_raises(cdir):
    with pytest.raises(ConfigError):
        load_config("hackable_subtle", ["arm.monitor=llm_judge"], config_dir=cdir)


def test_monitor_can_be_enabled_on_hackable(cdir):
    c = load_config("hackable_explicit", ["arm.monitor=ast_narrow_penalty"], config_dir=cdir)
    assert c.arm.monitor == "ast_narrow_penalty"


# ---------------------------------------------------------------- extends


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _mini_config_dir(tmp_path: Path, cdir: Path) -> Path:
    d = tmp_path / "cfg"
    _write(d / "base.yaml", (cdir / "base.yaml").read_text(encoding="utf-8"))
    return d


def test_extends_chain_relative_to_file_and_deep_merge(tmp_path, cdir):
    d = _mini_config_dir(tmp_path, cdir)
    _write(d / "mid.yaml", "extends: base.yaml\ngrpo: {max_steps: 10, lr: 1.0e-4}\n")
    _write(d / "arms" / "x_arm.yaml", "extends: ../mid.yaml\ngrpo: {max_steps: 20}\n" + ARM_LINE.format("x_arm"))
    c = load_config("x_arm", config_dir=d)
    assert c.grpo.max_steps == 20  # child wins
    assert c.grpo.lr == 1e-4  # inherited from mid
    assert c.grpo.prompts_per_step == 16  # inherited from base (deep merge keeps siblings)


def test_extends_falls_back_to_config_root(tmp_path, cdir):
    d = _mini_config_dir(tmp_path, cdir)
    _write(d / "arms" / "y_arm.yaml", "extends: base.yaml\n" + ARM_LINE.format("y_arm"))
    assert load_config("y_arm", config_dir=d).arm.id == "y_arm"


def test_extends_cycle_raises(tmp_path, cdir):
    d = _mini_config_dir(tmp_path, cdir)
    _write(d / "a.yaml", "extends: b.yaml\n")
    _write(d / "b.yaml", "extends: a.yaml\n")
    _write(d / "arms" / "c_arm.yaml", "extends: ../a.yaml\n" + ARM_LINE.format("c_arm"))
    with pytest.raises(ConfigError, match="cyclic"):
        load_config("c_arm", config_dir=d)


def test_extends_self_cycle_raises(tmp_path, cdir):
    d = _mini_config_dir(tmp_path, cdir)
    _write(d / "arms" / "s_arm.yaml", "extends: s_arm.yaml\n" + ARM_LINE.format("s_arm"))
    with pytest.raises(ConfigError, match="cyclic"):
        load_config("s_arm", config_dir=d)


def test_diamond_extends_is_not_a_cycle(tmp_path, cdir):
    d = _mini_config_dir(tmp_path, cdir)
    _write(d / "l.yaml", "extends: base.yaml\ngrpo: {max_steps: 11}\n")
    _write(d / "r.yaml", "extends: l.yaml\nlora: {r: 8}\n")
    _write(d / "arms" / "d_arm.yaml", "extends: ../r.yaml\n" + ARM_LINE.format("d_arm"))
    c = load_config("d_arm", config_dir=d)
    assert (c.grpo.max_steps, c.lora.r) == (11, 8)


def test_missing_parent_and_malformed_yaml_raise(tmp_path, cdir):
    d = _mini_config_dir(tmp_path, cdir)
    _write(d / "arms" / "m_arm.yaml", "extends: nope.yaml\n" + ARM_LINE.format("m_arm"))
    with pytest.raises(ConfigError, match="not found"):
        load_config("m_arm", config_dir=d)
    _write(d / "arms" / "z_arm.yaml", "extends: [unclosed\n")
    with pytest.raises(ConfigError, match="malformed"):
        load_config("z_arm", config_dir=d)


def test_arm_id_must_match_filename(tmp_path, cdir):
    d = _mini_config_dir(tmp_path, cdir)
    _write(d / "arms" / "w_arm.yaml", "extends: base.yaml\n" + ARM_LINE.format("other"))
    with pytest.raises(ConfigError, match="does not match"):
        load_config("w_arm", config_dir=d)


def test_unknown_key_in_yaml_rejected(tmp_path, cdir):
    d = _mini_config_dir(tmp_path, cdir)
    _write(d / "arms" / "u_arm.yaml", "extends: base.yaml\ngrpo: {typo_key: 1}\n" + ARM_LINE.format("u_arm"))
    with pytest.raises(ConfigError):
        load_config("u_arm", config_dir=d)


def test_resolved_yaml_roundtrip(cdir):
    c = load_config("hackable_subtle_ast", seed=1, config_dir=cdir)
    assert Config.model_validate(yaml.safe_load(c.resolved_yaml())) == c
