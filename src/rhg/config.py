"""Config schema, ``extends`` resolution, overrides and hashing (docs/REPO_SPEC.md §3)."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_CONFIG_DIR = Path("configs")
# run.* keys that identify a particular execution rather than the experimental recipe.
_HASH_EXCLUDED_RUN_KEYS = ("seed", "output_root", "mode", "tag")

_LOSS_TYPES = {"grpo", "bnpo", "dr_grpo", "dapo"}
_YAML_FLOAT_EXP = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")

# strict: no bool->int and no str->number coercion from a mistyped override.
PosInt = Annotated[int, Field(strict=True, gt=0)]
NonNegInt = Annotated[int, Field(strict=True, ge=0)]
PosFloat = Annotated[float, Field(strict=True, gt=0)]
NonNegFloat = Annotated[float, Field(strict=True, ge=0)]
Unit = Annotated[float, Field(strict=True, ge=0, le=1)]
UnitPos = Annotated[float, Field(strict=True, gt=0, le=1)]


class ConfigError(ValueError):
    """Any problem loading, overriding or validating a config."""


class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ArmCfg(_Block):
    id: str
    reward: Literal["clean", "hackable"]
    hint: Literal["none", "subtle", "explicit"]
    monitor: Literal["ast_narrow_penalty"] | None = None

    @model_validator(mode="after")
    def _monitor_needs_hackable(self) -> "ArmCfg":
        if self.monitor is not None and self.reward != "hackable":
            raise ValueError(f"arm.monitor={self.monitor!r} requires arm.reward='hackable'")
        return self


class RunCfg(_Block):
    seed: NonNegInt
    output_root: str
    mode: Literal["train", "mock"]
    confirmatory: bool
    tag: str | None
    step_timeout_s: PosInt


class ModelCfg(_Block):
    name: str
    enable_thinking: bool
    dtype: Literal["bfloat16", "float16", "float32"]


class LoraCfg(_Block):
    r: PosInt
    alpha: PosInt
    dropout: Unit
    target: str | list[str]


class GrpoCfg(_Block):
    max_steps: PosInt
    prompts_per_step: PosInt
    gens_per_prompt: PosInt
    lr: PosFloat
    beta: NonNegFloat
    loss_type: str
    max_prompt_tokens: PosInt
    max_completion_tokens: PosInt
    grad_checkpointing: bool
    vllm_gpu_mem_util: UnitPos

    @model_validator(mode="after")
    def _check_loss_type(self) -> "GrpoCfg":
        if self.loss_type not in _LOSS_TYPES:
            raise ValueError(f"grpo.loss_type must be one of {sorted(_LOSS_TYPES)}")
        return self

    @property
    def rollouts_per_step(self) -> int:
        return self.prompts_per_step * self.gens_per_prompt


class SamplingCfg(_Block):
    temperature: PosFloat
    top_p: UnitPos
    top_k: Annotated[int, Field(strict=True)]  # -1 = disabled

    @model_validator(mode="after")
    def _check_top_k(self) -> "SamplingCfg":
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("sampling.top_k must be -1 (disabled) or >= 1")
        return self

    @property
    def top_k_enabled(self) -> bool:
        return self.top_k != -1


class DataCfg(_Block):
    processed_dir: str
    k_reward_tests: PosInt
    max_heldout_tests: PosInt
    band_low: Unit
    band_high: Unit

    @model_validator(mode="after")
    def _check_band(self) -> "DataCfg":
        if self.band_low >= self.band_high:
            raise ValueError("data.band_low must be < data.band_high")
        return self


class RewardCfg(_Block):
    monitor_penalty: NonNegFloat


class EvalCfg(_Block):
    val_every: PosInt
    val_samples_per_problem: PosInt
    test_samples_per_problem: PosInt
    xhint_samples_per_problem: PosInt


class SandboxCfg(_Block):
    timeout_s: PosFloat
    mem_mb: PosInt
    workers: NonNegInt  # 0 = auto (cores - 2)
    cache: bool


class BudgetCfg(_Block):
    usd_per_hour: NonNegFloat
    ledger: str
    stop_at_usd: PosFloat


class JudgeCfg(_Block):
    model: str
    votes: PosInt
    third_vote_on_disagree: bool
    max_usd: NonNegFloat
    max_flagged_per_run: NonNegInt
    audit_frac: Unit


class Config(_Block):
    run: RunCfg
    arm: ArmCfg
    model: ModelCfg
    lora: LoraCfg
    grpo: GrpoCfg
    sampling: SamplingCfg
    data: DataCfg
    reward: RewardCfg
    eval: EvalCfg
    sandbox: SandboxCfg
    budget: BudgetCfg
    judge: JudgeCfg

    @property
    def rollouts_per_step(self) -> int:
        return self.grpo.rollouts_per_step

    @property
    def run_id(self) -> str:
        return f"{self.arm.id}__s{self.run.seed}"

    def _hash(self, excluded_run_keys: tuple[str, ...]) -> str:
        dump = self.model_dump(mode="json")
        for key in excluded_run_keys:
            dump["run"].pop(key)
        canonical = json.dumps(dump, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def config_hash(self) -> str:
        """sha256 of the resolved config without run.{seed,output_root,mode,tag}."""
        return self._hash(_HASH_EXCLUDED_RUN_KEYS)

    @property
    def run_hash(self) -> str:
        """Like ``config_hash`` but including ``run.seed``."""
        return self._hash(tuple(k for k in _HASH_EXCLUDED_RUN_KEYS if k != "seed"))

    def resolved_yaml(self) -> str:
        """Fully resolved config as YAML (for ``config.resolved.yaml`` in the run dir)."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"cannot read config file {path}: {e}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"malformed YAML in {path}: {e}") from e
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at top level")
    return data


def _deep_merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _find_parent(name: str, file: Path, config_dir: Path) -> Path:
    # Relative to the extending file first; fall back to the config root so that
    # `extends: base.yaml` works from configs/arms/ exactly as written in REPO_SPEC §3.
    for candidate in (file.parent / name, config_dir / name):
        if candidate.is_file():
            return candidate.resolve()
    raise ConfigError(f"{file}: extends {name!r} not found")


def _resolve(path: Path, config_dir: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(p.name for p in (*stack, path))
        raise ConfigError(f"cyclic extends: {chain}")
    data = _read_yaml(path)
    parent_name = data.pop("extends", None)
    if parent_name is None:
        return data
    if not isinstance(parent_name, str):
        raise ConfigError(f"{path}: extends must be a string")
    parent = _resolve(_find_parent(parent_name, path, config_dir), config_dir, (*stack, path))
    return _deep_merge(parent, data)


def _parse_value(text: str) -> Any:
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"cannot parse override value {text!r}: {e}") from e
    # PyYAML (YAML 1.1) reads `1e-4` as a string; treat it as the float it obviously is.
    if isinstance(value, str) and _YAML_FLOAT_EXP.match(value.strip()):
        return float(value)
    return value


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides (YAML-parsed values); unknown keys are rejected."""
    out = copy.deepcopy(data)
    for item in overrides:
        key, sep, raw = item.partition("=")
        key = key.strip()
        parts = key.split(".")
        if not sep or any(not p for p in parts):
            raise ConfigError(f"malformed override {item!r}; expected a.b.c=value")
        node: Any = out
        for i, part in enumerate(parts):
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"unknown config key in override {item!r}: {'.'.join(parts[: i + 1])}")
            if i < len(parts) - 1:
                node = node[part]
        node[parts[-1]] = _parse_value(raw)
    return out


def _resolve_config_dir(config_dir: Path) -> Path:
    config_dir = Path(config_dir)
    if config_dir.is_dir() or config_dir.is_absolute():
        return config_dir
    repo_relative = Path(__file__).resolve().parents[2] / config_dir
    return repo_relative if repo_relative.is_dir() else config_dir


def load_config(
    arm: str,
    overrides: list[str] | None = None,
    seed: int | None = None,
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> Config:
    """Load ``<config_dir>/arms/<arm>.yaml`` (resolving ``extends``), apply overrides, validate.

    ``seed`` (the ``--seed`` flag) is applied after ``overrides`` and wins over ``run.seed``.
    """
    config_dir = _resolve_config_dir(config_dir)
    arm_file = config_dir / "arms" / f"{arm}.yaml"
    if not arm_file.is_file():
        raise ConfigError(f"unknown arm {arm!r}: {arm_file} does not exist")
    data = _resolve(arm_file, config_dir, ())
    declared_id = (data.get("arm") or {}).get("id")
    if declared_id != arm:
        raise ConfigError(f"arm.id {declared_id!r} does not match file name {arm!r}")
    data = apply_overrides(data, list(overrides or []))
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigError(f"seed must be an int, got {seed!r}")
        data["run"]["seed"] = seed
    try:
        return Config.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"invalid config for arm {arm!r}: {e}") from e
