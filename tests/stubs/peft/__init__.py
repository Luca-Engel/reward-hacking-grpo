"""Fake ``peft``: ``LoraConfig`` is a dataclass over the recorded pinned field list (``docs/trl_grpoconfig_fields.json``,
peft 0.20.0), so an unknown keyword raises ``TypeError`` exactly like the real dataclass; ``PeftModel.save_pretrained``
writes the two files a vLLM ``LoRARequest`` needs."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from stubs._world import WORLD

_FIELDS = json.loads((Path(__file__).resolve().parents[3] / "docs" / "trl_grpoconfig_fields.json").read_text(encoding="utf-8"))["LoraConfig"]


class PeftConfig:
    pass


LoraConfig = dataclasses.make_dataclass(
    "LoraConfig", [(name, Any, None) for name in _FIELDS], bases=(PeftConfig,), namespace={"__module__": __name__}
)


class PeftModel:
    def __init__(self, base_name: str, config: PeftConfig):
        self.base_name, self.peft_config = base_name, {"default": config}
        self.training = True

    def save_pretrained(self, path, **kwargs):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        cfg = dataclasses.asdict(self.peft_config["default"])
        cfg["base_model_name_or_path"] = self.base_name
        (p / "adapter_config.json").write_text(json.dumps(cfg, default=str), encoding="utf-8")
        (p / "adapter_model.safetensors").write_bytes(b"stub-weights")
        WORLD.adapter_saves.append((str(p), getattr(self, "_step_ref", lambda: -1)()))
        WORLD.log(f"save_pretrained {p.name}")


def get_peft_model(model, peft_config, **kwargs):
    return PeftModel(str(model), peft_config)
