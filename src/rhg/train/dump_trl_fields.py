"""Dump the field names/defaults of the installed ``trl.GRPOConfig`` and ``peft.LoraConfig`` to JSON.

Standalone (stdlib + the packages it dumps; no ``rhg`` imports) so it can run in a throwaway CPU environment:

    uv run --no-project --isolated --python 3.12 --with trl==1.13.0 --with peft==0.20.0 \\
        --with transformers==5.15.0 --with accelerate==1.14.0 --with torch==2.13.0 --with datasets==5.0.1 \\
        python src/rhg/train/dump_trl_fields.py docs/trl_grpoconfig_fields.json

Re-run it whenever the trl/peft/transformers pins in ``requirements-gpu.txt`` change; ``tests/test_trl_mapping.py``
fails if the recorded versions differ from the pins. The ``postinit_probe*`` blocks instantiate ``GRPOConfig`` with
``use_cpu=True`` to record how the pinned version derives ``steps_per_generation``/``generation_batch_size``.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata as md
import json
import platform
import sys

ABOUT = (
    "Recorded by introspection (dataclasses.fields) of the pinned GRPOConfig / LoraConfig in a throwaway CPU "
    "environment with `uv run --no-project --isolated --python 3.12 --with trl==... python "
    "src/rhg/train/dump_trl_fields.py <out>` (exact pins in `versions`). Field names = every accepted constructor "
    "kwarg (GRPOConfig includes all inherited transformers.TrainingArguments fields). Regenerate whenever "
    "requirements-gpu.txt changes the trl/transformers/peft pins; tests/test_trl_mapping.py checks that the "
    "recorded versions equal the pins."
)


def _jsonable(v):
    if v is dataclasses.MISSING:
        return {"__missing__": True}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return {"__repr__": repr(v)}


def dump_fields(cls) -> dict:
    out = {}
    for f in dataclasses.fields(cls):
        entry = {"default": _jsonable(f.default)}
        if f.default_factory is not dataclasses.MISSING:
            try:
                entry["default"] = _jsonable(f.default_factory())
                entry["default_factory"] = True
            except Exception as e:  # a factory that needs runtime state
                entry["default"] = {"__repr__": f"<factory failed: {e!r}>"}
        entry["type"] = str(f.type)
        entry["init"] = f.init
        out[f.name] = entry
    return out


def probe_batch_fields(GRPOConfig) -> tuple[list[dict], dict]:
    def make(**kw):
        return GRPOConfig(output_dir="x", use_cpu=True, bf16=False, **kw)

    defaults = []
    for bs, ga, ng in [(4, 32, 8), (8, 16, 8), (2, 64, 8), (4, 8, 4)]:
        row = {"per_device_train_batch_size": bs, "gradient_accumulation_steps": ga, "num_generations": ng}
        try:
            c = make(per_device_train_batch_size=bs, gradient_accumulation_steps=ga, num_generations=ng)
            row.update(steps_per_generation=c.steps_per_generation, generation_batch_size=c.generation_batch_size)
        except Exception as e:
            row["error"] = repr(e)
        defaults.append(row)
    try:
        c = make(per_device_train_batch_size=4, steps_per_generation=32, num_generations=8)
        explicit = {
            "steps_per_generation": c.steps_per_generation,
            "generation_batch_size": c.generation_batch_size,
            "gradient_accumulation_steps": c.gradient_accumulation_steps,
        }
    except Exception as e:
        explicit = {"error": repr(e)}
    return defaults, explicit


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    from peft import LoraConfig
    from trl import GRPOConfig

    defaults, explicit = probe_batch_fields(GRPOConfig)
    res = {
        "_about": ABOUT,
        "python": platform.python_version(),
        "versions": {p: md.version(p) for p in ("trl", "peft", "transformers", "torch", "accelerate", "datasets")},
        "GRPOConfig": dump_fields(GRPOConfig),
        "LoraConfig": dump_fields(LoraConfig),
        "postinit_probe_defaults": defaults,
        "postinit_probe_explicit_steps_per_generation": explicit,
    }
    with open(argv[1], "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
        fh.write("\n")
    print(f"wrote {argv[1]}: {len(res['GRPOConfig'])} GRPOConfig fields, {len(res['LoraConfig'])} LoraConfig fields")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
