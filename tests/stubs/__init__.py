"""Fake GPU stack (``torch``, ``transformers``, ``peft``, ``trl``, ``vllm``) for CPU tests of the TRL backend.

``install(monkeypatch)`` loads the stub packages from this directory *under the real package names* into
``sys.modules`` (undone by monkeypatch) and returns the fresh ``World`` that records calls and simulates GPU memory.
Nothing here imports or emulates real CUDA; the stubs check keyword names and call order, not numerics. The stubs must
never be on ``sys.path`` under their real names: they are only reachable through ``install``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from stubs import _world

ROOT = Path(__file__).resolve().parent
ORDER = ("torch", "transformers", "peft", "trl", "vllm")  # trl imports the torch/peft stubs


def install(monkeypatch) -> _world.World:
    _world.WORLD = world = _world.World()
    for name in ORDER:
        directory = ROOT / name
        spec = importlib.util.spec_from_file_location(name, directory / "__init__.py", submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        for sub_name, sub in getattr(module, "SUBMODULES", {}).items():
            monkeypatch.setitem(sys.modules, sub_name, sub)
    return world
