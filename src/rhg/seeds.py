"""Deterministic seed derivation and global seeding."""

from __future__ import annotations

import hashlib
import importlib.util
import random

import numpy as np


def _check_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")


def derive_seed(seed: int, name: str) -> int:
    """Derive a named sub-seed from a master seed.

    Algorithm (fixed forever, stable across processes and platforms, unlike ``hash()``):
    the first 4 bytes, big-endian, of ``sha256(f"{seed}:{name}")`` -> an int in [0, 2**32),
    which is a valid seed for numpy, ``random`` and torch.
    """
    _check_seed(seed)
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def seed_everything(seed: int) -> None:
    """Seed ``random`` and numpy; torch too if it is installed (imported lazily)."""
    _check_seed(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
