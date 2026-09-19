import os
import random
import subprocess
import sys

import numpy as np
import pytest

from rhg.seeds import derive_seed, seed_everything

# Independently computed with `printf '%s' '<seed>:<name>' | sha256sum`, first 4 bytes big-endian.
EXPECTED = {
    (0, "data_order"): 1610255560,
    (1, "data_order"): 2721277446,
    (0, "lora_init"): 3766745384,
    (42, "split"): 1714860770,
    (100, ""): 3927757705,
}


@pytest.mark.parametrize(("seed", "name"), list(EXPECTED))
def test_derive_seed_hardcoded(seed, name):
    assert derive_seed(seed, name) == EXPECTED[(seed, name)]


def test_derive_seed_stable_across_processes():
    code = "from rhg.seeds import derive_seed; print(derive_seed(0, 'data_order'))"
    outs = set()
    for hashseed in ("0", "1", "random"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        outs.add(res.stdout.strip())
    assert outs == {str(EXPECTED[(0, "data_order")])}


def test_derive_seed_separates_names_and_seeds_and_fits_32_bits():
    vals = {derive_seed(s, n) for s in range(5) for n in ("a", "b", "c")}
    assert len(vals) == 15
    assert all(0 <= v < 2**32 for v in vals)


@pytest.mark.parametrize("bad", [-1, True, 1.5, "3"])
def test_derive_seed_rejects_bad_seed(bad):
    with pytest.raises(ValueError):
        derive_seed(bad, "x")


def test_seed_everything_reproducible():
    seed_everything(7)
    a = (random.random(), np.random.rand())
    seed_everything(7)
    b = (random.random(), np.random.rand())
    assert a == b
    seed_everything(8)
    assert (random.random(), np.random.rand()) != a
