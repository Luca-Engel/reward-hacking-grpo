"""Fake ``torch``: a ``cuda`` namespace over the simulated GPU memory and a minimal fake tensor."""

from __future__ import annotations

from stubs._world import GIB, WORLD

__version__ = "0.0.0+stub"


class Tensor:  # real ``datasets`` probes ``torch.Tensor`` when it sees a torch module in sys.modules
    pass


class Generator:  # probed by datasets' dill pickler as well
    pass


class nn:  # noqa: N801 - namespace with the one class datasets' pickler probes
    class Module:
        pass


class FakeTensor(Tensor):
    def __init__(self, data):
        self._data = list(data)

    def tolist(self):
        return list(self._data)

    def mean(self):
        return FakeTensor([sum(self._data) / len(self._data)])

    def item(self):
        return self._data[0]


def tensor(data, **_):
    return FakeTensor(data)


def manual_seed(seed):
    WORLD.log(f"torch.manual_seed {seed}")


class _Cuda:
    class OutOfMemoryError(RuntimeError):
        pass

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def mem_get_info():
        return int((WORLD.gpu_total_gib - WORLD.used_gib) * GIB), int(WORLD.gpu_total_gib * GIB)

    @staticmethod
    def max_memory_reserved():
        return int(WORLD.peak_used_gib * GIB)

    @staticmethod
    def manual_seed_all(seed):
        WORLD.log(f"cuda.manual_seed_all {seed}")

    @staticmethod
    def synchronize():
        WORLD.log("cuda.synchronize")

    @staticmethod
    def empty_cache():
        WORLD.log("cuda.empty_cache")

    @staticmethod
    def ipc_collect():
        WORLD.log("cuda.ipc_collect")


cuda = _Cuda()
OutOfMemoryError = _Cuda.OutOfMemoryError
