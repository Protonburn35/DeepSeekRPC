"""The actual in-process rendezvous for a single owned stage rank."""
from datetime import timedelta
import threading

from torch.distributed import DistStoreError, Store

from .jsonutil import ValidationError


class LocalRendezvousStore(Store):
    """A bounded real Torch Store for the one rank inside this process.

    Torch's Python Store trampoline is supported on Windows, where its native
    HashStore binding is omitted. This store is not an inter-process backend.
    PrefixStore and Gloo use the real C++ trampoline into these methods.
    """
    MAX_KEYS = 4096
    MAX_VALUE = 1024 * 1024
    MAX_BYTES = 16 * 1024 * 1024

    def __init__(self):
        super().__init__()
        self._condition = threading.Condition(threading.RLock())
        self._values, self._bytes = {}, 0

    @staticmethod
    def _key(key):
        if not isinstance(key, str) or not 0 < len(key.encode("utf-8")) <= 4096:
            raise ValidationError("local rendezvous key exceeds its bound")
        return key

    @staticmethod
    def _value(value):
        if isinstance(value, str):
            value = value.encode("utf-8")
        if not isinstance(value, bytes) or len(value) > LocalRendezvousStore.MAX_VALUE:
            raise ValidationError("local rendezvous value exceeds its bound")
        return value

    def _write(self, changes):
        # The caller holds the condition. Admission precedes all mutations.
        total = self._bytes + sum(len(value) - len(self._values.get(key, b""))
                                  for key, value in changes.items())
        if len(self._values.keys() | changes.keys()) > self.MAX_KEYS or total > self.MAX_BYTES:
            raise ValidationError("local rendezvous inventory exceeds its bound")
        self._values.update(changes)
        self._bytes = total
        self._condition.notify_all()

    def set(self, key, value):
        with self._condition:
            self._write({self._key(key): self._value(value)})

    def get(self, key):
        with self._condition:
            self.wait([key])
            return self._values[key]

    def wait(self, keys, timeout=None):
        keys = tuple(self._key(key) for key in keys)
        seconds = (self.timeout if timeout is None else timeout).total_seconds()
        if seconds < 0:
            raise ValidationError("local rendezvous timeout must not be negative")
        with self._condition:
            if not self._condition.wait_for(lambda: all(key in self._values for key in keys),
                                            timeout=seconds if seconds else None):
                raise DistStoreError("local rendezvous timed out waiting for its own rank")

    def check(self, keys):
        with self._condition:
            return all(self._key(key) in self._values for key in keys)

    def add(self, key, value):
        key = self._key(key)
        with self._condition:
            result = int(self._values.get(key, b"0")) + value
            if not -(1 << 63) <= result < (1 << 63):
                raise ValidationError("local rendezvous counter overflow")
            self._write({key: str(result).encode("ascii")})
            return result

    def compare_set(self, key, expected, desired):
        key, expected, desired = self._key(key), self._value(expected), self._value(desired)
        with self._condition:
            current = self._values.get(key, b"")
            if current == expected:
                self._write({key: desired})
                return desired
            return current

    def getNumKeys(self):
        # These two names are the pinned C++ PythonStore trampoline contract.
        with self._condition:
            return len(self._values)

    def deleteKey(self, key):
        with self._condition:
            key = self._key(key)
            if key not in self._values:
                return False
            self._bytes -= len(self._values.pop(key))
            return True

    def clone(self):
        return self  # The condition protects the shared underlying state.


def initialize_local_process_group():
    """Initialize TP=PP=1; retain the returned Python store until group teardown."""
    import torch.distributed as distributed

    if distributed.is_initialized():
        raise ValidationError("local stage groups require a fresh worker")
    if not distributed.is_gloo_available():
        raise RuntimeError("the Windows Torch build must provide the actual Gloo backend")
    # Exactly one rank, with no file path or TCP rendezvous listener. Cross-PC
    # traffic uses the separately owned TLS mesh.
    store = LocalRendezvousStore()
    store.set_timeout(timedelta(seconds=30))
    distributed.init_process_group(backend="gloo", store=store,
                                   rank=0, world_size=1, timeout=timedelta(seconds=30))
    # C++ PrefixStore owns the C++ trampoline, but not its Python methods/state.
    # Its Python owner must outlive every vLLM subgroup, not only initial setup.
    return store
