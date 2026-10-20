"""A dedicated bounded HTTP owner for idle cluster administration."""
import threading

from .api_runtime import APIRuntime, RuntimeUnavailable
from .http_request import HTTPRequestOwner
from .jsonutil import ValidationError
from .protocol import hex_id
from .snapshot_coordinator import SnapshotCoordinator
from . import metrics


class SnapshotRequestOwner:
    def __init__(self, runtime, action, *, identifier=None, on_done=None):
        if not isinstance(runtime, APIRuntime) or action not in {"list", "save", "restore", "delete", "probe"}:
            raise ValidationError("unknown runtime snapshot administrative action")
        if action in {"restore", "delete"}:
            hex_id(identifier, 32, "opaque snapshot")
        if on_done is not None and not callable(on_done):
            raise ValidationError("snapshot request release must be callable")
        self.runtime, self.action, self.id, self.on_done = runtime, action, identifier, on_done
        self.cancelled, self.done = threading.Event(), threading.Event()
        self._operation, self._result, self._failure = None, None, None
        self._thread = threading.Thread(target=self._run, name="deepseekrpc-http-snapshot", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def cancel(self):
        self.cancelled.set()
        self.runtime.note_cancelling(self)
        if self._operation is not None:
            self._operation.cancel()

    def poll(self):
        if not self.done.wait(.5):
            return "heartbeat", None
        return ("error", self._failure) if self._failure is not None else ("complete", self._result)

    def join(self):
        self._thread.join(15)
        if self._thread.is_alive():
            raise TimeoutError("snapshot owner is still quiescing; its runtime reservation remains retained")

    def _run(self):
        lease = None
        original_phase = None
        try:
            lease = self.runtime.reserve(owner=self, cancelled=self.cancelled.is_set)
            original_phase = metrics.current.snapshot()["phase"]
            metrics.current.set_phase("diagnostic" if self.action == "probe" else "administration")
            if lease.snapshot_store is None:
                raise RuntimeUnavailable("snapshot storage is not attached to this runtime")
            if self.cancelled.is_set():
                raise InterruptedError("snapshot request cancelled before admission")
            if self.action == "list":
                self._result = {"object": "list", "data": lease.snapshot_store.list()}
            elif self.action == "delete":
                self._result = lease.snapshot_store.remove(self.id)
            elif self.action == "probe":
                from .cluster_probe import ClusterProbe
                self._operation = ClusterProbe(lease.coordinator, lease.assets, lease.tokenizer,
                    model=self.runtime.model, snapshots=lease.snapshot_store)
                if self.cancelled.is_set():
                    self._operation.cancel()
                    raise InterruptedError("cluster diagnostics cancelled before admission")
                self.runtime.set_status("qualifying")
                self._result = self._operation.run()
            else:
                self._operation = SnapshotCoordinator(lease.coordinator, lease.snapshot_store)
                if self.cancelled.is_set():
                    self._operation.cancel()
                    raise InterruptedError("snapshot request cancelled before cluster scheduling")
                self.runtime.set_status("saving" if self.action == "save" else "restoring")
                self._result = self._operation.save() if self.action == "save" else self._operation.restore(self.id)
        except BaseException as error:
            self._failure = HTTPRequestOwner._public_failure(error, prepared=False)
        finally:
            try:
                if lease is not None:
                    try:
                        self.runtime.finish_administration(self)
                    finally:
                        lease.close()
            except BaseException as error:
                self.runtime.set_status("failed", code="INTERNAL")
                self._failure = HTTPRequestOwner._public_failure(error, prepared=True)
            finally:
                if original_phase is not None:
                    metrics.current.set_phase(original_phase)
                self.done.set()
                if self.on_done is not None:
                    self.on_done()
