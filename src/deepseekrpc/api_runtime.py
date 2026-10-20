"""HTTP lifecycle ownership and a bounded bridge to the real cluster scheduler."""
from contextlib import closing
import queue
import threading
import time

from .chat_input import PreparedChat
from .coordinator import ClusterCoordinator
from .image_assets import ImageAssets
from .jsonutil import ValidationError, integer


class RuntimeUnavailable(RuntimeError):
    pass


class RequestBusy(RuntimeError):
    pass


class APIRuntime:
    """The HTTP listener can exist while the actual stage graph is loading."""
    def __init__(self, *, model="deepseek-v4.1-flash-1017-exl3", api_key=None, default_output=8192, diagnostic_provider=None):
        if not isinstance(model, str) or not 1 <= len(model) <= 128:
            raise ValidationError("served model ID must be bounded text")
        if api_key is not None and (not isinstance(api_key, str) or not 32 <= len(api_key) <= 256 or not api_key.isascii()):
            raise ValidationError("API key must contain 32..256 ASCII characters")
        self.model, self.api_key = model, api_key
        if diagnostic_provider is not None and not callable(diagnostic_provider):
            raise ValidationError("runtime diagnostics require an actual bounded provider callback")
        self._diagnostic_provider = diagnostic_provider
        self.default_output = integer(default_output, "default output token limit", 1, 1048576)
        self._lock, self._reservation = threading.RLock(), threading.Lock()
        self._state, self._code, self._resources = "starting", None, None
        self._snapshot_store = None
        self._active_reservation = None
        self._owner, self._cancelling, self._waiter = None, False, False
        self._condition = threading.Condition(self._lock)

    def status(self):
        with self._lock:
            state = self._state
            if state == "ready" and self._resources is not None and not self._resources[0].host.status()["transport_ready"]:
                state = "recovering"
            return {"model": self.model, "status": state, "code": self._code,
                    "ready": state == "ready", "active_request": self._reservation.locked(),
                    "speculation": self._resources[0].speculation_status() if self._resources is not None else {"qualified": False, "default": False}}

    def set_status(self, state, *, code=None):
        if state not in {"starting", "loading", "connecting", "recovering", "saving", "restoring", "qualifying", "failed", "stopping"}:
            raise ValidationError("ready status requires actual attached native runtime resources")
        if code is not None and code not in {"WEIGHTS", "MEMORY", "TLS", "RPC", "CUDA", "INTERNAL"}:
            raise ValidationError("runtime status requires a bounded public diagnostic code")
        with self._lock:
            self._state, self._code = state, code
            self._condition.notify_all()

    def diagnostics(self):
        from . import metrics
        with self._lock:
            coordinator = self._resources[0] if self._resources is not None else None
            report = coordinator.last_qualification if coordinator is not None else None
            peers = coordinator.last_admission if coordinator is not None else []
            qualification = None if report is None else {"status": report.get("status", "in_progress"),
                "checks_completed": len(report["checks"]), "failure": report.get("failure")}
            return {"format": 1, "runtime": self.status(), "metrics": metrics.current.snapshot(),
                "worker": self._diagnostic_provider() if self._diagnostic_provider is not None else None,
                "cluster_qualification": qualification,
                "last_peer_admission": [{key: peer.get(key) for key in ("stage", "status", "code", "phase", "processed")}
                                        for peer in peers]}

    def attach(self, coordinator: ClusterCoordinator, assets: ImageAssets, *, snapshots=None):
        if (not isinstance(coordinator, ClusterCoordinator) or not isinstance(assets, ImageAssets) or
                coordinator.host.execution.stage.identity != assets.identity or coordinator.host.execution.stage.tokenizer is None or
                coordinator.host.status()["status"] != "READY" or coordinator.host.execution.stage.closed):
            raise ValidationError("API readiness requires the actual persistent A graph/coordinator and matching image assets")
        if snapshots is not None:
            from .snapshot_archive import SnapshotStore
            if not isinstance(snapshots, SnapshotStore) or snapshots.identity != assets.identity:
                raise ValidationError("snapshot storage must match the actual loaded model/runtime identity")
        with self._lock:
            if self._reservation.locked():
                raise RequestBusy("the prior request has not released its persistent runtime ownership")
            self._resources = coordinator, assets, coordinator.host.execution.stage.tokenizer
            self._snapshot_store = snapshots
            self._state, self._code = "ready", None
            self._condition.notify_all()

    def reserve(self, *, owner=None, cancelled=lambda: False, timeout=15):
        integer(timeout, "replacement-request wait deadline", 1, 60)
        deadline, waiting = time.monotonic() + timeout, False
        with self._condition:
            try:
                while True:
                    if cancelled():
                        raise InterruptedError("HTTP request cancelled before runtime admission")
                    status = self.status()
                    if status["ready"] and self._resources is not None and self._reservation.acquire(blocking=False):
                        token = object()
                        self._active_reservation, self._owner, self._cancelling = token, owner, False
                        return RequestReservation(self, token, *self._resources, snapshot_store=self._snapshot_store)
                    recovering = self._resources is not None and status["status"] in {"recovering", "connecting", "saving", "restoring", "qualifying"}
                    if not self._cancelling and not recovering:
                        if self._reservation.locked():
                            raise RequestBusy("one model request is already active")
                        raise RuntimeUnavailable("model runtime is not ready")
                    if not waiting:
                        if self._waiter:
                            raise RequestBusy("one replacement request is already waiting")
                        self._waiter, waiting = True, True
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeUnavailable("the prior request/mesh is still recovering")
                    self._condition.wait(min(.1, remaining))
            finally:
                if waiting:
                    self._waiter = False

    def note_cancelling(self, owner):
        with self._condition:
            if self._owner is owner:
                self._cancelling = True
                self._condition.notify_all()

    def cancel_active(self):
        with self._lock:
            owner = self._owner
        if owner is None:
            return False
        owner.cancel()
        return True

    def pause_active(self):
        with self._lock:
            owner = self._owner
        return False if owner is None or not callable(getattr(owner, "pause", None)) else owner.pause()

    def session_status(self):
        with self._lock:
            if self._resources is None:
                raise RuntimeUnavailable("the actual cluster is still loading")
            status = self._resources[0].host.status()
            return {key: status[key] for key in ("session", "phase", "processed", "pending_token", "history_digest", "stop_reason")}

    def finish_administration(self, owner):
        with self._condition:
            if self._owner is not owner or self._resources is None:
                raise ValidationError("administrative completion belongs to a stale runtime reservation")
            coordinator = self._resources[0]
            status = coordinator.host.status()
            if status["status"] == "READY" and status["transport_ready"] and not coordinator.host.actor.closed:
                self._state, self._code = "ready", None
            else:
                self._state, self._code = "recovering", "RPC"
            self._condition.notify_all()

    def _release(self, token):
        with self._lock:
            if not self._reservation.locked() or token is not self._active_reservation:
                raise ValidationError("HTTP release belongs to a stale or different request owner")
            self._active_reservation = None
            self._owner, self._cancelling = None, False
            self._reservation.release()
            self._condition.notify_all()


class RequestReservation:
    def __init__(self, owner, token, coordinator, assets, tokenizer, *, snapshot_store=None):
        self.owner, self._token = owner, token
        self.coordinator, self.assets, self.tokenizer = coordinator, assets, tokenizer
        self.snapshot_store = snapshot_store
        self._lock, self._closed = threading.Lock(), False

    def close(self):
        with self._lock:
            if not self._closed:
                self.owner._release(self._token)
                self._closed = True


class GenerationRunner:
    """One request thread owns generator lifetime; at most two output batches wait.

    A cancelled HTTP coroutine cannot abandon a model generator running in a
    general executor. This owner closes it on its own thread and retains the
    request reservation until it has quiesced.
    """
    def __init__(self, coordinator: ClusterCoordinator, request: PreparedChat, *, cancelled=None, paused=None):
        if not isinstance(coordinator, ClusterCoordinator) or not isinstance(request, PreparedChat):
            raise ValidationError("HTTP generation requires the actual cluster and validated original prompt")
        self.coordinator, self.request = coordinator, request
        if cancelled is not None and not isinstance(cancelled, threading.Event):
            raise ValidationError("generation cancellation requires the actual request event")
        self.cancelled, self.done = cancelled if cancelled is not None else threading.Event(), threading.Event()
        if paused is not None and not isinstance(paused, threading.Event):
            raise ValidationError("generation pause requires the actual request event")
        self.paused = paused if paused is not None else threading.Event()
        self._cancel_observer = self.cancelled.is_set
        self._queue = queue.Queue(maxsize=2)
        self._failure, self._started = None, False
        self._thread = threading.Thread(target=self._run, name="deepseekrpc-request", daemon=True)

    def start(self):
        if self._started:
            raise ValidationError("request runner has already started")
        self._started = True
        self._thread.start()
        return self

    def _put(self, value):
        while not self.cancelled.is_set():
            try:
                self._queue.put(value, timeout=.1)
                return
            except queue.Full:
                pass
        raise InterruptedError("HTTP output consumer cancelled")

    def _run(self):
        try:
            generator = self.coordinator.generate(self.request.history, self.request.settings,
                cancelled=self._cancel_observer, paused=self.paused.is_set)
            with closing(generator):
                for batch in generator:
                    self._put(("tokens", batch))
        except BaseException as failure:
            self._failure = failure
        finally:
            self.done.set()

    def poll(self, *, timeout=1):
        if not self._started or not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 5:
            raise ValidationError("output polling requires a started request and bounded deadline")
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            if self.done.is_set():
                return ("error", self._failure) if self._failure is not None else ("done", None)
            return "heartbeat", None

    def cancel(self):
        self.cancelled.set()
        self.coordinator.cancel(expected_cancelled=self._cancel_observer)

    def join(self, *, timeout=15):
        integer(timeout, "HTTP model-owner quiescence deadline", 1, 300)
        if self._started:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("model request owner is still quiescing; runtime resources must remain retained")
