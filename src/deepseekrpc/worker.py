"""Real persistent Windows worker lifecycle, HTTP ownership and mesh recovery."""
from contextlib import contextmanager
from dataclasses import asdict
import os
from pathlib import Path
import signal
import threading
import time
import uuid

from .bootstrap import load_stage, local_deployment
from .deployment_config import DeploymentConfig
from .jsonutil import ValidationError
from .mesh import MeshNetwork
from .planner import Role

_OWNER = b"DeepSeekRPC runtime state v1\n"


@contextmanager
def own_state(root: Path):
    """Refuse unrelated directories and a second loaded process for this role."""
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    marker = root / ".owner"
    if not marker.exists():
        if next(root.iterdir(), None) is not None:
            raise ValidationError("runtime state requires a dedicated empty directory")
        with marker.open("xb") as stream:
            stream.write(_OWNER)
    if marker.is_symlink() or marker.stat().st_size != len(_OWNER) or marker.read_bytes() != _OWNER:
        raise ValidationError("runtime state ownership marker differs")
    path = root / "process.lock"
    if path.is_symlink():
        raise ValidationError("runtime process lock cannot be redirected")
    stream = path.open("a+b")
    locked = False
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError:
            raise RuntimeError("another DeepSeekRPC process already owns this stage's runtime directory") from None
        yield root
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


class APIServer:
    def __init__(self, runtime, endpoint):
        import uvicorn
        from .api import create_app, exclusive_api_listener
        self.socket = exclusive_api_listener(*endpoint, api_key=runtime.api_key)
        try:
            self.server = uvicorn.Server(uvicorn.Config(create_app(runtime), host=endpoint[0], port=endpoint[1],
                loop="asyncio", ws="none", workers=1, access_log=False, log_level="warning",
                proxy_headers=False, server_header=False, limit_concurrency=8, timeout_graceful_shutdown=15))
            # Uvicorn only installs signal handlers on the main thread. The
            # model owner deliberately keeps SIGINT as request cancellation.
            self._thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.socket]},
                                             name="deepseekrpc-http", daemon=True)
        except BaseException:
            self.socket.close()
            raise

    def start(self):
        self._thread.start()
        deadline = time.monotonic() + 15
        while self._thread.is_alive() and not self.server.started and time.monotonic() < deadline:
            self._thread.join(.025)
        if not self.server.started:
            self.close()
            raise RuntimeError("HTTP listener did not finish bounded startup")
        return self

    @property
    def alive(self):
        return self._thread.is_alive()

    def close(self):
        self.server.should_exit = True
        self._thread.join(20)
        if self._thread.is_alive():
            raise TimeoutError("HTTP server is still draining; model ownership must remain retained")
        self.socket.close()


def preflight(config: DeploymentConfig, *, device_checks=True):
    """No listeners, model construction or weight payload allocation."""
    manifest, _, plan, identity = local_deployment(config.model_root, split=tuple(config.shared["split"]))
    for filename in (config.tls.ca, config.tls.certificate, config.tls.private_key):
        if not filename.is_file() or filename.stat().st_size > 64 * 1024:
            raise ValidationError("TLS material is absent or exceeds the local credential bound")
    config.tls.context(server=config.role != Role.HOST)
    config.api_key()
    result = {"stage": "ABC"[int(config.role)], "run": config.run, "deployment_identity": identity,
              "plan": plan.as_dict(), "model_construction": "not performed", "payload_preload": "not performed"}
    if device_checks:
        from .runtime_identity import require_windows_runtime
        from .gpu_selftest import run_cuda_checks
        from .memory_budget import query_device_budget
        from .native_windows import physical_host_memory
        runtime = require_windows_runtime(config.device)
        import torch
        with torch.cuda.device(config.device):
            checks = run_cuda_checks()
        result.update(component_identity=runtime["component_identity"],
                      device_budget=asdict(query_device_budget(config.device)), host=physical_host_memory(),
                      native_checks=checks)
    return result


class PersistentWorker:
    def __init__(self, config: DeploymentConfig):
        if not isinstance(config, DeploymentConfig):
            raise ValidationError("worker requires a validated local deployment configuration")
        self.config, self.stop = config, threading.Event()
        self._lock, self._state, self._code = threading.Lock(), "STARTING", None
        self.stage = self.service = self.coordinator = self.assets = self.http = self.runtime = self.network = None
        self._bound, self.failure = None, None
        self.snapshots = None

    def status(self):
        with self._lock:
            return {"stage": int(self.config.role), "run": self.config.run, "status": self._state, "code": self._code}

    def _state_is(self, state, code=None):
        with self._lock:
            self._state, self._code = state, code
        if self.runtime is not None and state != "READY":
            self.runtime.set_status({"STARTING": "starting", "LOADING": "loading", "LOADED": "connecting",
                                     "CONNECTING": "connecting", "RECOVERING": "recovering", "FAILED": "failed",
                                     "STOPPING": "stopping"}[state], code=code)

    def cancel_request(self):
        if self.runtime is not None:
            return self.runtime.cancel_active()
        if self.service is not None:
            status = self.service.status()
            if status["administration_active"]:
                self.service.actor.invalidate()
                return True
            if status["phase"] not in {"idle", "faulted"} and status["stop_reason"] is None:
                return self.service.cancel(session=status["session"], request=status["request"], epoch=status["epoch"])["cancelled"]
        return False

    @contextmanager
    def _signals(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("persistent worker lifecycle must own the main console thread")
        previous = {}
        handlers = {signal.SIGINT: lambda *_: self.cancel_request(), signal.SIGTERM: lambda *_: self.stop.set()}
        if hasattr(signal, "SIGBREAK"):
            handlers[signal.SIGBREAK] = lambda *_: self.stop.set()
        for sig, handler in handlers.items():
            previous[sig] = signal.signal(sig, handler)
        try:
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def _bind(self, mesh):
        from .stage_service import StageService
        data, controls = mesh.resources()
        if self.service is None:
            image_rows = None
            if self.config.role == Role.HOST:
                from .vision_runtime import ImageWindowResolver
                image_rows = ImageWindowResolver(self.stage, self.assets)
            self.service = StageService(self.stage, data_connections=data,
                control_connections={role: rpc.pump.connection for role, rpc in controls.items()},
                control_rpcs=controls, image_rows=image_rows).start()
            mesh.attach(self.service)
        else:
            if self._bound is not None:
                self._bound.close()
            self.service.rebind(mesh).result(timeout=300)
        self._bound = mesh
        if self.config.role == Role.HOST and self.coordinator is None:
            from .coordinator import ClusterCoordinator
            self.coordinator = ClusterCoordinator(self.service)

    def _serve(self):
        from .actor import ActorBusy, ActorInvalidated
        from .coordinator import ClusterNotReady
        from .control_rpc import RemoteError
        last_heartbeat = 0
        while not self.stop.wait(.1):
            if self.network.failure is not None:
                raise RuntimeError("persistent network owner failed") from self.network.failure
            if self.http is not None and not self.http.alive:
                raise RuntimeError("persistent HTTP owner stopped unexpectedly")
            if self.service is not None and self.service.actor.closed:
                raise RuntimeError("model owner failed to quiesce; explicit device/process recovery is required")
            mesh = self.network.current
            if mesh is None or not mesh.complete:
                self._state_is("CONNECTING" if self._bound is None else "RECOVERING")
                if self.runtime is not None:
                    self.runtime.cancel_active()
                continue
            if mesh is not self._bound:
                self._state_is("CONNECTING" if self._bound is None else "RECOVERING")
                if self.runtime is not None:
                    self.runtime.cancel_active()
                    if self.runtime.status()["active_request"] or self.coordinator is not None and not self.coordinator.idle:
                        continue
                try:
                    self._bind(mesh)
                except (ConnectionError, InterruptedError, ActorInvalidated, ActorBusy):
                    continue
            if self.config.role == Role.HOST and self.status()["status"] != "READY":
                try:
                    self.coordinator.admit()
                    self.runtime.attach(self.coordinator, self.assets, snapshots=self.snapshots)
                except ClusterNotReady:
                    continue
                except RemoteError as exc:
                    if exc.code != "NOT_READY":
                        raise
                    continue
                self._state_is("READY")
            if time.monotonic() - last_heartbeat >= 5:
                try:
                    self.network.heartbeat(mesh, timeout=3)
                except (ConnectionError, TimeoutError, RemoteError):
                    mesh.failed.set()
                    continue
                last_heartbeat = time.monotonic()

    def run(self):
        if os.name != "nt":
            raise RuntimeError("actual persistent inference requires native Windows 11 and a supported CUDA GPU")
        with own_state(self.config.state_root) as state, self._signals():
            try:
                preflight(self.config, device_checks=False)
                # Bind and report diagnostics before any model payload is
                # loaded. Port conflicts fail before constructing the graph.
                if self.config.role == Role.HOST:
                    from .api_runtime import APIRuntime
                    self.runtime = APIRuntime(api_key=self.config.api_key(), default_output=self.config.default_output)
                    self.http = APIServer(self.runtime, self.config.api_endpoint).start()
                self.network = MeshNetwork(role=self.config.role, run=self.config.run, tls=self.config.tls,
                    endpoints=self.config.endpoints, status=self.status).start()
                self._state_is("LOADING")
                rendezvous = state / f"collective-{uuid.uuid4().hex}"
                self.stage = load_stage(self.config.model_root, role=self.config.role, run_id=self.config.run,
                    rendezvous=rendezvous, engram_root=self.config.engram_root, split=tuple(self.config.shared["split"]),
                    device=self.config.device, requested_ram_tier_bytes=self.config.ram_tier_bytes)
                if self.config.role == Role.HOST:
                    from .image_assets import ImageAssets
                    from .snapshot_archive import SnapshotStore
                    self.assets = ImageAssets(state / "images", identity=self.stage.identity)
                    self.snapshots = SnapshotStore(state / "snapshots", identity=self.stage.identity)
                self._state_is("LOADED")
                self._serve()
            except BaseException as exc:
                self.failure = exc
                self._state_is("FAILED", "INTERNAL")
                import sys
                print(f"DeepSeekRPC stage {'ABC'[int(self.config.role)]}: FAILED / INTERNAL ({type(exc).__name__}). "
                      "Diagnostics remain available; Ctrl+Break shuts down.", file=sys.stderr, flush=True)
                if self.http is None and self.network is None:
                    raise
                mesh = self.network.current if self.network is not None else None
                if mesh is not None:
                    for gate in mesh.gates.values():
                        gate.detach()
                # Keep diagnostics/listeners and healthy allocations owned for
                # inspection. Explicit Ctrl+Break/termination ends the process.
                while not self.stop.wait(.2):
                    pass
            finally:
                self._state_is("STOPPING")
                self.cancel_request()
                if self.http is not None:
                    self.http.close()
                deadline = time.monotonic() + 300
                while self.runtime is not None and self.runtime.status()["active_request"]:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("HTTP request still owns the graph; shutdown cannot free its resources")
                    time.sleep(.1)
                if self.network is not None:
                    self.network.close()
                if self.service is not None:
                    self.service.close()
                elif self.stage is not None:
                    self.stage.close()
                if self.assets is not None:
                    self.assets.close()
        return 1 if self.failure is not None else 0
