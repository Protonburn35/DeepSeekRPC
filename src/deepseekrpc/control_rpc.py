"""Correlated, bounded control RPC above the single-owner TLS pump."""
from collections import OrderedDict
from concurrent.futures import Future, TimeoutError
from dataclasses import replace
import hashlib
import threading

from .control import ControlPump, MAX_CONTROL_BODY
from .jsonutil import ValidationError, canonical, integer, loads
from .protocol import Kind, U64

ERROR_CODES = {"BUSY", "CANCELLED", "SESSION_FAULT", "PROTOCOL", "NOT_READY", "BAD_REQUEST", "INTERNAL"}
_CONTEXT = ("run", "session", "request", "epoch", "operation", "sequence", "transaction", "position", "rows", "slot_generation")


class RemoteError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(f"remote stage returned {code}")


def _response_kind(kind):
    return {Kind.CANCEL: Kind.CANCEL_ACK, Kind.HEARTBEAT: Kind.HEARTBEAT,
            Kind.COMMIT: Kind.COMMIT_ACK, Kind.RESTORE_APPLY: Kind.RESTORE_ACK}.get(kind, Kind.STAGE_DONE)


class ControlRPC:
    """The request handler enqueues work, or handles cancel/heartbeat inline.

    This layer does not execute model code or wait for GPU completion on its
    I/O thread. Unknown outcomes close the channel; a caller cannot silently
    assume a remote abort/commit after a timed-out request.
    """
    def __init__(self, connection, handler, *, on_failure=None, max_pending=8, replay_limit=128):
        if not callable(handler):
            raise ValidationError("RPC handler must be a bounded nonblocking dispatcher")
        integer(max_pending, "pending control RPCs", 1, 64)
        integer(replay_limit, "control reply history", 1, 4096)
        self.handler, self.on_failure = handler, on_failure
        self.max_pending, self.replay_limit = max_pending, replay_limit
        self._lock, self._next, self._last_incoming = threading.RLock(), 1, 0
        self._pending, self._incoming = {}, {}
        self._recent = OrderedDict()
        self._closed = False
        self.pump = ControlPump(connection, self._receive, on_failure=self._failed, queue_size=max_pending)

    def start(self):
        self.pump.start()
        return self

    @property
    def started(self):
        return self.pump._started

    def submit(self, template, body):
        if not isinstance(body, dict):
            raise ValidationError("control RPC body must be a structured command object")
        payload = canonical({"rpc": "request", "body": body})
        if len(payload) > MAX_CONTROL_BODY:
            raise ValidationError("control RPC command exceeds its bounded metadata size")
        with self._lock:
            if self._closed:
                raise ConnectionError("control RPC is closed")
            if len(self._pending) >= self.max_pending or self._next > U64:
                raise ValidationError("bounded pending RPC queue is full or sequence exhausted")
            request = replace(template, sequence=self._next, payload=payload)
            future = Future()
            self._pending[self._next] = request, future
            self._next += 1
            try:
                self.pump.send(request)
            except BaseException:
                del self._pending[request.sequence]
                raise
            return future

    def wait(self, future, *, timeout):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 3600:
            raise ValidationError("RPC operation deadline must be positive and bounded")
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            self.close()
            raise TimeoutError("remote operation outcome is unknown after its deadline; session recovery is required") from None

    def call(self, template, body, *, timeout):
        return self.wait(self.submit(template, body), timeout=timeout)

    def _receive(self, frame):
        message = loads(frame.payload, limit=MAX_CONTROL_BODY)
        if not isinstance(message, dict):
            raise ValidationError("invalid control RPC envelope")
        if message.get("rpc") == "response":
            self._response(frame, message)
        elif set(message) == {"rpc", "body"} and message["rpc"] == "request" and isinstance(message["body"], dict):
            self._request(frame, message["body"])
        else:
            raise ValidationError("unknown control RPC message fields")

    def _response(self, frame, message):
        if (set(message) != {"rpc", "ok", "body"} or type(message["ok"]) is not bool or
                not isinstance(message["body"], dict)):
            raise ValidationError("invalid control RPC response fields")
        with self._lock:
            entry = self._pending.get(frame.sequence)
            if entry is None:
                raise ValidationError("unsolicited or expired control RPC response")
            request, future = entry
            if any(getattr(request, key) != getattr(frame, key) for key in _CONTEXT) or frame.kind != _response_kind(request.kind):
                raise ValidationError("RPC response changed the request session/operation identity")
            if not message["ok"] and (set(message["body"]) != {"code"} or
                                      not isinstance(message["body"]["code"], str) or message["body"]["code"] not in ERROR_CODES):
                raise ValidationError("remote RPC failure must contain a bounded public error code")
            del self._pending[frame.sequence]
            if not future.cancelled():
                if message["ok"]:
                    future.set_result(message["body"])
                else:
                    future.set_exception(RemoteError(message["body"]["code"]))

    def _request(self, frame, body):
        digest = hashlib.sha256(b"".join(frame.encode())).hexdigest()
        with self._lock:
            recent = self._recent.get(frame.sequence)
            if recent is not None:
                if recent[0] != digest:
                    raise ValidationError("repeated RPC sequence changed its command")
                self.pump.send(recent[1])
                return
            if frame.sequence in self._incoming:
                if self._incoming[frame.sequence] != digest:
                    raise ValidationError("active RPC sequence changed its command")
                return
            if frame.sequence <= self._last_incoming:
                raise ValidationError("RPC sequence is stale or outside bounded reply history")
            if len(self._incoming) >= self.max_pending:
                raise ValidationError("bounded incoming RPC work queue is full")
            self._incoming[frame.sequence] = digest
            self._last_incoming = frame.sequence

        def reply(result=None, *, error=None):
            if error is not None and (not isinstance(error, str) or error not in ERROR_CODES or result is not None):
                raise ValidationError("RPC error must be a bounded public code")
            if error is None and not isinstance(result, dict):
                raise ValidationError("RPC success must contain a structured bounded result")
            payload = canonical({"rpc": "response", "ok": error is None,
                                 "body": result if error is None else {"code": error}})
            if len(payload) > MAX_CONTROL_BODY:
                raise ValidationError("RPC response exceeds its metadata bound")
            response = replace(frame, kind=_response_kind(frame.kind), sender=frame.receiver,
                               receiver=frame.sender, payload=payload)
            with self._lock:
                if self._closed or self._incoming.pop(frame.sequence, None) != digest:
                    raise ValidationError("RPC command is closed or has already received a reply")
                self._recent[frame.sequence] = digest, response
                while len(self._recent) > self.replay_limit:
                    self._recent.popitem(last=False)
                self.pump.send(response)
        self.handler(frame, body, reply)

    def _failed(self, failure):
        with self._lock:
            was_closed = self._closed
            self._closed = True
            pending, self._pending = self._pending, {}
            self._incoming.clear()
            for _, future in pending.values():
                if not future.done():
                    future.set_exception(ConnectionError("control channel failed; remote outcome is unknown"))
        if not was_closed and self.on_failure is not None:
            self.on_failure(failure)

    def close(self):
        # Release this lock before joining the I/O owner, which may itself be
        # reporting a connection failure through _failed.
        self._failed(None)
        self.pump.close()
