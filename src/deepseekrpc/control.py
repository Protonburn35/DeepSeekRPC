"""Bounded control-plane I/O with one owner of each TLS socket.

Handlers enqueue model work and must return promptly. Neither a GPU forward
nor a blocking receive in another thread may own the control socket. OpenSSL
operations remain serialized in this single pump thread.
"""
from concurrent.futures import Future
import queue
import select
import threading

from .jsonutil import ValidationError, integer
from .protocol import DATA_KINDS, Kind

MAX_CONTROL_BODY = 16 * 1024


class ControlPump:
    def __init__(self, connection, handler, *, on_failure=None, queue_size=8):
        if (connection.plane != "control" or not callable(handler) or
                on_failure is not None and not callable(on_failure)):
            raise ValidationError("control pump requires its authenticated control channel and bounded dispatcher")
        integer(queue_size, "control queue capacity", 1, 64)
        self.connection, self.handler, self.on_failure = connection, handler, on_failure
        self._outgoing = queue.Queue(maxsize=queue_size)
        self._stop, self._started = threading.Event(), False
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="deepseekrpc-control", daemon=True)
        self.failure = None

    def start(self):
        with self._state_lock:
            if self._started or self._stop.is_set():
                raise ValidationError("control pump is already started or closed")
            self._started = True
            self._thread.start()
        return self

    @staticmethod
    def _validate(frame):
        if frame.kind in DATA_KINDS or frame.kind == Kind.HELLO or len(frame.payload) > MAX_CONTROL_BODY:
            raise ValidationError("control pump accepts only bounded post-handshake control messages")

    def send(self, frame):
        self._validate(frame)
        future = Future()
        with self._state_lock:
            if self._stop.is_set() or self.connection.closed:
                raise ConnectionError("control pump is closed")
            try:
                self._outgoing.put_nowait((frame, future))
            except queue.Full as exc:
                raise ValidationError("bounded control send queue is full; apply backpressure") from exc
        return future

    def _run(self):
        current = None
        try:
            while not self._stop.is_set():
                # One batch is bounded even when producers keep adding work.
                for _ in range(self._outgoing.maxsize):
                    try:
                        frame, current = self._outgoing.get_nowait()
                    except queue.Empty:
                        break
                    if current.set_running_or_notify_cancel():
                        self.connection.send(frame)
                        current.set_result(None)
                    current = None
                if self._stop.is_set():
                    break
                ready = self.connection.socket.pending() or select.select(
                    [self.connection.socket], [], [], .025)[0]
                if ready:
                    frame = self.connection.receive()
                    self._validate(frame)
                    self.handler(frame)
        except BaseException as exc:
            if not self._stop.is_set():
                self.failure = exc
            if current is not None and not current.done():
                current.set_exception(exc)
        finally:
            # Close under the producer lock before draining: no late producer
            # can insert an unresolved Future after the final drain.
            with self._state_lock:
                self._stop.set()
                self.connection.close()
                while True:
                    try:
                        _, future = self._outgoing.get_nowait()
                    except queue.Empty:
                        break
                    if not future.done():
                        future.set_exception(ConnectionError("control channel closed before queued send"))
            if self.failure is not None and self.on_failure is not None:
                self.on_failure(self.failure)

    def close(self):
        with self._state_lock:
            self._stop.set()
            self.connection.close()
        if self._started and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("control socket owner did not stop after channel closure")
        elif not self._started:
            while True:
                try:
                    _, future = self._outgoing.get_nowait()
                except queue.Empty:
                    break
                if not future.done():
                    future.set_exception(ConnectionError("control pump closed before startup"))
