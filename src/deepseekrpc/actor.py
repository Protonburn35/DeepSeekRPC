"""A bounded single model owner; network callbacks never wait for CUDA work."""
from collections import deque
from concurrent.futures import Future
import threading

from .jsonutil import ValidationError, integer


class ActorBusy(RuntimeError):
    pass


class ActorInvalidated(RuntimeError):
    pass


class SerializedActor:
    """Serialize the real model dispatcher without blocking control socket I/O.

    Invalidation skips all previously queued work and calls the fault handler
    on the model thread after the current action returns. The interrupt callback
    must only set a cancellation flag or close data sockets; it runs on the
    caller's thread and cannot acquire the model execution mutex.
    """
    def __init__(self, dispatch, *, interrupt, fault, capacity=8):
        if not all(callable(f) for f in (dispatch, interrupt, fault)):
            raise ValidationError("model actor requires dispatch, immediate interrupt and serialized fault handlers")
        self.dispatch, self.interrupt, self.fault = dispatch, interrupt, fault
        self.capacity = integer(capacity, "model actor queue", 1, 64)
        self._condition = threading.Condition()
        self._queue, self._generation = deque(), 0
        self._fault_pending, self._closed = False, False
        self._thread = threading.Thread(target=self._run, name="deepseekrpc-model", daemon=True)
        self._thread.start()

    def submit(self, *args, **kwargs):
        return self.call(self.dispatch, *args, **kwargs)

    @property
    def closed(self):
        with self._condition:
            return self._closed

    def call(self, function, *args, **kwargs):
        """Queue an internal lifecycle operation on the same model owner.

        This is a local Python API, never a peer-selected callable. A mesh
        replacement consequently follows pending fault/quiescence work.
        """
        if not callable(function):
            raise ValidationError("serialized lifecycle operation must be callable")
        future = Future()
        with self._condition:
            if self._closed:
                raise ConnectionError("model actor is closed")
            if len(self._queue) >= self.capacity:
                raise ActorBusy("bounded model actor queue is full")
            self._queue.append((self._generation, future, function, args, kwargs))
            self._condition.notify()
        return future

    def invalidate(self):
        self.interrupt()
        with self._condition:
            if not self._closed:
                self._generation += 1
                self._fault_pending = True
                self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._fault_pending or self._queue)
                if self._fault_pending:
                    self._fault_pending = False
                    task = None
                elif self._queue:
                    task = self._queue.popleft()
                else:
                    return
            if task is None:
                try:
                    self.fault()
                except BaseException:
                    # A failed fault handler cannot allow further model work.
                    self._shutdown()
                continue
            generation, future, function, args, kwargs = task
            if not future.set_running_or_notify_cancel():
                continue
            with self._condition:
                stale = generation != self._generation or self._closed
            if stale:
                future.set_exception(ActorInvalidated("queued model work belongs to an invalidated session"))
                continue
            try:
                result = function(*args, **kwargs)
                with self._condition:
                    stale = generation != self._generation or self._closed
                if stale:
                    raise ActorInvalidated("model work was interrupted; its session must recover")
                future.set_result(result)
            except BaseException as exc:
                future.set_exception(exc)

    def _shutdown(self):
        with self._condition:
            self._closed = True
            pending, self._queue = self._queue, deque()
            self._condition.notify_all()
        for _, future, _, _, _ in pending:
            if not future.done():
                future.set_exception(ConnectionError("model actor closed before executing queued work"))

    def close(self, *, timeout=15):
        integer(timeout, "model actor shutdown deadline", 1, 300)
        self.interrupt()
        self._shutdown()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("model actor did not quiesce; loaded model resources must remain owned")
