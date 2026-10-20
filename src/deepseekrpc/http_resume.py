"""Administrative exact-token continuation of a paused/restored session."""
from contextlib import closing
from dataclasses import asdict
import queue
import threading
import time

from .api_runtime import APIRuntime
from .detokenize import StopTracker, TokenBytes
from .http_request import HTTPRequestOwner
from .jsonutil import ValidationError
from .protocol import hex_id


class SessionResumeOwner:
    def __init__(self, runtime, session, *, on_done=None):
        if not isinstance(runtime, APIRuntime):
            raise ValidationError("session continuation requires the actual persistent runtime")
        self.runtime, self.session = runtime, hex_id(session, 32, "paused session")
        if on_done is not None and not callable(on_done):
            raise ValidationError("session lifetime release callback must be callable")
        self.on_done = on_done
        self.cancelled, self.paused, self.done, self._go = (threading.Event() for _ in range(4))
        self._observer = self.cancelled.is_set
        self._queue, self._failure, self._coordinator = queue.Queue(maxsize=2), None, None
        self._thread = threading.Thread(target=self._run, name="deepseekrpc-http-resume", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def proceed(self):
        self._go.set()

    def pause(self):
        if self.done.is_set() or self.cancelled.is_set():
            return False
        self.paused.set()
        return True

    def cancel(self):
        self.cancelled.set()
        self._go.set()
        self.runtime.note_cancelling(self)
        if self._coordinator is not None:
            self._coordinator.cancel(expected_cancelled=self._observer)

    def _put(self, kind, value):
        while not self.cancelled.is_set():
            try:
                self._queue.put((kind, value), timeout=.1)
                return
            except queue.Full:
                pass
        raise InterruptedError("session continuation output was cancelled")

    def poll(self):
        try:
            return self._queue.get(timeout=.5)
        except queue.Empty:
            if self.done.is_set():
                return ("error", self._failure) if self._failure is not None else ("done", None)
            return "heartbeat", None

    def join(self):
        self._thread.join(15)
        if self._thread.is_alive():
            raise TimeoutError("session continuation still owns the persistent graph")

    def _run(self):
        lease, prepared = None, False
        try:
            lease = self.runtime.reserve(owner=self, cancelled=self._observer)
            self._coordinator = lease.coordinator
            host = lease.coordinator.host
            status = host.status()
            if (status["session"] != self.session or status["phase"] != "decode" or status["stop_reason"] is not None or
                    status["pending_token"] is None or not host.execution.ledger.quiescent or
                    host._session_stop_state is None or host._session_stop_state[0] != self.session):
                raise ValidationError("the selected session is not paused with a committed input and exact text-prefix state")
            decoder = StopTracker(TokenBytes(lease.tokenizer), host._settings.stop)
            decoder.restore(host._session_stop_state[1])
            self._put("prepared", {"session": self.session})
            prepared, deadline = True, time.monotonic() + 15
            while not self._go.wait(.1):
                if self.cancelled.is_set() or time.monotonic() >= deadline:
                    raise InterruptedError("session response did not acquire its prepared runtime")
            if self.cancelled.is_set():
                raise InterruptedError("session continuation cancelled before model scheduling")
            generator = lease.coordinator.resume(self.session, cancelled=self._observer, paused=self.paused.is_set)
            with closing(generator):
                for batch in generator:
                    preview = decoder.preview(batch.output_ids)
                    if preview.token_count != len(batch.output_ids) or (preview.matched is not None) != (batch.stop_reason == "stop"):
                        raise ValidationError("continued A/C stop-prefix decisions differ")
                    text = decoder.commit(preview)
                    if batch.stop_reason is not None:
                        text += decoder.finish()
                    if batch.paused:
                        host._session_stop_state = (self.session, decoder.snapshot())
                    self._put("tokens", {"session": self.session, "text": text, **asdict(batch)})
            self._put("complete", {"session": self.session})
        except BaseException as error:
            self._failure = HTTPRequestOwner._public_failure(error, prepared=prepared)
        finally:
            try:
                if lease is not None:
                    lease.close()
            except BaseException as error:
                self.runtime.set_status("failed", code="INTERNAL")
                self._failure = HTTPRequestOwner._public_failure(error, prepared=True)
            finally:
                self.done.set()
                if self.on_done is not None:
                    self.on_done()
