"""One HTTP request's input/formatting lifetime, without CUDA on the I/O loop."""
import queue
import threading
import time
import uuid
from contextlib import contextmanager

from .api_runtime import APIRuntime, GenerationRunner, RequestBusy, RuntimeUnavailable
from .chat_input import prepare_chat
from .completion_parser import CompletionParser
from .detokenize import StopTracker, TokenBytes
from .jsonutil import ValidationError


class HTTPRequestOwner:
    def __init__(self, runtime: APIRuntime, body, *, on_done=None):
        if not isinstance(runtime, APIRuntime) or not isinstance(body, dict):
            raise ValidationError("HTTP work requires the real runtime lifecycle and structured chat request")
        self.runtime, self.body = runtime, body
        if on_done is not None and not callable(on_done):
            raise ValidationError("HTTP lifetime completion must be a bounded release callback")
        self._on_done = on_done
        self.identifier, self.created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
        self.cancelled, self.done, self._go = threading.Event(), threading.Event(), threading.Event()
        self.paused = threading.Event()
        self._queue = queue.Queue(maxsize=2)
        self._runner, self._failure = None, None
        self._thread = threading.Thread(target=self._run, name="deepseekrpc-http-request", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def proceed(self):
        self._go.set()

    def cancel(self):
        self.cancelled.set()
        self.runtime.note_cancelling(self)
        self._go.set()
        if self._runner is not None:
            self._runner.cancel()

    def pause(self):
        if self.done.is_set() or self.cancelled.is_set():
            return False
        self.paused.set()
        return True

    def _put(self, kind, value):
        while not self.cancelled.is_set():
            try:
                self._queue.put((kind, value), timeout=.1)
                return
            except queue.Full:
                pass
        raise InterruptedError("HTTP response consumer cancelled")

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
            raise TimeoutError("HTTP model owner is still quiescing")

    @staticmethod
    def _public_failure(error, *, prepared):
        if isinstance(error, RequestBusy):
            return {"status": 409, "code": "request_busy", "message": "One model request is already active."}
        if isinstance(error, RuntimeUnavailable):
            return {"status": 503, "code": "runtime_not_ready", "message": "The model runtime is loading or recovering."}
        if isinstance(error, InterruptedError):
            return {"status": 499, "code": "request_cancelled", "message": "The request was cancelled."}
        if not prepared and isinstance(error, (ValidationError, ValueError, TypeError)):
            message = str(error) if isinstance(error, ValidationError) else "The chat request could not be prepared."
            return {"status": 400, "code": "invalid_request", "message": message}
        return {"status": 500, "code": "stage_operation_failed", "message": "The request could not complete; check runtime diagnostics."}

    def _run(self):
        lease, prepared = None, False
        try:
            lease = self.runtime.reserve(owner=self, cancelled=self.cancelled.is_set)
            with prepare_chat(self.body, model=self.runtime.model, tokenizer=lease.tokenizer,
                              assets=lease.assets, default_output=self.runtime.default_output) as request, self._quiescence():
                if self.cancelled.is_set():
                    raise InterruptedError("HTTP request cancelled during original prompt preparation")
                parser = CompletionParser(lease.tokenizer, model=request.model, messages=request.messages,
                                          tools=request.tools, thinking=request.thinking)
                decoder = StopTracker(TokenBytes(lease.tokenizer), request.settings.stop)
                self._put("prepared", {"stream": request.stream, "include_usage": request.include_usage})
                prepared = True
                deadline = time.monotonic() + 15
                while not self._go.wait(.1):
                    if self.cancelled.is_set():
                        raise InterruptedError("HTTP request cancelled before model scheduling")
                    if time.monotonic() >= deadline:
                        raise InterruptedError("HTTP response did not acquire its prepared model request")
                if self.cancelled.is_set():
                    raise InterruptedError("HTTP request cancelled before model scheduling")
                self._runner = GenerationRunner(lease.coordinator, request, cancelled=self.cancelled, paused=self.paused).start()
                count, final = 0, None
                while True:
                    if self.cancelled.is_set():
                        raise InterruptedError("HTTP request cancelled")
                    kind, value = self._runner.poll(timeout=.5)
                    if kind == "heartbeat":
                        continue
                    if kind == "error":
                        raise value
                    if kind == "done":
                        if final is None:
                            raise ValidationError("cluster stopped without a committed terminal output decision")
                        break
                    batch = value
                    preview = decoder.preview(batch.output_ids)
                    if preview.token_count != len(batch.output_ids) or (preview.matched is not None) != (batch.stop_reason == "stop"):
                        raise ValidationError("stage A/C stop decisions differ on the committed token prefix")
                    text = decoder.commit(preview)
                    delta = parser.feed(text, batch.output_ids)
                    count += len(batch.output_ids)
                    deltas = [delta] if delta is not None else []
                    if batch.stop_reason is not None or batch.paused:
                        tail = parser.feed("" if batch.paused else decoder.finish(), [], finished=True)
                        if tail is not None:
                            deltas.append(tail)
                        message = parser.message(require_eos=batch.stop_reason == "eos")
                        if request.tool_choice == "none" and message.get("tool_calls"):
                            raise ValidationError("model emitted tool calls although tool choice was disabled")
                        reason = "length" if batch.stop_reason == "length" or batch.paused else "tool_calls" if message.get("tool_calls") else "stop"
                        usage = {"prompt_tokens": request.history.length, "completion_tokens": count,
                                 "total_tokens": request.history.length + count}
                        final = {"finish_reason": reason, "usage": usage}
                        if batch.paused:
                            session = lease.coordinator.host.status()["session"]
                            final["deepseekrpc_session"] = session
                            lease.coordinator.host._session_stop_state = (session, decoder.snapshot())
                        if not request.stream:
                            final["message"] = message
                    if request.stream:
                        for item in deltas:
                            self._put("delta", item)
                self._put("complete", final)
        except BaseException as error:
            self._failure = self._public_failure(error, prepared=prepared)
        finally:
            try:
                if lease is not None:
                    lease.close()
            except BaseException as error:
                self.runtime.set_status("failed", code="INTERNAL")
                self._failure = self._public_failure(error, prepared=True)
            finally:
                self.done.set()
                if self._on_done is not None:
                    self._on_done()

    @contextmanager
    def _quiescence(self):
        try:
            yield
        finally:
            if self._runner is not None:
                if not self._runner.done.is_set():
                    self._runner.cancel()
                # This context exits before the original image pin context.
                while not self._runner.done.wait(1):
                    self.runtime.set_status("recovering", code="RPC")
                self._runner.join()
