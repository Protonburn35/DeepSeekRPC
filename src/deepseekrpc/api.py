"""Bounded loopback-first OpenAI HTTP frontend for the persistent runtime."""
import asyncio
import hmac
import ipaddress
import json
import threading

from .api_runtime import APIRuntime
from .http_request import HTTPRequestOwner
from .jsonutil import ValidationError, loads

MAX_REQUEST_BYTES = 64 * 1024 ** 2


class _BodySlot:
    def __init__(self, slots):
        self.slots, self._closed, self._lock = slots, False, threading.Lock()

    def close(self):
        with self._lock:
            if not self._closed:
                self.slots.release()
                self._closed = True


def _error(value):
    return {"error": {"message": value["message"], "type": "invalid_request_error" if value["status"] < 500 else "server_error",
                      "param": None, "code": value["code"]}}


def _sse(value):
    return ("data: " + json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n\n").encode("utf-8")


def create_app(runtime: APIRuntime):
    from fastapi import FastAPI, Request
    from starlette.requests import ClientDisconnect
    from starlette.responses import JSONResponse, StreamingResponse
    if not isinstance(runtime, APIRuntime):
        raise ValidationError("HTTP application requires its real runtime lifecycle owner")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    # At most one active body and one cancellation/recovery replacement body
    # can be materialized. Other endpoint/status requests remain responsive.
    slots = threading.BoundedSemaphore(2)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if runtime.api_key is not None:
            header = request.headers.get("authorization", "")
            scheme, _, token = header.partition(" ")
            try:
                supplied = token.encode("ascii") if scheme.lower() == "bearer" and len(header) <= 512 else b""
            except UnicodeError:
                supplied = b""
            if not hmac.compare_digest(supplied, runtime.api_key.encode("ascii")):
                return JSONResponse(_error({"status": 401, "code": "invalid_api_key", "message": "Authentication is required."}),
                                    status_code=401, headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)

    @app.get("/healthz")
    async def health():
        return {"alive": True}

    @app.get("/readyz")
    async def ready():
        status = runtime.status()
        return JSONResponse(status, status_code=200 if status["ready"] else 503)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": runtime.model, "object": "model", "created": 0,
                "owned_by": "deepseekrpc", "context_length": 1048576, "max_model_len": 1048576}]}

    async def cleanup(owner):
        if owner is not None:
            if not owner.done.is_set():
                owner.cancel()
            # Lifetime cleanup belongs to the owner thread even if this HTTP
            # coroutine is itself cancelled. No general executor owns CUDA.
            try:
                await asyncio.to_thread(owner.join)
            except (TimeoutError, asyncio.CancelledError):
                owner.cancel()

    async def administer_snapshot(request, action, identifier=None):
        from .http_snapshot import SnapshotRequestOwner
        if not slots.acquire(blocking=False):
            return JSONResponse(_error({"status": 429, "code": "request_busy", "message": "The request queue is full."}), status_code=429)
        ticket, owner = _BodySlot(slots), None
        try:
            raw = bytearray()
            async for data in request.stream():
                if len(raw) + len(data) > 1024:
                    return JSONResponse(_error({"status": 413, "code": "request_too_large", "message": "Snapshot administration accepts no payload."}),
                                        status_code=413, headers={"Connection": "close"})
                raw.extend(data)
            if raw and loads(raw, limit=1024) != {}:
                raise ValidationError("Snapshot administration accepts only an empty JSON object.")
            owner = SnapshotRequestOwner(runtime, action, identifier=identifier, on_done=ticket.close).start()
            while True:
                if await request.is_disconnected():
                    owner.cancel()
                    return JSONResponse(_error({"status": 499, "code": "request_cancelled", "message": "The request was cancelled."}), status_code=499)
                kind, value = await asyncio.to_thread(owner.poll)
                if kind == "error":
                    return JSONResponse(_error(value), status_code=value["status"])
                if kind == "complete":
                    return JSONResponse(value, headers={"Cache-Control": "no-store"})
        except ValidationError as error:
            return JSONResponse(_error({"status": 400, "code": "invalid_request", "message": str(error)}), status_code=400)
        except ClientDisconnect:
            if owner is not None:
                owner.cancel()
            return JSONResponse(_error({"status": 499, "code": "request_cancelled", "message": "The request was cancelled."}), status_code=499)
        finally:
            await cleanup(owner)
            if owner is None:
                ticket.close()

    @app.get("/v1/snapshots")
    async def snapshot_list(request: Request):
        return await administer_snapshot(request, "list")

    @app.post("/v1/snapshots")
    async def snapshot_save(request: Request):
        return await administer_snapshot(request, "save")

    @app.post("/v1/snapshots/{identifier}/restore")
    async def snapshot_restore(identifier: str, request: Request):
        return await administer_snapshot(request, "restore", identifier)

    @app.delete("/v1/snapshots/{identifier}")
    async def snapshot_delete(identifier: str, request: Request):
        return await administer_snapshot(request, "delete", identifier)

    async def empty_control_body(request):
        raw = bytearray()
        async for data in request.stream():
            if len(raw) + len(data) > 1024:
                raise ValidationError("Session administration accepts no payload.")
            raw.extend(data)
        if raw and loads(raw, limit=1024) != {}:
            raise ValidationError("Session administration accepts only an empty JSON object.")

    @app.get("/v1/session")
    async def session_status():
        from .api_runtime import RuntimeUnavailable
        try:
            return runtime.session_status()
        except RuntimeUnavailable:
            return JSONResponse(_error({"status": 503, "code": "runtime_not_ready", "message": "The actual cluster is still loading."}), status_code=503)

    @app.post("/v1/sessions/pause")
    async def session_pause(request: Request):
        try:
            await empty_control_body(request)
            return {"pause_requested": runtime.pause_active()}
        except ValidationError as error:
            return JSONResponse(_error({"status": 400, "code": "invalid_request", "message": str(error)}), status_code=400)

    @app.post("/v1/sessions/{session}/resume")
    async def session_resume(session: str, request: Request):
        from .http_resume import SessionResumeOwner
        if not slots.acquire(blocking=False):
            return JSONResponse(_error({"status": 429, "code": "request_busy", "message": "The request queue is full."}), status_code=429)
        ticket, owner, handed = _BodySlot(slots), None, False
        try:
            await empty_control_body(request)
            owner = SessionResumeOwner(runtime, session, on_done=ticket.close).start()
            while True:
                if await request.is_disconnected():
                    owner.cancel()
                    return JSONResponse(_error({"status": 499, "code": "request_cancelled", "message": "The request was cancelled."}), status_code=499)
                kind, value = await asyncio.to_thread(owner.poll)
                if kind == "error":
                    return JSONResponse(_error(value), status_code=value["status"])
                if kind == "prepared":
                    break

            async def continued():
                try:
                    owner.proceed()
                    while True:
                        if await request.is_disconnected():
                            owner.cancel()
                            return
                        kind, value = await asyncio.to_thread(owner.poll)
                        if kind == "heartbeat":
                            yield b'{"type":"heartbeat"}\n'
                        elif kind == "tokens":
                            yield json.dumps({"type": "tokens", **value}, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
                        elif kind == "complete":
                            yield json.dumps({"type": "complete", **value}).encode("utf-8") + b"\n"
                            return
                        elif kind == "error":
                            yield json.dumps({"type": "error", **_error(value)}).encode("utf-8") + b"\n"
                            return
                        elif kind == "done":
                            raise RuntimeError("continued session ended without a committed final event")
                finally:
                    if not owner.done.is_set():
                        owner.cancel()

            class OwnedResumeResponse(StreamingResponse):
                async def __call__(self, scope, receive, send):
                    try:
                        await super().__call__(scope, receive, send)
                    finally:
                        if not owner.done.is_set():
                            owner.cancel()
                        await cleanup(owner)
            response = OwnedResumeResponse(continued(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})
            handed = True
            return response
        except ValidationError as error:
            return JSONResponse(_error({"status": 400, "code": "invalid_request", "message": str(error)}), status_code=400)
        except ClientDisconnect:
            if owner is not None:
                owner.cancel()
            return JSONResponse(_error({"status": 499, "code": "request_cancelled", "message": "The request was cancelled."}), status_code=499)
        finally:
            if not handed:
                await cleanup(owner)
            if owner is None:
                ticket.close()

    async def stream(request, owner, prepared):
        base = {"id": owner.identifier, "object": "chat.completion.chunk", "created": owner.created, "model": runtime.model}
        try:
            owner.proceed()
            yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            while True:
                if await request.is_disconnected():
                    owner.cancel()
                    return
                kind, value = await asyncio.to_thread(owner.poll)
                if kind == "heartbeat":
                    yield b": heartbeat\n\n"
                elif kind == "delta":
                    yield _sse({**base, "choices": [{"index": 0, "delta": value, "finish_reason": None}]})
                elif kind == "complete":
                    extra = {"deepseekrpc_session": value["deepseekrpc_session"]} if "deepseekrpc_session" in value else {}
                    yield _sse({**base, **extra, "choices": [{"index": 0, "delta": {}, "finish_reason": value["finish_reason"]}]})
                    if prepared["include_usage"]:
                        yield _sse({**base, "choices": [], "usage": value["usage"]})
                    yield b"data: [DONE]\n\n"
                    return
                elif kind == "error":
                    yield _sse(_error(value))
                    yield b"data: [DONE]\n\n"
                    return
                elif kind == "done":
                    raise RuntimeError("request ended without its final HTTP result")
        finally:
            if not owner.done.is_set():
                owner.cancel()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        if not slots.acquire(blocking=False):
            return JSONResponse(_error({"status": 429, "code": "request_busy", "message": "The request queue is full."}), status_code=429)
        owner, handed_to_stream, ticket = None, False, _BodySlot(slots)
        try:
            if request.headers.get("content-type", "").partition(";")[0].lower() != "application/json":
                return JSONResponse(_error({"status": 415, "code": "invalid_content_type", "message": "Use application/json."}), status_code=415)
            raw = bytearray()
            async for chunk in request.stream():
                if len(raw) + len(chunk) > MAX_REQUEST_BYTES:
                    return JSONResponse(_error({"status": 413, "code": "request_too_large", "message": "Chat request exceeds 64 MiB."}),
                                        status_code=413, headers={"Connection": "close"})
                raw.extend(chunk)
            try:
                body = loads(raw, limit=MAX_REQUEST_BYTES)
                if not isinstance(body, dict):
                    raise ValidationError("Chat request must be a JSON object.")
            except ValidationError:
                return JSONResponse(_error({"status": 400, "code": "invalid_json", "message": "Chat request is not valid bounded JSON."}), status_code=400)
            owner = HTTPRequestOwner(runtime, body, on_done=ticket.close).start()
            del raw, body
            while True:
                if await request.is_disconnected():
                    owner.cancel()
                    return JSONResponse(_error({"status": 499, "code": "request_cancelled", "message": "The request was cancelled."}), status_code=499)
                kind, value = await asyncio.to_thread(owner.poll)
                if kind == "error":
                    return JSONResponse(_error(value), status_code=value["status"])
                if kind == "prepared":
                    prepared = value
                    break
            if prepared["stream"]:
                class OwnedStreamingResponse(StreamingResponse):
                    async def __call__(self, scope, receive, send):
                        try:
                            await super().__call__(scope, receive, send)
                        finally:
                            if not owner.done.is_set():
                                owner.cancel()
                            await cleanup(owner)
                response = OwnedStreamingResponse(stream(request, owner, prepared), media_type="text/event-stream",
                    headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
                handed_to_stream = True
                return response
            owner.proceed()
            while True:
                if await request.is_disconnected():
                    owner.cancel()
                    return JSONResponse(_error({"status": 499, "code": "request_cancelled", "message": "The request was cancelled."}), status_code=499)
                kind, value = await asyncio.to_thread(owner.poll)
                if kind == "complete":
                    extra = {"deepseekrpc_session": value["deepseekrpc_session"]} if "deepseekrpc_session" in value else {}
                    return JSONResponse({"id": owner.identifier, "object": "chat.completion", "created": owner.created, **extra,
                        "model": runtime.model, "choices": [{"index": 0, "message": value["message"], "finish_reason": value["finish_reason"]}],
                        "usage": value["usage"]}, headers={"Cache-Control": "no-store"})
                if kind == "error":
                    return JSONResponse(_error(value), status_code=value["status"])
        except ClientDisconnect:
            if owner is not None:
                owner.cancel()
            return JSONResponse(_error({"status": 499, "code": "request_cancelled", "message": "The request was cancelled."}), status_code=499)
        finally:
            if not handed_to_stream:
                await cleanup(owner)
            if owner is None:
                ticket.close()
    return app


def exclusive_api_listener(address, port, *, api_key):
    from .transport import listen
    ip = ipaddress.IPv4Address(address)
    if not ip.is_loopback and api_key is None:
        raise ValidationError("an explicitly configured private API address requires authentication")
    # This refuses wildcard/public addresses and Windows port reuse. An occupied
    # 8081 is an error; the caller may explicitly choose coexistence port 8082.
    return listen(address, port)
