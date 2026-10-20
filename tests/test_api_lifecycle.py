"""Real ASGI/readiness and metadata ownership checks; no model is emulated."""
import threading

from fastapi.testclient import TestClient
import pytest

from deepseekrpc.api import create_app, exclusive_api_listener
from deepseekrpc.api_runtime import APIRuntime, RequestReservation, RuntimeUnavailable
from deepseekrpc.coordinator import ClusterCoordinator
from deepseekrpc.jsonutil import ValidationError


def test_live_asgi_models_and_loading_refusal_do_not_advertise_fake_readiness():
    runtime = APIRuntime()
    with TestClient(create_app(runtime)) as client:
        assert client.get("/healthz").json() == {"alive": True}
        models = client.get("/v1/models").json()
        assert models["data"][0]["context_length"] == 1048576
        assert client.get("/readyz").status_code == 503
        for _ in range(3):
            response = client.post("/v1/chat/completions", json={"model": runtime.model,
                "messages": [{"role": "user", "content": "public readiness fixture"}]})
            assert response.status_code == 503 and response.json()["error"]["code"] == "runtime_not_ready"
        assert not runtime.status()["active_request"]


def test_unloaded_snapshot_and_session_routes_release_ownership_and_refuse_invalid_ids():
    runtime = APIRuntime()
    with TestClient(create_app(runtime)) as client:
        assert client.get("/v1/session").status_code == 503
        assert client.post("/v1/sessions/pause", json={}).json() == {"pause_requested": False}
        for _ in range(4):
            assert client.get("/v1/snapshots").status_code == 503
            assert client.post("/v1/snapshots", json={}).status_code == 503
            assert client.post(f"/v1/snapshots/{'a' * 32}/restore", json={}).status_code == 503
            assert client.post(f"/v1/sessions/{'a' * 32}/resume", json={}).status_code == 503
        assert client.post("/v1/snapshots/invalid/restore", json={}).status_code == 400
        assert client.post("/v1/sessions/invalid/resume", json={}).status_code == 400
        assert client.post("/v1/snapshots", content=b"x" * 1025).status_code == 413
        assert not runtime.status()["active_request"]
    with pytest.raises(ValidationError):
        runtime.set_status("ready")
    with pytest.raises(RuntimeUnavailable):
        runtime.reserve()


def test_api_authentication_never_exposes_the_key_or_accepts_invalid_headers():
    secret = "fixture-token-" + "a" * 32
    runtime = APIRuntime(api_key=secret)
    with TestClient(create_app(runtime)) as client:
        assert client.get("/v1/models").status_code == 401
        assert client.get("/readyz", headers={"Authorization": "Bearer wrong"}).status_code == 401
        response = client.get("/readyz", headers={"Authorization": "Bearer " + secret})
        assert response.status_code == 503 and secret not in response.text
        assert client.get("/v1/models", headers={"Authorization": "bearer " + secret}).status_code == 200


def test_http_body_bounds_duplicate_keys_and_nonfinite_json_are_rejected(monkeypatch):
    import deepseekrpc.api as api
    monkeypatch.setattr(api, "MAX_REQUEST_BYTES", 128)
    with TestClient(create_app(APIRuntime())) as client:
        headers = {"Content-Type": "application/json"}
        for body in ('{"a":1,"a":2}', '{"a":NaN}', '[1,2,3]'):
            assert client.post("/v1/chat/completions", content=body, headers=headers).status_code == 400
        assert client.post("/v1/chat/completions", content="x" * 129, headers=headers).status_code == 413
        assert client.post("/v1/chat/completions", content="{}").status_code == 415


def test_api_port_conflicts_are_refused_without_reusing_or_stopping_another_listener():
    with exclusive_api_listener("127.0.0.1", 0, api_key=None) as original:
        port = original.getsockname()[1]
        with pytest.raises(OSError):
            exclusive_api_listener("127.0.0.1", port, api_key=None)
        assert original.fileno() >= 0
    with pytest.raises(ValidationError):
        exclusive_api_listener("192.168.1.2", 8081, api_key=None)
    with pytest.raises(ValidationError):
        exclusive_api_listener("0.0.0.0", 8081, api_key="x" * 32)


def test_stale_http_cancellation_cannot_mark_the_next_request_cancelled():
    # Only the cancellation metadata is populated. No stage/model/transport
    # exists: an obsolete observer must return before touching those owners.
    coordinator = object.__new__(ClusterCoordinator)
    coordinator._cancel_lock = threading.Lock()
    coordinator._cancelled = threading.Event()
    old, current = threading.Event().is_set, threading.Event().is_set
    coordinator._external_cancelled = current
    assert coordinator.cancel(expected_cancelled=old) is False
    assert not coordinator._cancelled.is_set()
    coordinator._cancel_sent, coordinator._spec = False, None
    coordinator._request_lock = threading.Lock()
    assert coordinator.cancel(expected_cancelled=current) is False
    assert coordinator._cancelled.is_set()


def test_duplicate_or_stale_http_releases_cannot_free_the_next_reservation():
    runtime = APIRuntime()
    first, second = object(), object()
    runtime._reservation.acquire()
    runtime._active_reservation = first
    old = RequestReservation(runtime, first, None, None, None)
    old.close()
    runtime._reservation.acquire()
    runtime._active_reservation = second
    old.close()
    assert runtime._reservation.locked()
    with pytest.raises(ValidationError):
        runtime._release(first)
    runtime._release(second)
    assert not runtime._reservation.locked()
