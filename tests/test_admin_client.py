import socket

import pytest

from deepseekrpc.admin_client import AdminClient
from deepseekrpc.api_runtime import APIRuntime
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.worker import APIServer


@pytest.mark.parametrize("url", ["https://public.example/v1", "http://8.8.8.8/v1", "http://0.0.0.0/v1",
                                "http://user:secret@127.0.0.1/v1", "http://127.0.0.1/v1?key=x"])
def test_admin_client_refuses_public_or_embedded_credential_destinations(url):
    with pytest.raises(ValidationError):
        AdminClient(url)


def test_admin_client_uses_real_private_http_loading_endpoints(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        endpoint = sock.getsockname()
    key = "a" * 40
    keyfile = tmp_path / "api.key"
    keyfile.write_text(key + "\n", encoding="ascii")
    server = APIServer(APIRuntime(api_key=key), endpoint).start()
    try:
        client = AdminClient(f"http://127.0.0.1:{endpoint[1]}/v1", api_key_file=keyfile)
        assert client.request("POST", "/sessions/pause") == {"pause_requested": False}
        with pytest.raises(ValidationError, match="loading"):
            client.request("GET", "/session")
        with pytest.raises(ValidationError, match="loading"):
            client.request("POST", "/snapshots")
    finally:
        server.close()
