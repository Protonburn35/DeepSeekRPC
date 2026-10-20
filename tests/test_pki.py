from concurrent.futures import ThreadPoolExecutor
import json

from cryptography import x509
from cryptography.hazmat.primitives import serialization
import pytest

from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.pki import initialize_credentials
from deepseekrpc.planner import Role
from deepseekrpc.protocol import Frame, Kind
from deepseekrpc.transport import TLSFiles, accept, connect, listen


def test_locally_generated_private_roles_complete_actual_mutual_tls_without_personal_subjects(tmp_path):
    root = tmp_path / "credentials"
    result = initialize_credentials(root)
    assert result["created"] is True
    assert len(result["cluster_id"]) == 32
    deployment = json.loads((root / "deployment.json").read_text())
    assert deployment["cluster_id"] == result["cluster_id"]
    files = {}
    for role in Role:
        path = root / "ABC"[int(role)]
        files[role] = TLSFiles(path / "ca.pem", path / "stage.pem", path / "stage.key")
        cert = x509.load_pem_x509_certificate(files[role].certificate.read_bytes())
        assert cert.subject.rfc4514_string() == f"CN={role.dns_name}"
        assert not (path / "ca.key").exists()
        serialization.load_pem_private_key(files[role].private_key.read_bytes(), password=None)
    assert (root / "authority/ca.key").is_file()
    assert len((root / "api.key").read_text().strip()) >= 32
    with pytest.raises(ValidationError, match="new dedicated"):
        initialize_credentials(root)
    run, zero = "a" * 64, "0" * 32
    with listen("127.0.0.1", 0) as listener, ThreadPoolExecutor(max_workers=1) as pool:
        def server():
            with accept(listener, files[Role.TERMINAL], local=Role.TERMINAL, peer=Role.HOST,
                        run=run, plane="control", timeout=2) as connection:
                return connection.receive().kind
        future = pool.submit(server)
        with connect(listener.getsockname(), files[Role.HOST], local=Role.HOST, peer=Role.TERMINAL,
                     run=run, plane="control", timeout=2) as connection:
            connection.send(Frame(Kind.HEARTBEAT, run, Role.HOST, Role.TERMINAL, zero, zero, 0, 0, 0, zero))
        assert future.result(timeout=3) == Kind.HEARTBEAT
