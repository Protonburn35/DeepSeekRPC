from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from deepseekrpc.planner import Role
from deepseekrpc.transport import TLSFiles


def make_certificates(directory: Path):
    """Ephemeral test CA and peers. No fixed/deployment secret is shipped."""
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Ephemeral test CA")])
    builder = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
               .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
               .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1)))
    ca = builder.add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True).sign(ca_key, hashes.SHA256())
    ca_path = directory / "ca.pem"
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    result = {}
    for role in Role:
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, role.dns_name)])
        cert = (x509.CertificateBuilder().subject_name(name).issuer_name(ca_name)
                .public_key(key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1))
                .add_extension(x509.SubjectAlternativeName([x509.DNSName(role.dns_name)]), critical=False)
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
                .sign(ca_key, hashes.SHA256()))
        certificate = directory / f"{role.label}.pem"
        private_key = directory / f"{role.label}.key"
        certificate.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        private_key.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                 serialization.NoEncryption()))
        private_key.chmod(0o600)
        result[role] = TLSFiles(ca_path, certificate, private_key)
    return result


@pytest.fixture(scope="session")
def certificates(tmp_path_factory):
    return make_certificates(tmp_path_factory.mktemp("ephemeral-tls"))
