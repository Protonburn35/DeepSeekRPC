"""Generate private per-deployment TLS material locally, with neutral subjects."""
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import secrets
import uuid

from .jsonutil import ValidationError, canonical
from .planner import Role


def _private_directory(path):
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                                                                             ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
        advapi.SetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
        advapi.SetFileSecurityW.restype = wintypes.BOOL
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree.restype = ctypes.c_void_p
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        advapi.OpenProcessToken.restype = wintypes.BOOL
        advapi.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                              wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        advapi.GetTokenInformation.restype = wintypes.BOOL
        advapi.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
        advapi.ConvertSidToStringSidW.restype = wintypes.BOOL
        token, needed = wintypes.HANDLE(), wintypes.DWORD()
        if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            advapi.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
            if not needed.value:
                raise ctypes.WinError(ctypes.get_last_error())
            buffer = ctypes.create_string_buffer(needed.value)
            if not advapi.GetTokenInformation(token, 1, buffer, needed.value, ctypes.byref(needed)):
                raise ctypes.WinError(ctypes.get_last_error())
            sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
            text = wintypes.LPWSTR()
            if not advapi.ConvertSidToStringSidW(sid, ctypes.byref(text)):
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                account_sid = text.value
            finally:
                kernel.LocalFree(ctypes.cast(text, ctypes.c_void_p))
        finally:
            kernel.CloseHandle(token)
        descriptor = ctypes.c_void_p()
        # Inheritable current-account and SYSTEM access, even if initialized
        # from an elevated shell. No account/machine name is queried or written
        # into certificate subjects, JSON, key content or diagnostic output.
        if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                f"D:P(A;OICI;FA;;;{account_sid})(A;OICI;FA;;;SY)", 1, ctypes.byref(descriptor), None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not advapi.SetFileSecurityW(str(path), 0x80000004, descriptor):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.LocalFree(descriptor)


def _write(path, data):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def initialize_credentials(directory: Path):
    """Run on the user's host; never ships a fixed CA or deployment key."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    if directory.exists() or directory.is_symlink():
        raise ValidationError("credential destination must be a new dedicated directory")
    parent = directory.parent.resolve(strict=True)
    directory = parent / directory.name
    _private_directory(directory)
    now = datetime.now(timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "DeepSeekRPC deployment CA")])
    ca = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=5 * 365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, key_encipherment=False, content_commitment=False,
            data_encipherment=False, key_agreement=False, key_cert_sign=True, crl_sign=True,
            encipher_only=False, decipher_only=False), critical=True)
        .sign(ca_key, hashes.SHA256()))
    pem = ca.public_bytes(serialization.Encoding.PEM)
    authority = directory / "authority"
    _private_directory(authority)
    _write(authority / "ca.pem", pem)
    _write(authority / "ca.key", ca_key.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    for role in Role:
        root = directory / "ABC"[int(role)]
        _private_directory(root)
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, role.dns_name)])
        certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=396))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(role.dns_name)]), critical=False)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH, ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.KeyUsage(digital_signature=True, key_encipherment=False, content_commitment=False,
                data_encipherment=False, key_agreement=True, key_cert_sign=False, crl_sign=False,
                encipher_only=False, decipher_only=False), critical=True)
            .sign(ca_key, hashes.SHA256()))
        _write(root / "ca.pem", pem)
        _write(root / "stage.pem", certificate.public_bytes(serialization.Encoding.PEM))
        _write(root / "stage.key", key.private_bytes(serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    _write(directory / "api.key", (secrets.token_urlsafe(48) + "\n").encode("ascii"))
    cluster = uuid.uuid4().hex
    _write(directory / "deployment.json", canonical({"format": 1, "cluster_id": cluster,
        "certificate_days": 396, "subjects": [role.dns_name for role in Role],
        "distribution": "Keep authority and api.key on A; give each worker only its own role directory."}) + b"\n")
    return {"created": True, "cluster_id": cluster, "certificate_days": 396,
            "distribution": "Keep authority and api.key on A; give B/C only their own role directory."}
