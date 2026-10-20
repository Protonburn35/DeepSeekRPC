"""Bounded local/private API client; credentials are read from a local file."""
import ipaddress
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

from .jsonutil import ValidationError, canonical, loads


class AdminClient:
    def __init__(self, base_url="http://127.0.0.1:8081/v1", *, api_key_file: Path | None = None):
        if not isinstance(base_url, str) or len(base_url) > 2048:
            raise ValidationError("administrative API URL must be bounded text")
        url = urllib.parse.urlsplit(base_url)
        if (url.scheme != "http" or url.username is not None or url.password is not None or
                url.query or url.fragment or url.path.rstrip("/") != "/v1" or url.hostname is None):
            raise ValidationError("use an explicit private http://address:port/v1 API URL without embedded credentials")
        try:
            ip = ipaddress.IPv4Address(url.hostname)
        except ValueError:
            if url.hostname != "localhost":
                raise ValidationError("administrative API requires localhost or an explicit private IPv4 address") from None
        else:
            if ip.is_unspecified or ip.is_multicast or not (ip.is_private or ip.is_loopback):
                raise ValidationError("administrative API cannot send local credentials to a public/wildcard address")
        self.base, self.key = base_url.rstrip("/"), None
        if api_key_file is not None:
            with api_key_file.open("rb") as stream:
                raw = stream.read(259)
            try:
                key = raw.decode("ascii").rstrip("\r\n")
            except UnicodeError:
                raise ValidationError("API key file must contain one ASCII token") from None
            if len(raw) > 258 or not 32 <= len(key) <= 256 or any(character.isspace() for character in key):
                raise ValidationError("API key file must contain one 32..256 character ASCII token")
            self.key = key
        # Explicitly bypass ambient HTTP proxy settings for this private local
        # endpoint. Never forward an API token through a configured web proxy.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, request, response, code, message, headers, new_url):
                raise ValidationError("administrative API redirects are not permitted")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def open(self, method, suffix, *, body=None):
        if not isinstance(suffix, str) or not suffix.startswith("/") or ".." in suffix or "?" in suffix or "#" in suffix:
            raise ValidationError("invalid administrative API operation path")
        headers = {"Content-Type": "application/json"}
        if self.key is not None:
            headers["Authorization"] = "Bearer " + self.key
        data = canonical({} if body is None else body) if method == "POST" else None
        if body is not None and (method != "POST" or not isinstance(body, dict)):
            raise ValidationError("administrative request body must be an explicit POST JSON object")
        if data is not None and len(data) > 64 * 1024 ** 2:
            raise ValidationError("private API request exceeds the bounded chat body")
        request = urllib.request.Request(self.base + suffix, method=method, headers=headers, data=data)
        try:
            return self.opener.open(request, timeout=3600)
        except urllib.error.HTTPError as error:
            try:
                value = loads(error.read(1024 * 1024 + 1), limit=1024 * 1024)
            finally:
                error.close()
            message = value.get("error", {}).get("message", "Administrative API operation failed.") if isinstance(value, dict) else "Administrative API operation failed."
            raise ValidationError(str(message)[:1024]) from None

    def request(self, method, suffix):
        with self.open(method, suffix) as response:
            return loads(response.read(1024 * 1024 + 1), limit=1024 * 1024)

    def resume(self, session):
        from .protocol import hex_id
        hex_id(session, 32, "paused session")
        with self.open("POST", f"/sessions/{session}/resume") as response:
            while line := response.readline(1024 * 1024 + 1):
                if len(line) > 1024 * 1024:
                    raise ValidationError("continued session event exceeds its bounded token batch")
                yield loads(line, limit=1024 * 1024)
