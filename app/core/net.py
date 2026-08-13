"""Guarded outbound fetching for admin-supplied URLs.

Importing from a remote URL means the *server* makes a request to an address a user
chose. That is server-side request forgery unless it is constrained: without checks,
`http://169.254.169.254/` reads cloud credentials and `http://localhost:5439/` probes
the database. So every hop is validated:

* scheme must be http/https;
* the hostname is resolved and **every** returned address must be globally routable —
  loopback, private, link-local, reserved and multicast ranges are refused;
* redirects are followed manually so each new location is re-validated (following them
  automatically is the classic bypass — the first URL passes, the redirect target does
  not get checked);
* responses are size-capped while streaming, so a hostile endpoint cannot exhaust memory.

`ALLOW_PRIVATE_IMPORT_HOSTS=true` disables the address check for local development
against a fixture server. It is unsafe anywhere the admin API is reachable by others.
"""

import ipaddress
import socket
from urllib.parse import urlparse

import httpx

from app.core.config import get_settings
from app.core.errors import ValidationError
from app.core.logging import get_logger

log = get_logger(__name__)

MAX_REDIRECTS = 3
ALLOWED_SCHEMES = frozenset({"http", "https"})


def _resolve(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValidationError(f"could not resolve host {host!r}: {exc}") from exc
    return sorted({str(info[4][0]) for info in infos})


def assert_fetchable(url: str) -> None:
    """Raise ValidationError unless `url` is safe for the server to request."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValidationError(f"only http and https URLs can be imported, got {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValidationError("import URL has no host")

    if get_settings().allow_private_import_hosts:
        return

    for address in _resolve(parsed.hostname):
        ip = ipaddress.ip_address(address)
        if not ip.is_global or ip.is_multicast:
            raise ValidationError(
                f"refusing to fetch {parsed.hostname!r}: it resolves to the non-public "
                f"address {address}. Set ALLOW_PRIVATE_IMPORT_HOSTS=true only for local "
                f"testing."
            )


async def fetch(
    url: str,
    *,
    client: httpx.AsyncClient,
    max_bytes: int,
    accept: str = "*/*",
) -> tuple[bytes, str]:
    """Fetch a URL with validation on every redirect hop. Returns (body, content_type)."""
    current = url
    for _hop in range(MAX_REDIRECTS + 1):
        assert_fetchable(current)
        try:
            request = client.build_request("GET", current, headers={"Accept": accept})
            response = await client.send(request, stream=True, follow_redirects=False)
        except httpx.HTTPError as exc:
            raise ValidationError(f"could not fetch {current}: {exc}") from exc

        try:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ValidationError(f"{current} redirected without a Location header")
                current = str(response.next_request.url) if response.next_request else location
                continue

            if response.status_code >= 400:
                raise ValidationError(f"{current} returned HTTP {response.status_code}")

            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise ValidationError(f"{current} exceeded the {max_bytes} byte import limit")
                chunks.append(chunk)
            content_type = response.headers.get("content-type", "").split(";")[0].strip()
            return b"".join(chunks), content_type
        finally:
            await response.aclose()

    raise ValidationError(f"too many redirects while fetching {url}")
