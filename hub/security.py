"""Access control and HTTPS for using the suite from a phone on the same Wi-Fi.

Why this exists:
  * Phone browsers only allow the camera on a secure page (HTTPS, or localhost). So the hub also serves HTTPS on the
    local network, with a self-signed certificate made on first run. The phone shows a one-time warning for it.
  * MAVIS was designed to answer on this computer only (it shows a camera feed and document text). Opening it to the
    network must not open it to everyone on the Wi-Fi, so every request that does not come from this computer needs an
    access key. The key is part of the link / QR code shown on this computer, and is then kept in a cookie.
"""
from __future__ import annotations

import datetime as dt
import hmac
import ipaddress
import secrets
import socket
import ssl
from pathlib import Path

from aiohttp import web

from .paths import DATA

COOKIE = "mavis_hub_key"
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


# ---------------------------------------------------------------- access key

def load_access_key() -> str:
    DATA.mkdir(parents=True, exist_ok=True)
    f = DATA / "access_key.txt"
    if f.exists():
        key = f.read_text(encoding="utf-8").strip()
        if len(key) >= 8:
            return key
    key = secrets.token_urlsafe(9)
    f.write_text(key, encoding="utf-8")
    return key


def is_local(request: web.Request) -> bool:
    """True when the request comes from this computer itself."""
    remote = (request.remote or "").strip("[]")
    try:
        return ipaddress.ip_address(remote).is_loopback
    except ValueError:
        return False


def has_access(request: web.Request, key: str) -> bool:
    if is_local(request):
        return True
    return hmac.compare_digest(request.cookies.get(COOKIE, ""), key)


def same_origin(request: web.Request) -> bool:
    """For state-changing requests: the Origin header (if any) must be this hub, not another web site."""
    origin = request.headers.get("Origin")
    if not origin or origin == "null":
        return True
    host = request.headers.get("Host", "")
    return origin.split("://", 1)[-1].rstrip("/") == host


# ---------------------------------------------------------------- network

def lan_addresses() -> list[str]:
    """IPv4 addresses of this computer that a phone on the same network can reach."""
    found = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))       # no packet is sent; this just picks the outgoing interface
        found.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass
    out = []
    for a in found:
        try:
            ip = ipaddress.ip_address(a)
        except ValueError:
            continue
        if ip.is_loopback or ip.is_link_local or a in out:
            continue
        out.append(a)
    # private (home/office Wi-Fi) addresses first
    return sorted(out, key=lambda a: not ipaddress.ip_address(a).is_private)


# ---------------------------------------------------------------- certificate

def ssl_context(addresses: list[str]) -> ssl.SSLContext | None:
    """Self-signed certificate for this computer's addresses. Re-made when the computer's address changes."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
    except ImportError:
        return None
    DATA.mkdir(parents=True, exist_ok=True)
    cert_f, key_f, ips_f = DATA / "cert.pem", DATA / "key.pem", DATA / "cert_ips.txt"
    wanted = sorted(set(addresses) | {"127.0.0.1"})
    if not (cert_f.exists() and key_f.exists() and ips_f.exists()
            and ips_f.read_text(encoding="utf-8").split() == wanted):
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "MAVIS (this computer)")])
        san = [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(a)) for a in wanted]
        now = dt.datetime.now(dt.timezone.utc)
        cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=825))
                .add_extension(x509.SubjectAlternativeName(san), critical=False)
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .sign(key, hashes.SHA256()))
        key_f.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                            serialization.NoEncryption()))
        cert_f.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        ips_f.write_text(" ".join(wanted), encoding="utf-8")
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(str(cert_f), str(key_f))
    return ctx


def cert_path() -> Path:
    return DATA / "cert.pem"
