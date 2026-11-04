"""Start MAVIS:   python -m hub   [--port 8000] [--https-port 8443] [--no-lan] [--no-browser]"""
import argparse
import sys

from .server import Hub, port_free, serve


def main():
    ap = argparse.ArgumentParser(description="MAVIS - Drishti, MAVIS OCR and Gesture control in one app")
    ap.add_argument("--port", type=int, default=8000, help="port for this computer (http://127.0.0.1:PORT)")
    ap.add_argument("--https-port", type=int, default=8443, help="port for phones on the same Wi-Fi (HTTPS)")
    ap.add_argument("--relay-port", type=int, default=8766, help="internal port (phone camera -> MAVIS)")
    ap.add_argument("--no-lan", action="store_true", help="this computer only; no phone access")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    a = ap.parse_args()
    for p in (a.port, a.relay_port) + (() if a.no_lan else (a.https_port,)):
        if not port_free(p, "127.0.0.1"):
            sys.exit(f"Port {p} is already in use (is MAVIS already running?). Pick another with --port / "
                     f"--https-port / --relay-port.")
    serve(Hub(a.port, a.https_port, a.relay_port, lan=not a.no_lan), open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
