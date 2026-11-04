"""Run the MAVIS web app:  py -3.11 -m webapp  [--port 8765] [--no-browser]"""
import argparse
import threading
import time
import webbrowser

from webapp.server import serve


def main():
    ap = argparse.ArgumentParser(description="MAVIS - local AI document assistant")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[MAVIS] web app on {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.2), webbrowser.open(url)), daemon=True).start()
    serve(port=args.port)


if __name__ == "__main__":
    main()
