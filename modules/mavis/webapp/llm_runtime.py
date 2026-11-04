"""Ollama client whose in-flight requests can be aborted (used when a new scan must take over the machine).

Subclasses llm_stage.OllamaRuntime and replaces only the transport: each request runs on its own http.client
connection, and abort_all() shuts those sockets down, which makes the blocked call return at once and tells Ollama to stop
generating (it stops when the client disconnects). The aborted call raises LLMError("cancelled").
"""
import ctypes
import http.client
import json
import os
import socket
import sys
import threading
from urllib.parse import urlparse

from llm_stage import LLMError, OllamaRuntime


DEFAULT_THREADS = 8


class CancellableOllamaRuntime(OllamaRuntime):
    """`threads` caps the CPU threads the model may use (MAVIS_LLM_THREADS, 0 = Ollama's default, i.e. every core).

    Measured here (Core Ultra 5 125H, 18 logical CPUs, Gemma 3 4B): 4 threads 7.2 tokens/s, 6 -> 9.1, 8 -> 11.0, 12 -> 12.6,
    default 11.8. With every core busy for a minute or more the browser and this server stall, so the page seemed to hang;
    8 threads costs about 5 % of the speed and leaves 10 logical CPUs free."""

    def __init__(self, cfg, threads=None):
        super().__init__(cfg)
        self.threads = int(os.environ.get("MAVIS_LLM_THREADS", DEFAULT_THREADS)) if threads is None else threads
        self._conn_lock = threading.Lock()
        self._conns = set()
        self._generation = 0

    def abort_all(self):
        """Abort every request in flight; requests started afterwards are not affected."""
        with self._conn_lock:
            self._generation += 1
            conns = list(self._conns)
        for conn in conns:
            sock = conn.sock
            if sock is None:
                continue
            if sys.platform == "win32":
                # On Windows neither shutdown() nor close() wakes a thread blocked in recv(); closing the handle does.
                # detach() first so Python never closes this (possibly reused) handle a second time.
                try:
                    handle = sock.detach()
                    ctypes.windll.ws2_32.closesocket(ctypes.c_size_t(handle))
                except (OSError, AttributeError):
                    pass
            else:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        return len(conns)

    def _post(self, path, payload, timeout):
        if self.threads > 0 and path in ("/api/chat", "/api/generate"):
            # the same value on every request (warm-up, priming, questions), so Ollama never reloads the model for it
            payload = {**payload, "options": {**(payload.get("options") or {}), "num_thread": self.threads}}
        url = urlparse(self.cfg.base_url)
        conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
        with self._conn_lock:
            generation = self._generation
            self._conns.add(conn)
        try:
            conn.request("POST", path, body=json.dumps(payload).encode("utf-8"),
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", "replace")
        except (OSError, http.client.HTTPException) as exc:
            if generation != self._generation:
                raise LLMError("cancelled", "The assistant was stopped.") from exc
            if isinstance(exc, (TimeoutError, socket.timeout)) or "timed out" in str(exc):
                raise LLMError("timeout", f"The model did not answer within {timeout:.0f} s (MAVIS_LLM_TIMEOUT_S).")
            raise LLMError("runtime_unavailable",
                           f"Cannot reach the local LLM runtime at {self.cfg.base_url} ({exc}). "
                           "Install and start Ollama (https://ollama.com) or set MAVIS_LLM_URL.") from exc
        finally:
            with self._conn_lock:
                self._conns.discard(conn)
            conn.close()
        if generation != self._generation:
            raise LLMError("cancelled", "The assistant was stopped.")
        if resp.status >= 400:
            if resp.status == 404 and "not found" in body.lower():
                raise LLMError("model_missing",
                               f"Model '{self.cfg.model}' is not installed in Ollama. Install it with: "
                               f"ollama pull {self.cfg.model}   (or set MAVIS_LLM_MODEL to a model you have).")
            raise LLMError("runtime_error", f"Ollama returned HTTP {resp.status}: {body[:300]}")
        try:
            return json.loads(body)
        except ValueError as exc:
            raise LLMError("bad_output", "The local LLM runtime sent an unreadable reply.") from exc
