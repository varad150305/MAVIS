"""Page reader: a cloud OCR API reads every page (Google Gemini by default, or OCR.space). The local PP-OCR reader is used
ONLY while the computer is offline.

Provider: MAVIS_OCR_PROVIDER = gemini (default) | ocrspace. Only the API layer differs (GeminiReader here, OcrSpaceReader in
ocrspace_reader.py); retries, rate-limit waiting, offline fallback and failed-page handling are shared (RemoteFirstEngine).

Rules (decided with the user):
  * every captured page is sent to the provider (several in parallel, at most the provider's RPM setting requests per minute);
  * if the computer cannot reach the provider at all (no internet), that page is read by the local engine instead, and the
    provider is tried again every 15 s until the connection is back;
  * any other failure (rate limit, server error, bad key, blocked reply) is NOT hidden behind the local reader: transient
    ones are retried with a back-off, and the page that still fails is reported as failed with the reason.

Configuration (environment; the key is never stored, logged or shown by MAVIS, and travels in a request header only):
    GEMINI_API_KEY            the user's Google AI Studio key (required)
    MAVIS_GEMINI_MODEL        model name (default below; model names change)
    MAVIS_GEMINI_PARALLEL=3   pages read at the same time
    MAVIS_GEMINI_RPM=10       requests per minute (free-tier limits differ by model)
    MAVIS_GEMINI_RETRIES=2    retries for rate limits, server errors and timeouts
    MAVIS_GEMINI_URL          API base URL (tests point it at a local mock)
    MAVIS_READER=local        development only: skip Gemini and always read locally

What leaves the computer: the page photos (JPEG). Question answering stays local (Gemma) and runs after reading finished.
Free-tier AI Studio inputs may be used by Google to improve its products.
"""
import base64
import http.client
import json
import os
import queue
import socket
import threading
import time
from urllib.parse import urlparse

import cv2

PROVIDERS = ("gemini", "ocrspace")
REMOTE_ENGINES = frozenset(PROVIDERS)       # ocr_result["engine"] values of pages read by a cloud API (text only, no boxes)
LABELS = {"gemini": "Google (Gemini)", "ocrspace": "OCR.space"}
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_URL = "https://generativelanguage.googleapis.com"
MAX_IMAGE_WIDTH = 1920
OFFLINE_REPROBE_S = 15.0
MAX_RATE_WAIT_S = 180.0        # a page waits at most this long for Gemini's rate limit to clear, then fails with the reason
MAX_RATE_RETRIES = 6           # and gives up after this many rate-limit answers in a row
SERVER_ERROR_RETRY_S = 5.0     # pause before retrying after an HTTP 5xx

PROMPT = (
    "This is a photo of a printed page. Transcribe ALL the text on the page exactly as written, in reading order "
    "(for several columns read each column from top to bottom, left to right). Keep line breaks where the text breaks "
    "lines. Do not summarise, translate, correct, explain or add anything. Ignore anything that is not part of the page: "
    "the small 'using droidcam.app' overlay, keyboard keys, the table, fingers. If a word cannot be read write [illegible]. "
    "Reply with the transcription only; if there is no text, reply with nothing."
)


class ReaderUnavailable(Exception):
    """Gemini could not read this page.

    network     the request never got a reply (no connection, timeout, reset): offline or a blip, decided by a probe
    retryable   worth trying again (rate limit, server error, timeout); False for a bad key, a blocked image, a bad request
    retry_after seconds the service asked us to wait
    """

    def __init__(self, reason, retry_after=None, network=False, retryable=False, rate_limited=False):
        super().__init__(reason)
        self.reason, self.retry_after, self.network, self.retryable = reason, retry_after, network, retryable
        self.rate_limited = rate_limited         # HTTP 429 that will clear by itself: wait and retry, do not fail the page


class ReaderSettings:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.provider = (env.get("MAVIS_OCR_PROVIDER") or "gemini").strip().lower()
        if self.provider not in PROVIDERS:
            raise ValueError(f"MAVIS_OCR_PROVIDER must be one of {', '.join(PROVIDERS)} (got {self.provider!r})")
        self.mode = (env.get("MAVIS_READER") or "api").strip().lower()          # "local" is a development override
        self.label = LABELS[self.provider]
        retries = env.get("MAVIS_OCR_RETRIES") or env.get("MAVIS_GEMINI_RETRIES") or 2
        self.retries = max(0, min(5, int(retries)))
        if self.provider == "gemini":
            self.key_env = "GEMINI_API_KEY"
            self.model = (env.get("MAVIS_GEMINI_MODEL") or DEFAULT_MODEL).strip()
            self.base_url = (env.get("MAVIS_GEMINI_URL") or DEFAULT_URL).rstrip("/")
            self.parallel = max(1, min(8, int(env.get("MAVIS_GEMINI_PARALLEL") or 3)))
            self.rpm = max(1.0, float(env.get("MAVIS_GEMINI_RPM") or 10))
            # Gemini 2.5 models "think" before answering, which made a page take 7 to 35 s. Transcription needs no thinking.
            # 0 switches it off; a negative number leaves the model's default.
            self.thinking_budget = int(env.get("MAVIS_GEMINI_THINKING_BUDGET") or 0)
            self.setup_hint = ("Create a key in Google AI Studio, run  setx GEMINI_API_KEY \"...\"  in a terminal, then restart "
                               "MAVIS from a new terminal.")
        else:
            self.key_env = "OCRSPACE_API_KEY"
            self.base_url = (env.get("MAVIS_OCRSPACE_URL") or "https://api.ocr.space").rstrip("/")
            self.ocr_engine = (env.get("MAVIS_OCRSPACE_ENGINE") or "2").strip()          # 2 = default, 3 = most accurate
            self.language = (env.get("MAVIS_OCRSPACE_LANGUAGE") or "eng").strip()        # "auto" = let OCR.space detect
            self.table = (env.get("MAVIS_OCRSPACE_TABLE") or "0").strip().lower() in ("1", "true", "yes")
            self.parallel = max(1, min(8, int(env.get("MAVIS_OCRSPACE_PARALLEL") or 2)))
            self.rpm = max(1.0, float(env.get("MAVIS_OCRSPACE_RPM") or 20))
            self.model = f"engine {self.ocr_engine}"
            self.setup_hint = ("Get a free key at https://ocr.space/ocrapi/freekey, run  setx OCRSPACE_API_KEY \"...\"  and  "
                               "setx MAVIS_OCR_PROVIDER ocrspace  in a terminal, then restart MAVIS from a new terminal.")
        self._key = (env.get(self.key_env) or "").strip()

    @property
    def local_only(self):
        return self.mode == "local"

    @property
    def has_key(self):
        return bool(self._key)

    def api_key(self):
        return self._key

    def public(self):
        """What the browser may see. Never the key."""
        if self.local_only:
            return {"provider": "local (development)", "remote": False, "label": "this computer", "model": None, "has_key": False,
                    "key_env": None, "setup": ""}
        return {"provider": self.provider, "remote": True, "label": self.label, "model": self.model, "has_key": self.has_key,
                "key_env": self.key_env, "setup": self.setup_hint}


def _host_port(base_url):
    url = urlparse(base_url)
    return url.hostname, url.port or (443 if url.scheme == "https" else 80), url.scheme == "https"


def online_probe(base_url, timeout=3.0):
    """True if a TCP connection to the API host can be opened (the computer has a route to Google). Sends no data."""
    host, port, _ = _host_port(base_url)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class GeminiReader:
    def __init__(self, settings, timeout_s=90):
        self.settings = settings
        self.timeout_s = timeout_s
        self._host, self._port, self._https = _host_port(settings.base_url)
        self._base_path = urlparse(settings.base_url).path.rstrip("/")
        self._send_thinking_config = settings.thinking_budget >= 0     # dropped for good if the model rejects it

    @staticmethod
    def _retry_delay(headers_value, body):
        """Seconds Google asked us to wait: Retry-After header, else error.details[].retryDelay ('34s'), else None."""
        if headers_value and str(headers_value).isdigit():
            return float(headers_value)
        try:
            for detail in (json.loads(body).get("error") or {}).get("details") or []:
                delay = detail.get("retryDelay")
                if delay:
                    return float(str(delay).rstrip("s"))
        except (ValueError, TypeError, AttributeError):
            pass
        return None

    def _connection(self, timeout):
        cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        return cls(self._host, self._port, timeout=timeout)

    def probe(self):
        return online_probe(self.settings.base_url)

    def check(self):
        """(status, message) of the key and model, without sending any page: status is ok, missing_key, rejected,
        model_missing, offline or unreachable."""
        if not self.settings.has_key:
            return "missing_key", "GEMINI_API_KEY is not set."
        conn = self._connection(10)
        try:
            conn.request("GET", f"{self._base_path}/v1beta/models/{self.settings.model}",
                         headers={"x-goog-api-key": self.settings.api_key()})
            resp = conn.getresponse()
            resp.read()
        except (OSError, http.client.HTTPException):
            return ("unreachable", "Google could not be reached.") if self.probe() else \
                   ("offline", "No internet connection.")
        finally:
            conn.close()
        if resp.status == 200:
            return "ok", ""
        if resp.status in (400, 401, 403):
            return "rejected", f"Google rejected the API key (HTTP {resp.status})."
        if resp.status == 404:
            return "model_missing", f"The model '{self.settings.model}' was not found. Set MAVIS_GEMINI_MODEL to a model your key can use."
        return "unreachable", f"Google returned HTTP {resp.status}."

    def _post(self, image_b64):
        generation = {"temperature": 0, "maxOutputTokens": 8192}
        if self._send_thinking_config:
            generation["thinkingConfig"] = {"thinkingBudget": self.settings.thinking_budget}
        body = json.dumps({"contents": [{"parts": [{"text": PROMPT},
                                                   {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}]}],
                           "generationConfig": generation}).encode("utf-8")
        path = f"{self._base_path}/v1beta/models/{self.settings.model}:generateContent"
        conn = self._connection(self.timeout_s)
        try:
            # the key travels in a header, never in the URL, so it cannot end up in an error message or a log line
            conn.request("POST", path, body=body, headers={"Content-Type": "application/json",
                                                           "x-goog-api-key": self.settings.api_key()})
            resp = conn.getresponse()
            return resp, resp.read().decode("utf-8", "replace")
        except (OSError, http.client.HTTPException) as exc:
            raise ReaderUnavailable(f"network error ({type(exc).__name__})", network=True, retryable=True) from None
        finally:
            conn.close()

    def read(self, frame):
        """Transcribe one BGR frame. Raises ReaderUnavailable when it cannot."""
        if not self.settings.has_key:
            raise ReaderUnavailable("GEMINI_API_KEY is not set")
        h, w = frame.shape[:2]
        if w > MAX_IMAGE_WIDTH:
            frame = cv2.resize(frame, (MAX_IMAGE_WIDTH, int(h * MAX_IMAGE_WIDTH / w)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise ReaderUnavailable("could not encode the frame")
        image_b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
        resp, raw = self._post(image_b64)
        if resp.status == 400 and self._send_thinking_config and "thinking" in raw.lower():
            self._send_thinking_config = False                    # this model cannot switch thinking off: ask without it
            resp, raw = self._post(image_b64)
        if resp.status == 429:
            delay = self._retry_delay(resp.getheader("Retry-After"), raw)
            if "perday" in raw.lower().replace(" ", "").replace("_", ""):
                raise ReaderUnavailable("Gemini's free daily quota for this model is used up (HTTP 429); it resets daily, "
                                        "or set MAVIS_GEMINI_MODEL to another model")
            raise ReaderUnavailable("rate limit reached (HTTP 429)", delay if delay is not None else 30.0,
                                    retryable=True, rate_limited=True)
        if resp.status in (401, 403):
            raise ReaderUnavailable(f"the API key was rejected (HTTP {resp.status})")
        if resp.status >= 500:
            raise ReaderUnavailable(f"Gemini returned HTTP {resp.status}", SERVER_ERROR_RETRY_S, retryable=True)
        if resp.status >= 400:
            raise ReaderUnavailable(f"Gemini returned HTTP {resp.status}")
        try:
            data = json.loads(raw)
        except ValueError:
            raise ReaderUnavailable("unreadable reply from Gemini", retryable=True) from None
        block = (data.get("promptFeedback") or {}).get("blockReason")
        if block:
            raise ReaderUnavailable(f"the image was blocked ({block})")
        cands = data.get("candidates") or []
        if not cands:
            raise ReaderUnavailable("Gemini returned no answer", retryable=True)
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()
        if cands[0].get("finishReason") in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT"):
            raise ReaderUnavailable(f"the reply was withheld ({cands[0]['finishReason']})")
        return text


class RemoteFirstEngine:
    """Same interface as ocr_engine.PrototypeOCREngine. Each page goes to Gemini (several in parallel); the local engine is
    used only while the computer is offline. Results are (page_number, text, metrics, ocr_result, error)."""

    OCR_QUEUE_MAXSIZE = 8
    CONFIDENCE_THRESHOLD = 0.50

    def __init__(self, reader, local_factory, parallel=3, rpm=10.0, retries=2, probe=None, backoff_scale=1.0,
                 provider="gemini", label="Gemini"):
        self.provider, self.label = provider, label                  # which cloud API, and how the UI names it
        self._reader = reader
        self._local_factory, self._local = local_factory, None      # created on first use: no local OCR load while online
        self._parallel = parallel
        self._interval = 60.0 / rpm
        self._retries = retries
        self._probe = probe or reader.probe
        self._backoff_scale = backoff_scale
        self._jobs = queue.Queue(maxsize=self.OCR_QUEUE_MAXSIZE)
        self._results = queue.Queue()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._busy = 0
        self._next_request_at = 0.0
        self._paused_until = 0.0
        self._offline = False
        self._next_probe_at = 0.0
        self._local_lock = threading.Lock()          # the local reader is shared and CPU-heavy: one page at a time

    # ---- interface used by ScanSession
    def submit(self, page_number, crop, metrics):
        metrics["queue_entered_at"] = time.perf_counter()
        try:
            self._jobs.put_nowait((page_number, crop, metrics))
            metrics["ocr_queue_depth"] = self._jobs.qsize()
            return True
        except queue.Full:
            return False

    def has_capacity(self):
        return not self._jobs.full()

    def queue_depth(self):
        return self._jobs.qsize()

    def is_processing(self):
        with self._lock:
            return self._busy > 0

    def is_offline(self):
        return self._offline

    def poll_result(self):
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stop_event.set()

    def wait(self):
        self._jobs.join()

    def _reconstruct_layout(self, ocr_data):
        return self._local_engine()._reconstruct_layout(ocr_data)

    def worker_loop(self):
        others = [threading.Thread(target=self._work, name=f"web-read-{i}", daemon=True) for i in range(1, self._parallel)]
        for t in others:
            t.start()
        self._work()
        for t in others:
            t.join()

    # ---- internals
    def _work(self):
        while not self._stop_event.is_set() or not self._jobs.empty():
            try:
                page_number, crop, metrics = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                self._busy += 1
            try:
                self._results.put(self._read_page(page_number, crop, metrics))
            finally:
                with self._lock:
                    self._busy -= 1
                self._jobs.task_done()

    def rate_limited_until(self):
        """Epoch seconds until which the service asked us to wait, or None. For the scan screen."""
        with self._lock:
            return self._paused_until if self._paused_until > time.time() else None

    def _wait_for_slot(self, metrics):
        """Space request starts by 60/rpm seconds and honour a rate-limit pause. A page waits for the quota (the scan screen
        says so) instead of failing; it fails only if Gemini asks for longer than MAX_RATE_WAIT_S."""
        with self._lock:
            now = time.time()
            start = now
            if now < self._paused_until:
                remaining = self._paused_until - now
                if remaining > MAX_RATE_WAIT_S:
                    raise ReaderUnavailable(f"rate limit: {self.label} asked us to wait {remaining:.0f} s, longer than MAVIS "
                                            "waits. Read this page again later")
                start = self._paused_until
                metrics["rate_limit_wait_s"] = metrics.get("rate_limit_wait_s", 0.0) + remaining
            start = max(start, self._next_request_at)
            self._next_request_at = start + self._interval
        if start > now:
            time.sleep(start - now)

    def _is_offline(self):
        """Offline mode: pages go to the local reader; the service is probed again every OFFLINE_REPROBE_S seconds."""
        with self._lock:
            if not self._offline:
                return False
            if time.time() < self._next_probe_at:
                return True
        online = self._probe()
        with self._lock:
            if online:
                self._offline = False
            else:
                self._next_probe_at = time.time() + OFFLINE_REPROBE_S
            return self._offline

    def _read_page(self, page_number, crop, metrics):
        metrics["ocr_started_at"] = time.perf_counter()
        metrics["queue_wait_ms"] = (metrics["ocr_started_at"] - metrics["queue_entered_at"]) * 1000.0
        if self._is_offline():
            return self._read_locally(page_number, crop, metrics, "no internet connection")
        attempt = rate_attempts = 0
        while True:
            try:
                self._wait_for_slot(metrics)
                started = time.perf_counter()
                text = self._reader.read(crop)
                metrics.update(ocr_inference_ms=(time.perf_counter() - started) * 1000.0, layout_reconstruction_ms=0.0,
                               ocr_finished_at=time.perf_counter(), reader=self.provider, attempts=attempt + 1)
                return page_number, text, metrics, {"raw_text": text, "detections": [], "engine": self.provider,
                                                    "page_number": page_number}, None
            except ReaderUnavailable as exc:
                if exc.network and not self._probe():
                    with self._lock:
                        self._offline, self._next_probe_at = True, time.time() + OFFLINE_REPROBE_S
                    return self._read_locally(page_number, crop, metrics, "no internet connection")
                if exc.retry_after and exc.retryable and exc.network is False:
                    with self._lock:
                        self._paused_until = max(self._paused_until, time.time() + exc.retry_after)
                if exc.rate_limited:
                    # a rate limit clears by itself: slow the pace for the rest of the session and wait, do not fail the page
                    with self._lock:
                        self._interval = min(self._interval * 1.5, 30.0)
                    rate_attempts += 1
                    if rate_attempts <= MAX_RATE_RETRIES:
                        continue
                if not exc.retryable or attempt >= self._retries:
                    metrics.update(error=exc.reason, ocr_finished_at=time.perf_counter(), attempts=attempt + 1)
                    return page_number, "", metrics, None, exc
                attempt += 1
                if exc.network:                                  # online but the request failed: short pause, try again
                    time.sleep(1.0 * attempt * self._backoff_scale)
            except Exception as exc:  # noqa: BLE001 - never let one page take a worker down
                metrics.update(error=str(exc), ocr_finished_at=time.perf_counter())
                return page_number, "", metrics, None, exc

    def _local_engine(self):
        if self._local is None:
            self._local = self._local_factory()
        return self._local

    def _read_locally(self, page_number, crop, metrics, reason):
        metrics["fallback_reason"] = reason
        try:
            started = time.perf_counter()
            with self._local_lock:
                local = self._local_engine()
                from ocr_engine import PrototypeOCREngine
                if type(local).ocr is PrototypeOCREngine.ocr:       # the real local engine (not a test double)
                    from webapp.ocr_threads import install_reader
                    install_reader()                  # first offline page: build the reader with capped threads
                prepared, _ = local._prepare_crop(crop)
                ocr_result, inference_ms, layout_ms = local.ocr(prepared)
            metrics.update(ocr_inference_ms=inference_ms, layout_reconstruction_ms=layout_ms,
                           ocr_finished_at=time.perf_counter(), reader="local")
            ocr_result["page_number"] = page_number
            return page_number, ocr_result["raw_text"], metrics, ocr_result, None
        except Exception as exc:  # noqa: BLE001
            metrics["error"] = str(exc)
            metrics["ocr_finished_at"] = time.perf_counter()
            return page_number, "", metrics, None, exc


def make_reader(settings):
    """The API client for the chosen provider (both have read(frame), check() and probe())."""
    if settings.provider == "ocrspace":
        from webapp.ocrspace_reader import OcrSpaceReader
        return OcrSpaceReader(settings)
    return GeminiReader(settings)


def make_engine_factory(settings, local_factory):
    """engine_factory for ScanSession: the chosen cloud API first with the local engine as the offline fallback (or local only
    with the development override MAVIS_READER=local)."""
    if settings.local_only:
        return local_factory
    reader = make_reader(settings)
    return lambda: RemoteFirstEngine(reader, local_factory, parallel=settings.parallel, rpm=settings.rpm,
                                     retries=settings.retries, provider=settings.provider, label=settings.label)
