"""OCR.space as the page reader (https://ocr.space/ocrapi). Same interface as GeminiReader: read(frame), check(), probe().

Request (documented): POST {base}/parse/image, multipart/form-data, API key in the `apikey` header; form fields `OCREngine`
(2 = default, 3 = most accurate and slower, with its own quota), `language`, `scale`, `detectOrientation`, `isTable`,
`isOverlayRequired`, and the image as the `file` part. The free plan accepts files up to 1 MB and 500 requests a day per IP
(25,000 a month), so each photo is re-encoded to stay under 900 KB. The key travels in the header only.

Response: {"ParsedResults": [{"ParsedText", "FileParseExitCode", "ErrorMessage"}], "OCRExitCode", "IsErroredOnProcessing",
"ErrorMessage"}. The free API reports most problems inside a 200 reply, so errors are classified from those fields:
rate-limit text ("... within 600 seconds") -> wait and retry, daily limit -> fail, bad key -> fail, timeout -> retry.

OCR.space has no key-check call, and a test image would use up one of the 500 daily requests, so check() only confirms that
the key is set and the service is reachable; a rejected key shows up as an error on the first page.
"""
import http.client
import json
import re
import uuid

import cv2

from webapp.gemini_reader import ReaderUnavailable, _host_port, online_probe

MAX_UPLOAD_BYTES = 900_000
MAX_WIDTH = 1600


def encode_under_limit(frame, limit=MAX_UPLOAD_BYTES):
    """JPEG bytes of the frame that fit the free plan's 1 MB file limit (width and quality are lowered until they do)."""
    width = min(frame.shape[1], MAX_WIDTH)
    for quality in (88, 80, 72, 64, 55):
        scaled = frame
        if frame.shape[1] > width:
            scaled = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", scaled, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok and len(buf) <= limit:
            return buf.tobytes()
        if quality == 64:
            width = int(width * 0.75)                  # still too big: also shrink the picture
    raise ReaderUnavailable("the photo could not be compressed under OCR.space's 1 MB limit")


def _multipart(fields, file_field, filename, content_type, data):
    boundary = "----mavis" + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
                 f"Content-Type: {content_type}\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return boundary, b"".join(parts)


def _message(value):
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    return str(value or "")


class OcrSpaceReader:
    def __init__(self, settings, timeout_s=90):
        self.settings = settings
        self.timeout_s = timeout_s
        self._host, self._port, self._https = _host_port(settings.base_url)

    def probe(self):
        return online_probe(self.settings.base_url)

    def check(self):
        if not self.settings.has_key:
            return "missing_key", f"{self.settings.key_env} is not set."
        if not self.probe():
            return "offline", "No internet connection."
        return "ok", "The key is checked when the first page is read."

    def read(self, frame):
        """Transcribe one BGR frame. Raises ReaderUnavailable when it cannot."""
        if not self.settings.has_key:
            raise ReaderUnavailable(f"{self.settings.key_env} is not set")
        fields = {"OCREngine": self.settings.ocr_engine, "scale": "true", "detectOrientation": "true",
                  "isOverlayRequired": "false", "isTable": "true" if self.settings.table else "false"}
        if self.settings.language and self.settings.language != "auto":
            fields["language"] = self.settings.language
        boundary, body = _multipart(fields, "file", "page.jpg", "image/jpeg", encode_under_limit(frame))
        cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        conn = cls(self._host, self._port, timeout=self.timeout_s)
        try:
            conn.request("POST", "/parse/image", body=body,
                         headers={"apikey": self.settings.api_key(), "Content-Type": f"multipart/form-data; boundary={boundary}"})
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", "replace")
        except (OSError, http.client.HTTPException) as exc:
            raise ReaderUnavailable(f"network error ({type(exc).__name__})", network=True, retryable=True) from None
        finally:
            conn.close()
        if resp.status == 429:
            retry = resp.getheader("Retry-After")
            raise ReaderUnavailable("rate limit reached (HTTP 429)", float(retry) if retry and retry.isdigit() else 60.0,
                                    retryable=True, rate_limited=True)
        if resp.status in (401, 403):
            raise ReaderUnavailable(f"OCR.space rejected the API key or blocked this address (HTTP {resp.status})")
        if resp.status >= 500:
            raise ReaderUnavailable(f"OCR.space returned HTTP {resp.status}", 5.0, retryable=True)
        try:
            data = json.loads(raw)
        except ValueError:
            raise ReaderUnavailable("unreadable reply from OCR.space", retryable=True) from None
        if data.get("IsErroredOnProcessing") or data.get("OCRExitCode") in (3, 4):
            self._raise_for(_message(data.get("ErrorMessage")) or _message(
                (data.get("ParsedResults") or [{}])[0].get("ErrorMessage")), data)
        results = data.get("ParsedResults") or []
        if not results:
            raise ReaderUnavailable("OCR.space returned no result", retryable=True)
        first = results[0]
        code = first.get("FileParseExitCode")
        if code not in (None, 1):
            self._raise_for(_message(first.get("ErrorMessage")) or f"parse error {code}", data, code)
        # OCR.space separates lines with CRLF
        return (first.get("ParsedText") or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    @staticmethod
    def _raise_for(message, data, parse_code=None):
        low = message.lower()
        if "per day" in low or "daily" in low:
            raise ReaderUnavailable("OCR.space's daily request limit is used up; it resets daily")
        if "within" in low and "second" in low or "too many" in low or "rate" in low and "limit" in low or "maximum" in low and "times" in low:
            match = re.search(r"within\s+(\d+)\s*seconds", low)
            raise ReaderUnavailable(f"rate limit reached ({message.strip()[:120]})", float(match.group(1)) if match else 60.0,
                                    retryable=True, rate_limited=True)
        if "api key" in low or "apikey" in low or "invalid key" in low:
            raise ReaderUnavailable(f"OCR.space rejected the API key ({message.strip()[:120]})")
        if "size" in low and ("limit" in low or "exceed" in low):
            raise ReaderUnavailable(f"the photo is larger than OCR.space allows ({message.strip()[:120]})")
        if parse_code in (-20, -10) or "timed out" in low or "timeout" in low:
            raise ReaderUnavailable(f"OCR.space could not finish this page ({message.strip()[:120] or parse_code})", 3.0,
                                    retryable=True)
        raise ReaderUnavailable(f"OCR.space could not read the page ({message.strip()[:160] or 'unknown error'})")
