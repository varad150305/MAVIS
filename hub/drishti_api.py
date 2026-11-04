"""Drishti's three serverless proxies (drishti-web/api/groq.js, sarvam.js, cloudflare.js), served by the hub.

On Vercel those .js files run as serverless functions. When Drishti runs inside the suite there is no Vercel, so these
routes give the same behaviour at the same URLs (/api/groq, /api/sarvam, /api/cloudflare):
  GET   -> {"ok": true, "configured": <is the key set on this computer>}   (Drishti checks this on page load)
  POST  -> relays the request body to the provider with the key from the environment
Same per-visitor limit as the originals: 40 requests a minute. The original .js files are unchanged and still used when
Drishti is deployed to Vercel on its own.

Keys (environment variables, same names as on Vercel): GROQ_API_KEY, SARVAM_API_KEY,
CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN (optional; otherwise the visitor's own, sent in X-CF-* headers).
"""
from __future__ import annotations

import os
import time

import aiohttp
from aiohttp import web

CORS = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"}
CORS_CF = {**CORS, "Access-Control-Allow-Headers": "Content-Type, X-CF-Account, X-CF-Token, X-CF-Model"}


class RateLimiter:
    def __init__(self, window_s=60.0, max_requests=40):
        self.window, self.max, self.buckets = window_s, max_requests, {}

    def limited(self, request) -> bool:
        ip = (request.headers.get("X-Forwarded-For") or request.remote or "unknown").split(",")[0].strip()
        now = time.monotonic()
        start, count = self.buckets.get(ip, (now, 0))
        if now - start > self.window:
            start, count = now, 0
        self.buckets[ip] = (start, count + 1)
        if len(self.buckets) > 5000:
            self.buckets = {k: v for k, v in self.buckets.items() if now - v[0] <= self.window}
        return count + 1 > self.max


def _j(data, status=200, headers=CORS):
    return web.json_response(data, status=status, headers=headers)


async def _relay(url, headers, body):
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as s:
            async with s.post(url, json=body, headers=headers) as r:
                try:
                    data = await r.json(content_type=None)
                except Exception:  # noqa: BLE001
                    data = {"error": "bad_upstream_response", "text": (await r.text())[:500]}
                return data, r.status
    except Exception:  # noqa: BLE001
        return {"error": "upstream_unreachable"}, 502


async def _body(request):
    try:
        return await request.json()
    except Exception:  # noqa: BLE001
        return {}


def add_routes(app: web.Application):
    limits = {"groq": RateLimiter(), "sarvam": RateLimiter(), "cloudflare": RateLimiter()}

    async def groq(request):
        if request.method == "OPTIONS":
            return web.Response(headers=CORS)
        key = os.environ.get("GROQ_API_KEY")
        if request.method == "GET":
            return _j({"ok": True, "configured": bool(key)})
        if limits["groq"].limited(request):
            return _j({"error": "rate_limited"}, 429)
        if not key:
            return _j({"error": "proxy_not_configured"}, 500)
        data, status = await _relay("https://api.groq.com/openai/v1/chat/completions",
                                    {"Authorization": "Bearer " + key}, await _body(request))
        return _j(data, status)

    async def sarvam(request):
        if request.method == "OPTIONS":
            return web.Response(headers=CORS)
        key = os.environ.get("SARVAM_API_KEY")
        if request.method == "GET":
            return _j({"ok": True, "configured": bool(key)})
        if limits["sarvam"].limited(request):
            return _j({"error": "rate_limited"}, 429)
        if not key:
            return _j({"error": "proxy_not_configured"}, 500)
        data, status = await _relay("https://api.sarvam.ai/text-to-speech",
                                    {"api-subscription-key": key}, await _body(request))
        return _j(data, status)

    async def cloudflare(request):
        if request.method == "OPTIONS":
            return web.Response(headers=CORS_CF)
        if request.method == "GET":
            configured = bool(os.environ.get("CLOUDFLARE_ACCOUNT_ID") and os.environ.get("CLOUDFLARE_API_TOKEN"))
            return _j({"ok": True, "configured": configured}, headers=CORS_CF)
        if limits["cloudflare"].limited(request):
            return _j({"error": "rate_limited"}, 429, CORS_CF)
        account = os.environ.get("CLOUDFLARE_ACCOUNT_ID") or request.headers.get("X-CF-Account")
        token = os.environ.get("CLOUDFLARE_API_TOKEN") or request.headers.get("X-CF-Token")
        model = request.headers.get("X-CF-Model")
        if not account or not token:
            return _j({"error": "missing_credentials"}, 400, CORS_CF)
        if not model:
            return _j({"error": "missing_model"}, 400, CORS_CF)
        data, status = await _relay(f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}",
                                    {"Authorization": "Bearer " + token}, await _body(request))
        return _j(data, status, CORS_CF)

    for path, handler in (("/api/groq", groq), ("/api/sarvam", sarvam), ("/api/cloudflare", cloudflare)):
        app.router.add_route("*", path, handler)


def status() -> dict:
    return {"groq": bool(os.environ.get("GROQ_API_KEY")), "sarvam": bool(os.environ.get("SARVAM_API_KEY"))}
