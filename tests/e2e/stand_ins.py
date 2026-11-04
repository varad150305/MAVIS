"""TEST STAND-INS for the end-to-end run, used ONLY when the real services are not available (no Gemini key, no Ollama).

    python tests/e2e/stand_ins.py --port 18080
    set MAVIS_GEMINI_URL=http://127.0.0.1:18080   GEMINI_API_KEY=test   MAVIS_LLM_URL=http://127.0.0.1:18080

* Gemini protocol (GET /v1beta/models/<m>, POST /v1beta/models/<m>:generateContent): the page photo is read with
  Tesseract, a real OCR engine, so the text MAVIS shows really comes from the scanned picture. Accuracy is
  Tesseract's, not Gemini's.
* Ollama protocol (GET /api/tags, POST /api/chat, /api/generate): NOT a language model. For a question it quotes the
  sentence of the document that shares the most words with the question, in MAVIS's answer schema (page + exact
  quote), so MAVIS's question flow, quote checking and citations can be exercised. Real answers need Ollama + gemma3:4b.

MAVIS's code is not changed for this: it is pointed at these URLs with its own documented settings
(MAVIS_GEMINI_URL, MAVIS_LLM_URL; see modules/mavis/docs/SETUP.md).
"""
import argparse
import base64
import json
import re
import subprocess
import tempfile

from aiohttp import web

STOP = set("the a an of to in on for and or is are was were be what which who how why when where does do did this that "
           "it its with as by at from about please tell me document page".split())


def tesseract(jpeg: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        f.write(jpeg)
        f.flush()
        out = subprocess.run(["tesseract", f.name, "-", "--psm", "3"], capture_output=True, text=True, timeout=60)
    return out.stdout.strip()


async def gemini_model(_r):
    return web.json_response({"name": "models/stand-in", "displayName": "Tesseract stand-in"})


async def gemini_generate(request):
    body = await request.json()
    img = None
    for part in body["contents"][0]["parts"]:
        d = part.get("inline_data") or part.get("inlineData")
        if d:
            img = base64.b64decode(d["data"])
    text = tesseract(img) if img else ""
    return web.json_response({"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
                              "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}})


def _pages(user: str):
    return [(int(n), t) for n, t in re.findall(r"=== PAGE (\d+) ===\n(.*?)(?=\n=== PAGE \d+ ===|\Z)", user, re.S)]


def _sentences(text):
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = s.strip()
        if len(s.split()) >= 3:
            yield s


def _words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1}


def answer(system: str, user: str, schema: dict) -> dict:
    props = (schema or {}).get("properties", {})
    if "corrected_text" in props:                                  # "recover" stage: keep the OCR text as it is
        m = re.search(r"<<<\n(.*)\n>>>", user, re.S)
        return {"corrected_text": m.group(1) if m else "", "uncertain_spans": []}
    m = re.search(r"\n\s*(?:USER REQUEST|QUESTION|Question)\s*:\s*(.+)", user)
    q = m.group(1).strip() if m else user.strip().splitlines()[-1]
    pages = _pages(user[:m.start()] if m else user)
    qw = _words(q)
    best = (0, None, None)
    for n, text in pages:
        for s in _sentences(text):
            score = len(qw & _words(s))
            if score > best[0]:
                best = (score, n, s)
    if best[1] is None:
        if pages and "summar" in (system + q).lower():
            n, text = pages[0]
            s = next(_sentences(text), text[:120])
            return {"answerable_from_document": True, "answer": s, "general_knowledge": "",
                    "evidence": [{"page": n, "quote": s}]}
        return {"answerable_from_document": False, "answer": "The document does not say.", "general_knowledge": "",
                "evidence": []}
    return {"answerable_from_document": True, "answer": best[2], "general_knowledge": "",
            "evidence": [{"page": best[1], "quote": best[2]}]}


async def ollama_tags(_r):
    return web.json_response({"models": [{"name": "gemma3:4b"}]})


async def ollama_chat(request):
    p = await request.json()
    msgs = p.get("messages") or []
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), p.get("prompt", ""))
    fmt = p.get("format") if isinstance(p.get("format"), dict) else {}
    content = json.dumps(answer(system, user, fmt)) if fmt else "ok"
    return web.json_response({"message": {"role": "assistant", "content": content}, "response": content, "done": True,
                              "done_reason": "stop", "total_duration": 1e8, "load_duration": 0,
                              "prompt_eval_count": len(user) // 4, "prompt_eval_duration": 5e7,
                              "eval_count": len(content) // 4, "eval_duration": 5e7})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18080)
    a = ap.parse_args()
    app = web.Application(client_max_size=20 * 1024 ** 2)
    app.router.add_get("/v1beta/models/{m}", gemini_model)
    app.router.add_post("/v1beta/models/{m}", gemini_generate)          # ".../<model>:generateContent"
    app.router.add_get("/api/tags", ollama_tags)
    app.router.add_post("/api/chat", ollama_chat)
    app.router.add_post("/api/generate", ollama_chat)
    web.run_app(app, host="127.0.0.1", port=a.port, print=None)


if __name__ == "__main__":
    main()
