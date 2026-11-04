"""Finished documents on disk: the frozen snapshot, per-page images, scan metadata and the question history.

Layout of one document (exports/web/<id>/):
    export/            raw_output.json, corrected_output.json, metadata.json   (the pipeline's own export, schema v2)
    snapshot.json      the exact frozen document text the model is asked about
    meta.json          title, date, source, per-page OCR confidence/timing
    pages/page_N.jpg   the captured page images
    chat.json          the question/answer history shown in the workspace
"""
import hashlib
import json
import shutil
import threading
import time
import uuid
from pathlib import Path

CHAT_KEYS = ("mode", "request", "answer", "answerable_from_document", "source", "evidence", "general_knowledge",
             "warnings", "unsupported_numbers", "error", "timing", "model", "ocr_legibility", "document_sha256")


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


class Document:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.id = self.dir.name
        self._lock = threading.Lock()
        self.meta = _read_json(self.dir / "meta.json", {})
        self.snapshot_json = (self.dir / "snapshot.json").read_text(encoding="utf-8")
        self.snapshot = json.loads(self.snapshot_json)
        self.turns = _read_json(self.dir / "chat.json", {"turns": []}).get("turns", [])

    @property
    def title(self):
        return self.meta.get("title") or self.id

    def summary(self):
        last = self.turns[-1]["question"] if self.turns else None
        return {"id": self.id, "title": self.title, "created_at": self.meta.get("created_at"),
                "page_count": self.meta.get("page_count", len(self.snapshot.get("pages", []))),
                "pages_failed": self.meta.get("pages_failed", []), "blank_pages": self.meta.get("blank_pages", []),
                "readers": self.meta.get("readers", ["local"]),
                "source": (self.meta.get("source") or {}).get("kind"),
                "turn_count": len(self.turns), "last_question": last,
                "mean_confidence": self.meta.get("mean_confidence")}

    def detail(self):
        """Every captured page, not only the ones that were read: a page that failed or came out blank is listed with its
        photo and the reason, so the side panel always matches what was scanned."""
        per_page = {p["page"]: p for p in self.meta.get("pages", [])}
        read = {p["page_number"]: p for p in self.snapshot.get("pages", [])}
        pages = []
        for n in sorted(set(per_page) | set(read)):
            page, info = read.get(n), per_page.get(n, {})
            if page is not None:
                status = "ok"
            elif info.get("error"):
                status = "failed"
            else:
                status = "blank"
            text = (page or {}).get("raw_text", "")
            pages.append({"n": n, "status": status, "error": info.get("error"),
                          "raw_text": text, "corrected_text": (page or {}).get("corrected_text", ""),
                          "label": (page or {}).get("confidence_label"), "score": (page or {}).get("confidence_score"),
                          "ocr_confidence": info.get("mean_confidence"), "ocr_s": info.get("ocr_s"),
                          "has_image": (self.dir / "pages" / f"page_{n}.jpg").exists(),
                          "low_content": status == "ok" and len(text.strip()) < 40})
        return {**self.summary(), "pages": pages, "turns": self.turns, "processing": self.snapshot.get("processing"),
                "snapshot_sha256": self.meta.get("snapshot_sha256"), "scan_seconds": self.meta.get("scan_seconds"),
                "captured_count": len(pages)}

    def page_state(self, n):
        info = next((p for p in self.meta.get("pages", []) if p["page"] == n), None)
        if info is None:
            return None
        if any(p["page_number"] == n for p in self.snapshot.get("pages", [])):
            return "ok"
        return "failed" if info.get("error") else "blank"

    def add_page_text(self, n, text, pipeline_factory, reader):
        """A page that failed or was blank at capture has now been read: put it into the document in its place.

        Rebuilds the post-processed pages, the three export files and the frozen snapshot (new hash), keeping the document
        id. Returns "ok", or "blank" if there was still no text (the page stays out of the document)."""
        with self._lock:
            info = next((p for p in self.meta["pages"] if p["page"] == n), None)
            if info is None:
                raise ValueError("unknown page")
            info.update(error=None, state="done", chars=len(text), reader=reader)
            if not text.strip():
                (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=1, ensure_ascii=False), encoding="utf-8")
                return "blank"
            raws = {p["page_number"]: p["raw_text"] for p in self.snapshot["pages"]}
            raws[n] = text
            numbers = sorted(raws)
            pipeline = pipeline_factory(self.dir / "export")
            processed = pipeline.process_document([raws[k] for k in numbers], document_id=self.meta.get("document_id"))
            for k, page in zip(numbers, processed["pages"]):
                page["page_number"] = k                    # keep capture numbering
            failed = sorted(p["page"] for p in self.meta["pages"] if p.get("error"))
            processing = {"status": "complete" if not failed else "complete_with_errors", "pages_ok": len(numbers),
                          "pages_failed": failed}
            pipeline.export_results(processed, processing=processing)
            document = {**processed, "processing": processing}
            snapshot_json = json.dumps(document, sort_keys=True, ensure_ascii=False)
            (self.dir / "snapshot.json").write_text(snapshot_json, encoding="utf-8")
            self.meta.update(page_count=len(numbers), pages_failed=failed,
                             blank_pages=[b for b in self.meta.get("blank_pages", []) if b != n],
                             snapshot_sha256=hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest(),
                             readers=sorted(set(self.meta.get("readers", [])) | {reader}))
            (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=1, ensure_ascii=False), encoding="utf-8")
            self.snapshot_json, self.snapshot = snapshot_json, json.loads(snapshot_json)
            return "ok"

    def page_image_path(self, n):
        path = self.dir / "pages" / f"page_{int(n)}.jpg"
        return path if path.exists() else None

    def rename(self, title):
        with self._lock:
            self.meta["title"] = title.strip()[:120] or self.title
            (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=1, ensure_ascii=False), encoding="utf-8")

    def add_turn(self, mode, question, result):
        turn = {"id": uuid.uuid4().hex[:10], "ts": time.time(), "mode": mode, "question": question,
                "result": {k: result.get(k) for k in CHAT_KEYS}}
        with self._lock:
            self.turns.append(turn)
            (self.dir / "chat.json").write_text(json.dumps({"turns": self.turns}, indent=1, ensure_ascii=False),
                                                encoding="utf-8")
        return turn

    def text_export(self):
        parts = []
        for page in self.snapshot.get("pages", []):
            parts.append(f"=== Page {page['page_number']} ===\n{page.get('corrected_text', '')}")
        return "\n\n".join(parts) + "\n"

    def export_path(self, kind):
        names = {"raw": "raw_output.json", "corrected": "corrected_output.json", "metadata": "metadata.json"}
        return self.dir / "export" / names[kind] if kind in names else None


class Library:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._docs = {}
        self._lock = threading.Lock()

    def get(self, doc_id):
        if not doc_id or "/" in doc_id or "\\" in doc_id or doc_id.startswith("."):
            return None
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None:
                directory = self.root / doc_id
                if not (directory / "snapshot.json").exists():
                    return None
                doc = self._docs[doc_id] = Document(directory)
            return doc

    def forget(self, doc_id):
        with self._lock:
            self._docs.pop(doc_id, None)

    def delete(self, doc_id):
        """Permanently remove one document folder. Only a folder directly inside the library that holds a snapshot."""
        if not doc_id or "/" in doc_id or "\\" in doc_id or doc_id.startswith("."):
            raise ValueError("invalid document id")
        directory = (self.root / doc_id).resolve()
        if directory.parent != self.root.resolve() or not (directory / "snapshot.json").is_file():
            raise ValueError("not a document folder")
        with self._lock:
            self._docs.pop(doc_id, None)
            shutil.rmtree(directory)

    def list(self):
        docs = []
        for directory in self.root.iterdir():
            if directory.is_dir() and (directory / "snapshot.json").exists():
                doc = self.get(directory.name)
                if doc is not None:
                    docs.append(doc.summary())
        docs.sort(key=lambda d: d.get("created_at") or "", reverse=True)
        return docs
