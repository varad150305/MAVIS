from __future__ import annotations

import json
from pathlib import Path


class OCRFormatter:
    """Write postprocessed OCR outputs to exports/ in a traceable format."""

    def __init__(self, export_dir: str | Path):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_raw(self, payload: dict) -> str:
        path = self.export_dir / "raw_output.json"
        self._write(path, payload)
        return str(path)

    def export_corrected(self, payload: dict) -> str:
        path = self.export_dir / "corrected_output.json"
        self._write(path, payload)
        return str(path)

    def export_metadata(self, payload: dict) -> str:
        path = self.export_dir / "metadata.json"
        self._write(path, payload)
        return str(path)

    @staticmethod
    def load_document(path: str | Path) -> dict:
        """Rebuild the per-page view of an export (the shape PostProcessingPipeline.process_document returns).

        `path` is an export directory or one of its files. Merges raw_output.json (raw_text) with
        corrected_output.json by page_number; a page without normalized_text gets raw_text (that is what
        the export means by omitting it). Also reads version-1 exports, which carry raw_text and
        normalized_text inside corrected_output.json. A page whose raw text cannot be found is flagged
        `raw_text_missing` (its raw_text is then the corrected text). Raises FileNotFoundError for a path that does
        not exist. Returns {"document_id", "pages", "processing"?}.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no OCR export at {path}")
        directory = path if path.is_dir() else path.parent
        def read(name):
            file = directory / name
            return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None
        raw, corrected, meta = read("raw_output.json"), read("corrected_output.json"), read("metadata.json")
        if path.is_file() and path.name not in ("raw_output.json", "corrected_output.json", "metadata.json"):
            corrected, raw, meta = json.loads(path.read_text(encoding="utf-8")), None, None
        raw_by_page = {p["page_number"]: p["raw_text"] for p in (raw or {}).get("pages", [])}
        pages = []
        for p in ((corrected or raw) or {}).get("pages", []):
            page = dict(p)
            if "raw_text" in p:
                page["raw_text"] = p["raw_text"]
            elif p["page_number"] in raw_by_page:
                page["raw_text"] = raw_by_page[p["page_number"]]
            else:                      # raw_output.json missing or lacking this page: do not pass corrected text off as raw
                page["raw_text"] = p.get("corrected_text", "")
                page["raw_text_missing"] = True
            page.setdefault("normalized_text", page["raw_text"])
            pages.append(page)
        document = {"document_id": ((corrected or raw) or {}).get("document_id"), "pages": pages}
        if meta and "processing" in meta:
            document["processing"] = meta["processing"]
        return document

    def _write(self, path: Path, payload: dict) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
