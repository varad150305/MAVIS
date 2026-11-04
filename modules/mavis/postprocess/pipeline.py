from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List
import uuid

from .confidence import ConfidenceScorer
from .formatter import OCRFormatter
from .normalizer import OCRNormalizer
from .segmenter import OCRSegmenter
from .wordfreq_correction import WordfreqCorrector


EXPORT_SCHEMA_VERSION = 2


class PostProcessingPipeline:
    """Heuristic OCR postprocessing pipeline for MAVIS V2.

    Production output keeps only fields with an actual consumer (UI,
    exports, or benchmarking): raw/normalized/corrected text, confidence,
    lightweight structure counts, and a corrections-applied count. The
    full per-token correction trace and the nested segmentation lists are
    expensive to produce as JSON and have no consumer, so they are only
    attached when a call passes debug=True.

    Correction stage: WordfreqCorrector (postprocess/wordfreq_correction.py)
    - the conservative, token-level candidate selected by the
    postprocessing benchmark (see postprocessing_benchmark/reports/
    FINAL_RECOMMENDATION.md). It replaces the prior confusion-rules-based
    correction (postprocess/confusion_rules.py, still present and still
    independently usable/tested - OCRConfusionResolver - just no longer
    wired into this pipeline by default: the benchmark measured it as the
    worst-performing correction strategy tested, see
    postprocessing_benchmark/reports/QUALITY_ANALYSIS.md). Gating is
    entirely per-token inside WordfreqCorrector, not by line confidence -
    the benchmark's follow-up ground-truth validation found line-level
    confidence gating did not reliably protect correction quality (see
    external_validation/reports/REPORT.md section 8), so confidence_score/
    confidence_label are computed here only as reported metadata, not used
    to decide whether a line gets corrected.
    """

    def __init__(self, export_dir: str | Path | None = None, enable_correction: bool = True):
        base_dir = Path(__file__).resolve().parents[1]
        self.export_dir = Path(export_dir) if export_dir is not None else base_dir / "exports"
        self.enable_correction = enable_correction
        self.normalizer = OCRNormalizer()
        self.segmenter = OCRSegmenter()
        self.confidence = ConfidenceScorer()
        self.corrector = WordfreqCorrector()
        self.formatter = OCRFormatter(self.export_dir)

    def process_page(self, raw_text: str, page_number: int = 1, debug: bool = False) -> dict:
        normalized_text = self.normalizer.normalize_text(raw_text)
        segmentation = self.segmenter.segment(normalized_text)

        corrected_lines = []
        debug_trace = [] if debug else None
        corrections_applied = 0
        total_score = 0.0
        scored_lines = 0

        for line in segmentation["raw_lines"]:
            if not line.strip():
                corrected_lines.append("")
                continue

            line_score = self.confidence.score_line(line)
            confidence_label = self.confidence.classify_confidence(line_score)
            total_score += line_score
            scored_lines += 1

            if not self.enable_correction:
                corrected_line = line
                line_trace = []
            else:
                result = self.corrector.apply_rules(line)
                corrected_line = result["corrected_text"]
                line_trace = result["trace"]

            corrections_applied += sum(1 for item in line_trace if item.get("changed"))
            corrected_lines.append(corrected_line)
            if debug:
                debug_trace.append({
                    "line": line,
                    "corrected_line": corrected_line,
                    "line_score": line_score,
                    "confidence": confidence_label,
                    "trace": line_trace,
                })

        corrected_text = "\n".join(corrected_lines).strip()
        avg_score = total_score / scored_lines if scored_lines else 0.0

        page = {
            "page_number": page_number,
            "raw_text": raw_text,
            "normalized_text": normalized_text,
            "corrected_text": corrected_text,
            "confidence_score": round(avg_score, 2),
            "confidence_label": self.confidence.classify_confidence(avg_score),
            "structure": {
                "headings": len(segmentation["headings"]),
                "paragraphs": len(segmentation["paragraphs"]),
                "noise_blocks": len(segmentation["noise_blocks"]),
            },
            "corrections_applied": corrections_applied,
        }
        if debug:
            page["segmentation"] = segmentation
            page["correction_trace"] = debug_trace
        return page

    def process_document(self, raw_pages: List[str], document_id: str | None = None, debug: bool = False) -> dict:
        document_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        processed_pages = []

        for page_number, page_text in enumerate(raw_pages, start=1):
            processed_pages.append(
                self.process_page(page_text or "", page_number=page_number, debug=debug)
            )

        summary = {
            "document_id": document_id,
            "timestamp": timestamp,
            "page_count": len(processed_pages),
            "high_confidence_pages": sum(1 for page in processed_pages if page["confidence_label"] == "HIGH"),
            "medium_confidence_pages": sum(1 for page in processed_pages if page["confidence_label"] == "MEDIUM"),
            "low_confidence_pages": sum(1 for page in processed_pages if page["confidence_label"] == "LOW"),
            "corrections_applied": sum(page["corrections_applied"] for page in processed_pages),
        }

        return {
            "document_id": document_id,
            "timestamp": timestamp,
            "pages": processed_pages,
            "summary": summary,
        }

    def export_results(self, document_result: dict, processing: dict | None = None) -> dict:
        """Write raw_output.json, corrected_output.json and metadata.json (schema_version 2).

        Each fact is stored once:
          * raw_output.json       the original OCR text per page (the only copy of raw_text)
          * corrected_output.json post-processed text and scores per page, joined to raw_output.json by
                                  page_number; normalized_text appears only where it differs from raw_text
          * metadata.json         document identity, timestamp, pipeline provenance, summary counts and the
                                  processing status (see `processing`)
        `processing` = {"status": "complete" | "complete_with_errors" | "incomplete", "pages_ok": int,
        "pages_failed": [page_number, ...]}; defaults to a complete run with every page ok. Use
        OCRFormatter.load_document(export_dir) to get the merged per-page view back.
        """
        document_id = document_result["document_id"]
        pages = document_result["pages"]
        raw_payload = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "document_id": document_id,
            "pages": [{"page_number": page["page_number"], "raw_text": page["raw_text"]} for page in pages],
        }
        corrected_payload = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "document_id": document_id,
            "pages": [self._corrected_page_payload(page) for page in pages],
        }
        summary = {k: v for k, v in document_result["summary"].items() if k not in ("document_id", "timestamp")}
        metadata_payload = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "document_id": document_id,
            "timestamp": document_result["timestamp"],
            "pipeline": {"correction": "wordfreq_token_level", "heuristic_only": True},
            "summary": summary,
            "processing": processing or {"status": "complete", "pages_ok": len(pages), "pages_failed": []},
        }

        raw_path = self.formatter.export_raw(raw_payload)
        corrected_path = self.formatter.export_corrected(corrected_payload)
        metadata_path = self.formatter.export_metadata(metadata_payload)

        return {
            "raw_output": raw_path,
            "corrected_output": corrected_path,
            "metadata": metadata_path,
        }

    @staticmethod
    def _corrected_page_payload(page: dict) -> dict:
        payload = {"page_number": page["page_number"]}
        # raw_text lives only in raw_output.json; normalized_text only where it adds information
        if page["normalized_text"] != page["raw_text"]:
            payload["normalized_text"] = page["normalized_text"]
        payload.update({
            "corrected_text": page["corrected_text"],
            "confidence_score": page["confidence_score"],
            "confidence_label": page["confidence_label"],
            "structure": page["structure"],
            "corrections_applied": page["corrections_applied"],
        })
        # Debug-only fields: only present when the page was processed with
        # debug=True, so a mixed-mode document only carries them where asked.
        if "segmentation" in page:
            payload["segmentation"] = page["segmentation"]
        if "correction_trace" in page:
            payload["correction_trace"] = page["correction_trace"]
        return payload
