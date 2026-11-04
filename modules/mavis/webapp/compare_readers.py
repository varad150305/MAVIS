"""Compare the local reader (PP-OCR) with Gemini on page photos that MAVIS already saved. Run by the user, with their own key.

    py -3.11 -m webapp.compare_readers --yes scan-20261028-191653 scan-20261028-193024  [--pages 3 4 5]

SENDS THE PAGE PHOTOS OF THE LISTED SCANS TO GOOGLE (Gemini). Nothing is sent unless you pass --yes. The key is read from
GEMINI_API_KEY (set it with:  setx GEMINI_API_KEY "..."  then open a new terminal). Output: a table (characters read and the
share of recognisable English words, a rough proxy for 'garbled or not') and a report with both transcriptions side by side:
exports/web/_compare/<timestamp>.md. Note: the stored photos are 1100 px wide; live scans send the full camera frame, so
Gemini does at least as well live as here, and the local reader does about the same.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "prototype", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from llm_stage import legibility  # noqa: E402
from ocr_engine import PrototypeOCREngine  # noqa: E402

from webapp.gemini_reader import ReaderSettings, ReaderUnavailable, make_reader  # noqa: E402
from webapp.watermark import clean_text_lines, split_detections  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scans", nargs="+", help="scan folder names under exports/web/")
    ap.add_argument("--pages", nargs="*", type=int, help="only these page numbers")
    ap.add_argument("--yes", action="store_true", help="confirm that the photos may be sent to Google")
    args = ap.parse_args()

    settings = ReaderSettings()                              # provider and key come from the environment
    if not settings.has_key:
        sys.exit(f"{settings.key_env} is not set. {settings.setup_hint}")
    if not args.yes:
        sys.exit("This sends the page photos of the listed scans to the cloud OCR provider (MAVIS_OCR_PROVIDER). Re-run with --yes if that is OK.")

    reader, local = make_reader(settings), PrototypeOCREngine()
    lines, report = [], [f"# Reader comparison {time.strftime('%Y-%m-%d %H:%M')}", f"Model: {settings.model}\n"]
    print(f"{'scan':26} {'page':>4} | {'local chars':>11} {'local words ok':>14} | {'gemini chars':>12} {'gemini words ok':>15}")
    for scan in args.scans:
        for photo in sorted((ROOT / "exports" / "web" / scan / "pages").glob("page_*.jpg"),
                            key=lambda p: int(p.stem.split("_")[1])):
            n = int(photo.stem.split("_")[1])
            if args.pages and n not in args.pages:
                continue
            frame = cv2.imread(str(photo))
            prepared, _ = PrototypeOCREngine._prepare_crop(frame)
            result, _, _ = local.ocr(prepared)
            kept, _ = split_detections(result["detections"])
            local_text = local._reconstruct_layout(kept) if kept else ""
            try:
                gem_text, _ = clean_text_lines(reader.read(frame))
            except ReaderUnavailable as exc:
                gem_text = f"[not read: {exc.reason}]"
            lg = lambda t: "-" if legibility(t) is None else f"{legibility(t):.0%}"      # noqa: E731
            print(f"{scan:26} {n:>4} | {len(local_text):>11} {lg(local_text):>14} | {len(gem_text):>12} {lg(gem_text):>15}")
            report += [f"## {scan} page {n}", "### Local (PP-OCR)", "```", local_text, "```", "### Gemini", "```", gem_text, "```", ""]
            time.sleep(60.0 / settings.rpm)
    out = ROOT / "exports" / "web" / "_compare"
    out.mkdir(parents=True, exist_ok=True)
    path = out / (time.strftime("%Y%m%d-%H%M%S") + ".md")
    path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nSide-by-side transcriptions: {path}")


if __name__ == "__main__":
    main()
