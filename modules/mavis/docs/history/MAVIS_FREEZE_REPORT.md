# MAVIS OCR + LLM MVP: final stabilization report (2026-10-25)

Scope: verify the v2 export cleanup and the strict OCR -> LLM hand-off; fix only genuine defects. B0 default and C3 opt-in unchanged;
RapidOCR / PP-OCRv5 pipeline, frame selectors, frozen benchmark data untouched.

## Defects found and fixed (3)
| # | Evidence | Root cause | Minimal fix | Regression test |
|---|---|---|---|---|
| D1 | `OCRFormatter.load_document("<missing path>")` returned the PARENT directory's export (probe printed the parent's page) | non-existent path fell through to `path.parent` | raise `FileNotFoundError`; also flag a page whose raw text cannot be found (`raw_text_missing`) instead of silently presenting corrected text as raw, and warn in the LLM result | `tests/test_load_document.py` |
| D2 | with page 2 failed, the finalized document contained pages `[1, 2]` where "2" was capture page 3, while `processing.pages_failed` said `[2]` (test failed before the fix: `[1, 2] != [1, 3]`) | `process_document()` numbers pages 1..k by position | `_run_postprocessing` restores the capture page number on each page | `test_page_numbers_of_the_finalized_document_are_the_capture_page_numbers`; end-to-end `test_a_failed_ocr_page_is_reported_not_hidden...` |
| D3 | real run 1: a correct answer from clean page 1 was labelled `document_unverified` only because page 3 was garbled | low-legibility downgrade applied to the whole document; evidence page numbers were taken from the model unchecked | downgrade only when the answer's evidence is on a garbled page; evidence is attributed to the page where the quote is actually found (`page_reported_by_model` kept when it differs) | 3 tests in `tests/test_llm_stage.py` |

Not defects (verified OK): window close during active capture, snapshot immutability, single-flight requests, gate ordering.

## Files changed
- `prototype/main.py`: D2.
- `postprocess/formatter.py`: D1.
- `prototype/llm_stage.py`: `raw_text_missing` warning (D1), evidence page attribution and cited-page legibility rule (D3).
- Tests: `tests/test_load_document.py` (new, 7), `tests/test_llm_ui.py` (+2: page numbering, mutating-LLM-stage immutability), `tests/test_llm_stage.py` (+3, 1 updated), `tests/smoke_pipeline.py` (options: close during capture, injected OCR failure, OCR delay; finalized-state events), `tests/test_pipeline_smoke.py` (+3: close while active x2, partial failure; idle-at-finalization assertion), `tests/e2e_real_multipage.py` (new manual real-OCR + real-Gemma script, not collected by pytest).
- No file deleted. Backups from the previous change remain in `_backups/`.

## Verification summary (carried over, not re-run for the freeze)
- Full suite `py -3.11 -m pytest prototype/test_cooldown.py prototype/test_ocr_engine.py tests`: **98 passed, 0 skipped** (executed 2026-10-25 after the last code change); `PYTHONPATH=. py -3.11 tests/test_postprocessing.py`: all passed.
- Real run (manual, `tests/e2e_real_multipage.py`): real RapidOCR + real Gemma 3 4B through the real app, with one injected OCR failure and one degraded page: ordering, failed-page reporting, evidence and labels as described in the stabilization report.
- Defaults confirmed at freeze: B0 default selector, `--selector c3` / `MAVIS_FRAME_SELECTOR=c3` opt-in.

## FROZEN CHECKPOINT (2026-10-25)
The MAVIS OCR + LLM MVP is frozen at this state. No new features or refactors. Development reopens only for a reproducible, delivery-blocking defect.
There is no git repository here, so the checkpoint is identified by file hashes (SHA-256, first 16 hex characters). Every listed file was last modified before the 98-test run.

| File | SHA-256 (first 16) |
|---|---|
| prototype/main.py | 06abc9b492ab1093 |
| prototype/frame_selector.py | 640ae1ac085427a3 |
| prototype/llm_stage.py | e8db90116a38d299 |
| prototype/ocr_engine.py | fb80e47b3554263a |
| prototype/profiling.py | d37c972f09ba2795 |
| postprocess/pipeline.py | 9aa9ac56d11e3981 |
| postprocess/formatter.py | 8ad3bd562b69cf4a |
| tests/test_llm_ui.py | cad37deb633c3c69 |
| tests/test_llm_stage.py | 6cbc61f686121277 |
| tests/test_load_document.py | ecb28b50cdd12325 |
| tests/test_export_cleanup.py | 1e3c327c90abb944 |
| tests/test_pipeline_smoke.py | 9c2ed7496e675be2 |
| tests/smoke_pipeline.py | 178f0d648e2b187f |
| tests/e2e_real_multipage.py | 7631814218ee312f |

Backups of earlier states are kept in `_backups/` (nothing deleted). Runtime installed on this machine: Ollama 0.40.11 with `gemma3:4b` (Q4_K_M); the model is currently unloaded.

## KNOWN ISSUES (accepted for the freeze; none reproduced as delivery-blocking)
1. **Shutdown can wait on a stalled camera.** `Finish & Stop` waits for the camera thread with no timeout. A frozen phone stream may keep the app in "Stopping..." until the stream read returns. Not reproduced (needs a real network fault).
2. **Dropped commits are not reflected in the processing status.** If the OCR queue (8 jobs) is full, a selector commit is dropped and counted (`selector_commit_dropped`) but `processing.status` stays `complete`. Possible only with C3 and very fast page changes; B0 retries. Not reproduced.
3. **OCR misreadings pass through.** When RapidOCR misreads a value (e.g. "(6 marks)" read as "(8 marks)"), the model repeats it. The answer is flagged `document_unverified` only when the cited page is heavily garbled (word-recognisability below 0.60, a threshold read off the earlier validation set, so in-sample); a confident misread on a legible page is not detectable.
4. **OCR recovery has no demonstrated accuracy benefit** (8/18 correct answers with and without it on the validation set) and is limited to the first 6 pages (`MAVIS_LLM_MAX_RECOVER_PAGES`). It can be turned off with `MAVIS_LLM_RECOVER=0`.
5. **Document size is bounded by the model context.** Default 16,384 tokens (about 35,000 characters, roughly 40 pages of the OCR text measured). Larger documents return `context_too_large` and nothing is sent. Behaviour above 16,384 tokens (RAM, latency) was not measured.
6. **Latency on this CPU.** First question after idle: about 30 to 57 s (model load plus recovery); later questions about 10 to 12 s; a summary about 32 s. Local Gemma runs only on CPU here.
7. **Questions only after Finish & Stop.** No live Q&A during capture and no rescan without restarting the app (the OCR engine cannot restart).
8. **Not tested:** closing the window while a model request is in flight; more than 4 real-model questions in the end-to-end run; the real model with repeated rapid clicks (that lock is model-independent and tested with a scripted model).
9. **Environment.** One transient Windows Application Control block of an OCR dependency (`pyclipper`) occurred once and did not recur. Use `py -3.11`; plain `python` on this machine is 3.10 without dependencies. The Ollama desktop app starts at login.
10. **LLM result JSON repeats some fields** (`pages[].ocr_text` plus top-level `corrected_text` and `uncertain_spans`); it is LLM output, not OCR output, and was left unchanged.

## ADDENDUM 2026-10-28 (UI work after the freeze)
Only the web UI layer (`webapp/`) was added, plus one performance change in `prototype/llm_stage.py`: `quote_match` skips windows
whose `quick_ratio()` is below the 0.85 threshold (verdicts identical, see `tests/test_quote_match_equivalence.py`). The
`llm_stage.py` hash above therefore no longer matches; every other listed hash is unchanged. Full suite on 2026-10-28:
125 passed (98 earlier + 27 new). See `docs/MAVIS_WEB_UI.md`.
