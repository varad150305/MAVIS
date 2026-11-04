# History

`MAVIS_FREEZE_REPORT.md` is the stabilization and freeze report of the earlier, desktop (Tk) version of the OCR + LLM MVP. It is
kept unchanged as a record. When reading it:

- The desktop app, its 98 tests and the research folders it mentions (benchmarks, validation sets, backups, papers) are **not part of
  this repository**; they live in the project's private research workspace.
- Its statement "there is no git repository here" is out of date, and its hash table applies to the frozen modules that are
  here: `prototype/frame_selector.py`, `ocr_engine.py`, `profiling.py`, `postprocess/pipeline.py` and `formatter.py` still match.
  `prototype/llm_stage.py` differs on purpose (a speed-only change to `quote_match`, described in the report's addendum).
- The product's primary entry point is now the web app (`python -m webapp`), with a cloud OCR provider and the local engine
  only as an offline fallback. See `../ARCHITECTURE.md`.
