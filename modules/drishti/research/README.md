# Drishti — Research Evaluation Framework

**Status: hazard-scenario results are now real (see below) — everything requiring real photos or a real API key is still infrastructure-only, so results can be produced honestly when that data is available, not to present placeholder numbers as findings.**

## Real results that exist right now

`test_scenarios/` has actually been run — `run_scenarios.mjs` drives 12 named,
described situations through the real, unmodified hazard-engine functions
(`scoreHazard`, `isInPathCorridor`, `proximityFromBbox`) and produced genuine
output in `results/hazard_scenarios_*.json`. This is real evidence for the
paper's "Hazard detection" metrics row — deterministic logic, not something
needing photos or a cloud key. Everything else below (object detection
accuracy, latency/accuracy comparisons against baselines, usefulness) still
needs real images and/or user sessions neither of which exist yet.

## Research question

> Can a local-first hybrid computer-vision + vision-language architecture improve reliability, latency, and API independence compared with an API-first VLM architecture for assistive visual applications?

## What to compare

- **Baseline A** — generic VLM prompt (image → single generic "describe this" prompt → response). This is roughly what a naive integration looks like.
- **Baseline B** — task-specific VLM prompt (image → mode-specific prompt, e.g. the pre-migration Drishti prompts → response). This is what Drishti looked like before this session's work.
- **Drishti Hybrid** — image → local CV (object detection, tracking, hazard scoring) → structured environment state → local response if sufficient, else VLM call with local context injected. This is the current, partially-implemented architecture — see `../ARCHITECTURE_AUDIT.md` and `../IMPLEMENTATION_PROGRESS.md` for exactly what's real today.

## Metrics to collect, once data exists

| Category | Metrics |
|---|---|
| Object detection | Precision, recall, F1, mAP — requires a labeled test set (see `datasets/`) |
| Hazard detection | True/false positive rate, false negative rate, hazard detection rate — requires labeled hazard scenarios (see `test_scenarios/`) |
| Latency | Camera-to-response time, local inference time, API round-trip time, total — see `../PERFORMANCE_REPORT.md` for what's already genuinely measured (local-only metrics; cloud latency not yet measured, no valid API key was available this session) |
| API dependency | % of interactions completed with zero network call, % requiring cloud reasoning — derivable today from `../API_DEPENDENCY_REDUCTION.md`'s per-feature table, not yet aggregated into a single measured percentage across real usage sessions |
| Usefulness | Task completion, user feedback, confidence, usability — requires actual user sessions (ideally with visually impaired testers, given the target audience), none have been run |

## Folder layout

- **`datasets/images/`** — empty except for its own README. Needs real, ideally licensed-for-use images: indoor/outdoor scenes with labeled objects and hazards. Nothing here should be synthetic/AI-generated if the goal is to measure real-world accuracy. `baseline/compare_modes.mjs` reads directly from this folder.
- **`test_scenarios/`** — **done**: `scenarios.md` documents 12 named, realistic situations (a person close in the walking corridor, a car far off to the side, a truck cut off by the frame edge, etc.), and `run_scenarios.mjs` runs them through the real hazard engine, producing genuine output in `results/`. Re-run it any time the hazard-engine logic changes, to see if the real numbers moved.
- **`evaluation/`** — still empty. Once `datasets/images/` has real photos, extend `baseline/compare_modes.mjs`'s pattern here for richer comparisons (e.g. object-detection precision/recall against hand-labeled ground truth, not just latency/text output).
- **`baseline/`** — **infrastructure done, unexecuted**: `compare_modes.mjs` runs Baseline A (generic prompt), Baseline B (Drishti's task-specific but cloud-only prompt), and Drishti Hybrid (the real local-first pipeline) side by side against every image in `datasets/images/`, timing each and saving real output to `results/`. Requires real photos (see above) and, for Baselines A/B specifically, a real `GROQ_KEY` or `GEMINI_KEY` env var — Drishti Hybrid runs without any key. Verified to correctly refuse to run (rather than fabricate anything) when no dataset exists yet.
- **`results/`** — has real hazard-scenario output now (`hazard_scenarios_*.json`). Everything else here should stay empty until real photos/keys produce genuine `baseline_comparison_*.json` files — never invented numbers.

## Why the rest is still empty, honestly

Producing baseline-comparison and object-detection-accuracy numbers requires: (1) a labeled image dataset representative of the target use case (indoor/outdoor navigation for a blind user in India, per the project's own scope), and (2) either a working Groq/Gemini API key with real quota, or campus/lab access to run comparable local baselines. Neither exists yet. Building fake results to fill these folders would defeat the entire purpose of a research evaluation — the infrastructure above is ready to produce real numbers the moment either becomes available, rather than needing to be built from scratch later.
