# Hazard-engine test scenarios

Twelve named, described situations covering the real class/proximity/path-corridor
space the deterministic hazard engine (`scoreHazard()` in `drishti-web/index.html`)
actually reasons over — HAZARD_CLASSES, `objectRiskWeight()`, `proximityFromBbox()`,
and `isInPathCorridor()`. These are specifications of realistic inputs, not
measured results: `run_scenarios.mjs` feeds each one through the real,
unmodified functions and records what they actually output — nothing here is a
guessed or pre-computed answer.

This complements (doesn't replace) `drishti-web/tests/test_hazard_engine.mjs`,
which unit-tests the same functions with synthetic edge cases. This file's
scenarios are written to read as real-world situations a blind pedestrian in
India would plausibly encounter, for a research write-up rather than a code
review.

| # | Scenario | Class | Position in frame | Notes |
|---|---|---|---|---|
| 1 | Person standing directly ahead, close range, in the walking corridor | person | center, large bbox | Expected: high risk, in-path |
| 2 | Person far across the street, small in frame | person | center, tiny bbox | Expected: low risk despite being a hazard class |
| 3 | Car approaching head-on in the corridor, close | car | center, large bbox | Expected: critical — high-risk class + close + in-path |
| 4 | Car parked at the far side of the frame, not in the walking corridor | car | frame edge, small bbox | Expected: low risk — out of path, far |
| 5 | Bicycle passing on the far left, not in the corridor | bicycle | far left edge, medium bbox | Expected: low/medium — hazard class but out of path |
| 6 | Dog close by, directly ahead | dog | center, large bbox | Expected: medium/high — moderate risk weight, close, in-path |
| 7 | Chair directly ahead, close (indoor obstacle) | chair | center, large bbox | Expected: low/medium — low risk weight even when close and in-path |
| 8 | Bench off to the side, far away | bench | frame edge, small bbox | Expected: negligible |
| 9 | Motorcycle close, off to the side but not in the direct corridor | motorcycle | left third, large bbox | Expected: medium — high risk weight offsets being out of path |
| 10 | Bus far away, centered | bus | center, small bbox | Expected: low-medium — high risk weight but far reduces it |
| 11 | Stop sign nearby, off to the side | stop sign | right edge, medium bbox | Expected: negligible — lowest risk weight in the class table |
| 12 | Truck close, in the corridor, at the frame edge (partially out of view) | truck | edge-touching bbox, large area | Expected: critical risk, but LOW distance-confidence (edge-touching bbox is flagged unreliable, not silently trusted) |

Run `node run_scenarios.mjs` (with `drishti-web` served — see below) to produce
real output for these, written to `../results/hazard_scenarios_<timestamp>.json`.
