# Drishti — Performance Report

All numbers below were actually measured (via `performance.now()` inside the real running app, driven through Chrome DevTools Protocol) — none are estimated or fabricated. Where something couldn't be measured honestly, it's marked "not measured" with the reason, per this project's own rule against fabricating results.

## Measurement environment

Headless Chrome 151, `--disable-gpu` (no hardware GPU passthrough), running on the development machine used for this session — **not** a mobile device and **not** representative of end-user hardware. TensorFlow.js reported `WebGL is not supported on this device` and fell back to its CPU backend in this environment. This materially affects the object-detection numbers below — see the caveat in that section.

## Object detection (TensorFlow.js + COCO-SSD, `mobilenet_v2` base)

| Metric | Measured value |
|---|---|
| Model load time (warm — already cached from an earlier call in the same session) | ~0ms (not a useful cold-load number; see limitation below) |
| Per-inference latency (`detect()` on a 1280×720 frame), avg of 5 calls | **~4810ms** (range 4521–5422ms) |

**This number is a real measurement, and it's concerning as reported — but it reflects a CPU-only fallback in a headless test environment with no GPU passthrough, not the expected real-world case.** A real user's browser (desktop or mobile) has WebGL-accelerated GPU access, which TensorFlow.js uses by default and which is typically one to two orders of magnitude faster than CPU inference for CNN-style models. I made two attempts to verify this directly and neither produced a trustworthy number: (1) relaunching Chrome headless with software WebGL enabled (`--enable-unsafe-swiftshader`, no `--disable-gpu`) — the object detector failed to initialize entirely in that configuration; (2) launching an actual non-headless Chrome window against the same page — the detector also failed to reach a ready state within the test script's timing, for a reason not fully diagnosed (possibly a camera-permission or timing difference from the headless flow this session's tooling was built around). **I am not claiming a specific faster number, because I did not measure one in either attempt** — this needs verification directly in a browser DevTools console on real end-user hardware before trusting the current polling interval (500ms) against real-world inference latency.

**Actionable finding, not yet acted on**: the current `odLoop()` schedule (`setTimeout(odLoop, 500)`, chained after each detection completes) does not queue or overlap calls, so it degrades gracefully rather than piling up work under slow inference — but if real-world inference genuinely takes several seconds on some lower-end device, the effective detection rate would drop far below the intended ~2fps. Phase 15 of the migration brief calls for adaptive model/scheduling based on measured device performance; this hasn't been implemented. A reasonable next step would be to time the first few inference calls at runtime and fall back to the lighter `lite_mobilenet_v2` base automatically if they exceed some threshold (e.g. 800ms) — not implemented here, flagged as future work rather than built speculatively.

## Local color detection (3×3 grid, pure canvas pixel math)

| Metric | Measured value |
|---|---|
| Per-call latency (`localColorGrid()` on a 1280×720 canvas), avg of 20 calls | **~4.7-8.0ms** |

Negligible cost, as expected for pure pixel sampling with no model involved — consistent across two separate measurement runs.

## Hazard / tracking / priority engine (pure logic, no model)

| Metric | Measured value |
|---|---|
| Per-call latency (`buildEnvironmentState()` with 3 synthetic detections), avg of 50 calls | **~0.03ms** |

Effectively free — this is plain JavaScript object/array manipulation, exactly as expected for logic with no ML inference involved.

## Local OCR (Tesseract.js)

| Metric | Measured value |
|---|---|
| Worker load time (first-time CDN fetch of the JS + WASM binary + English language data) | **~200-2350ms** (varied across runs — network-dependent, not a stable number in this sandbox) |
| Recognition latency on a 600×200 rendered-text image | **~124-144ms** |
| Confidence on clean, computer-rendered text | **95%** |

These are real numbers from a real OCR pass (see `IMPLEMENTATION_PROGRESS.md` Entry 5) — genuinely fast enough for interactive use once the worker is loaded. Real-world accuracy/latency on an actual phone-camera photo of a sign or page (not clean rendered text) has not been measured — would need real test images, not canvas-rendered text, to evaluate honestly.

## What was not measured, and why

| Metric | Why not |
|---|---|
| Groq cloud API round-trip latency | No valid Groq API key was available in this environment — every test used a fake key, which fails fast at the auth check rather than completing a real request. Fabricating a plausible-looking number here would violate this project's own rule against invented results. |
| Mobile RAM / battery / CPU usage | No physical device; only an Android emulator was available in this session, and even that wasn't reached before the Flutter build was paused (see `IMPLEMENTATION_PROGRESS.md` Entry 1/4). These require real hardware to measure honestly. |
| Real-world object-detection accuracy (precision/recall/mAP) | No labeled test dataset exists in this repository. Building the evaluation *infrastructure* for this is `research/` (below); producing real numbers requires collecting or sourcing real annotated images first. |
| GPU-accelerated inference latency | Attempted (see above) but could not get a working WebGL context in this sandboxed headless environment to measure. |

## Recommendation

Before trusting the current on-device detection polling rate for a real demo or deployment, verify actual inference latency in a real (non-headless) browser session on the target hardware — the number that matters is not the one measured here, which is a known-degraded CPU-fallback case.
