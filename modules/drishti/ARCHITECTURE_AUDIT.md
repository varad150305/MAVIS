# Drishti — Architecture Audit

Date: 2026-08-21
Scope: `drishti-web/` (deployed, working), `drishti-app/` (Flutter, previously non-buildable), `website/` (standalone marketing page, out of scope), `drishti-web/api/` (Vercel serverless proxies).

This document reflects what was actually found by inspecting the repository and, where possible, actually running it — not assumptions about what a project like this typically contains. Every claim below was verified against source or a live run; anything not verified is marked as such.

---

## 0. Environment facts that constrain what follows

These aren't architecture, but they determine what's honestly achievable and were discovered during this audit:

- **Flutter SDK 3.29.3 is installed and working** on this machine.
- **`drishti-app/android/` and `drishti-app/ios/` were incomplete stubs** — only `AndroidManifest.xml` and `Info.plist` existed; no Gradle files, no Xcode project, no wrapper. **The Flutter app could not be built at all before this audit.** Fixed by running `flutter create . --platforms=android,ios` (non-destructive — regenerates only platform scaffolding, `lib/` untouched, the hand-written `AndroidManifest.xml` permissions were preserved by Flutter's merge).
- An Android emulator (`Medium_Phone_API_36`, API 36) is available and was booted for this audit — a real (emulated) Android device is now reachable via `adb`/`flutter run`, so Flutter-side work in this project can actually be built and verified, not just written speculatively.
- No physical device, no iOS toolchain (Windows host), no Android build previously confirmed working — battery/RAM/real-camera-hardware numbers are not obtainable here and won't be fabricated.

## 1. Current architecture (as found)

### 1a. `drishti-web/` — the mature, actually-deployed surface
Single static page (`index.html`, ~1900 lines of inline JS, no build step) plus two Vercel serverless functions (`api/groq.js`, `api/sarvam.js`).

```
Camera (getUserMedia)
   → capture() [canvas snapshot, JPEG]
   → callGroq(img, mode)  — picks one of 13 hardcoded prompts by mode
   → Groq API (Llama-4-Scout / Qwen vision model) — ALL reasoning happens here
   → free-text response
   → regex post-processing (parsePrices, showSafetyBadge, showCrossingBadge, updateIntelBar)
   → speak() → Sarvam TTS API, or Web Speech API fallback
```

13 modes, all API-routed the same way: Describe, Read Text, Currency, Color, People, Navigate, Objects, Prices, Food, Safety, Crossing, Translate, Gesture.

**One local CV component already exists** (added this session, not part of the original build): a TensorFlow.js + COCO-SSD on-device object detector that triages Guardian Mode — it runs continuously client-side with zero network calls, draws live bounding boxes, and only escalates to Groq when a hazard-class object is close in frame. This is a real, working, already-verified example of exactly the "local triage → optional cloud reasoning" pattern this migration is asking for — it should be generalized, not thrown away and rebuilt.

Also already local/non-API:
- **Voice command routing** — `CMD_MAP` / `REGIONAL_CMDS` are pure regex pattern matching against STT output. Zero LLM calls to decide what a spoken command means. This already satisfies Phase 10's requirement.
- **STT** — browser's native `SpeechRecognition` (no app-level API key or call).
- **SOS** — pure client logic (geolocation + `sms:`/`tel:` deep links), no AI involved at all.
- **TTS fallback** — Web Speech API when no Sarvam key/proxy is configured.

### 1b. `drishti-app/` — the Flutter app, much less developed
`GroqService` + `SarvamService` + `SosService`, `HomeScreen` (665 lines), `SettingsScreen`. Same conceptual pipeline as the web app but a **strict subset**: 8 modes (no Navigate-crossing, Translate, or Gesture modes; no Guardian Mode; no on-device anything). Camera usage is snap-based (`_cam.takePicture()`), not a continuous frame stream — there is currently no code path that could feed a per-frame detector even if one were added, so that's new plumbing, not a hookup.

`GroqService`'s hardcoded model list includes `llama-3.2-11b-vision-preview` and `llama-3.2-90b-vision-preview` — **the web app's own code comments (index.html:283-285) already documented that Groq decommissioned these models**. The Flutter app was never updated to match; this is a live bug (any request using the Flutter app's default/fallback path could hit a dead model), separate from and predating this migration.

`SarvamService` already does cloud-TTS-with-local-fallback (`flutter_tts`) correctly — same hybrid pattern as the web app, nothing to change here.

### 1c. Backend
Two stateless Vercel functions, each: health-check (`GET`, returns whether a server-side key is configured) + proxy-with-rate-limit (`POST`, forwards to Groq/Sarvam using an env-var key so end users never need their own key). No database, no model hosting, no compute beyond forwarding HTTP requests. This is infrastructure for *hiding* an API key, not for running any AI — it has no bearing on the local-first migration other than that it must keep working as the "cloud" side of the hybrid orchestrator.

## 2. Current API dependencies

| Dependency | Used for | Where |
|---|---|---|
| Groq API (Llama-4-Scout / Qwen vision-language model) | All 13 web modes + all 8 Flutter modes + chat Q&A + Guardian's cloud explanation step | `callGroq()` (web), `GroqService` (Flutter) |
| Sarvam AI TTS | Indian-language voice synthesis (optional) | `speak()` (web), `SarvamService` (Flutter) |
| OpenStreetMap Nominatim + OSRM | Turn-by-turn walking directions (free, no key) | `streetNavigate()` (web only) |
| TensorFlow.js + COCO-SSD model weights (CDN) | On-device object detection (already local at *inference* time; the model file itself is fetched once from a CDN) | `loadObjectDetector()` (web only, added this session) |

## 3. Current strengths

- The web app already has a working hybrid pattern in one place (Guardian Mode's on-device triage) — proof this architecture is achievable in this exact codebase, not theoretical.
- Voice command parsing, STT, SOS, and TTS-fallback are already local-first. Less needs to change here than the migration brief assumes.
- The Vercel proxy design (zero-setup for blind/elderly end users, graceful fallback to manual key entry) is a genuinely good piece of engineering and should be preserved as the "cloud" side of the new orchestrator, not replaced.
- The web app has a proven, repeatable **empirical verification method** from this session (headless-Chrome CDP driving every function, catching real runtime errors) — this is a real asset for this migration: every phase below should be verified the same way, not just read-reviewed.
- Prompts are already mode-specific and reasonably engineered (clock-position/distance language, forced SAFE/CAUTION/DANGER and GO/WAIT vocabularies, anti-boilerplate style rules added this session).

## 4. Current weaknesses

- **Every mode, including safety-critical ones (Safety, Crossing, Navigate), routes through a general-purpose cloud VLM with no local corroboration.** If the network is down or the API key is invalid, hazard detection doesn't degrade gracefully — it simply doesn't run at all. This is the core problem this migration exists to fix.
- **The "Crossing" mode currently asks the cloud VLM to output GO or WAIT for a road-crossing decision.** This directly violates the safety principle this migration brief itself states ("do not ask a cloud VLM to make the final decision for immediate safety-critical actions," "never claim a road is definitely safe to cross"). This predates the migration and needs explicit redesign, not just a local model bolted in front of it — see §7.
- No distance measurement anywhere is uncertainty-aware — the bbox-area proxy added this session for Guardian Mode is a *rough proximity heuristic*, not depth estimation, and is not currently labeled as such in the UI. That's a real gap this audit is flagging honestly rather than overselling. *(Update — see `IMPLEMENTATION_PROGRESS.md` Entry 11: a genuine monocular depth model was subsequently added as an on-demand "how far" voice command; the continuous per-frame signal described here is still the bbox-area proxy, since real depth estimation measured ~12.7s/inference — too slow to run continuously.)*
- No object tracking — every snap/frame is independent; nothing suppresses repeated "chair, chair, chair" announcements outside of Guardian's own gating.
- The Flutter app duplicates the web app's prompts/logic by hand with no shared source of truth, has already drifted (dead model IDs), and until this audit could not even be built.
- No tests, no evaluation framework, no performance measurements exist anywhere in the repository.

## 5. Feature classification

| Feature | Class | Reasoning |
|---|---|---|
| Object presence/count (chair, person, bottle, etc.) | **A — local** | COCO-SSD via TF.js already does this on web, verified working this session. Flutter equivalent: `tflite_flutter` + a mobile detection model (see §8). |
| Voice command routing | **A — local** | Already fully local (regex). No change needed beyond porting the same idea to Flutter (currently absent there). |
| SOS | **A — local** | Already fully local. No AI involved; not in scope for this migration. |
| Color of a specific region | **A — local** | Pure pixel math (RGB→named color), no model needed. |
| OCR (read text) | **A — local** | Tesseract.js (web, WASM) / Google ML Kit Text Recognition (Flutter, on-device, free, no key) both exist and are production-grade. |
| Guardian hazard triage (is something close?) | **B — hybrid** | Already implemented this session: local detection gates *when* to call the cloud model; cloud model still writes the explanation. Correct split. |
| "Objects" mode (name + estimated price) | **B — hybrid** | Local model gives class/position for free; price estimation needs real-world product/market knowledge no local model here has — stays cloud. |
| Currency recognition | **B — hybrid, honestly scoped** | No trained Indian-currency classifier exists in this repo and none can be fabricated without a labeled dataset. Realistic local-first approach: local OCR reads the printed numeral (₹500/₹100/etc., which is large-print and high-contrast on Indian notes) plus a size/color heuristic, with the cloud model as a fallback/confidence check — not a genuine trained classifier claiming to be "AI-graded." See §9. |
| Free-space / "clear path" heuristic | **B — hybrid** | Local: flag where detected objects occupy the lower-frame walking corridor. Not real depth, must never claim "safe" — only "path unclear" vs "obstacle ahead." |
| People (age, expression, clothing detail) | **C — cloud** | Fine-grained semantic description; no small local model here does this credibly. Local model can still say "a person is present, 2 o'clock" for free — the *rich* description stays cloud. |
| Food identification (dish name, nutrition) | **C — cloud** | Needs a large food-specific classifier this project doesn't have. COCO's few food classes (banana, pizza, sandwich, cake...) are too coarse to replace this. |
| Scene description (Describe mode's narrative) | **C — cloud, but gated** | The narrative itself needs a language model. The *gate* (should we even call it — motion/change detected?) is already local (Smart Live's frame-diff). Keep as-is; this is Phase 11's model, already half-built. |
| Translate mode | **C — cloud** | Small on-device translation models for 9 Indian languages at usable quality aren't available in this stack; not worth faking. |
| Gesture mode | **C — cloud** | Needs semantic body-language understanding; no compact local model exists for this. |
| Street navigation (OSRM/Nominatim) | **C — cloud, but acceptable** | Turn-by-turn routing inherently requires a maps service. It's free and has no vendor lock-in; not a target for localization. |
| **Crossing mode's GO/WAIT decision** | **D — redesign required** | This is the one place the current app violates its own stated safety principle. See §7 for the specific redesign — this is not simply "run local detection instead of Groq," because *neither* a general VLM *nor* a lightweight local model can reliably read a pedestrian signal and judge vehicle speed from a single monocular frame. The fix is in the response framing and consent model, not just the inference source. |

## 6. Recommended architecture

```
Camera
  → Frame manager (throttled — not every mode needs every frame)
  → Local pipeline (runs first, always, free, offline-capable):
        - Object detection (generalize the existing COCO-SSD integration)
        - Proximity/"distance bucket" heuristic (formalize Guardian's bbox-area proxy; label it honestly as a proxy, not depth)
        - Local OCR (new)
        - Local color sampling (new)
        - Hazard scoring (new — deterministic, testable, no LLM)
  → Environment State (new — structured object, since nothing like this exists yet)
  → Priority Engine (new — decide what's worth saying, and in what order)
  → Decision: can this be answered locally?
        YES → speak the local structured result (fast, offline, free)
        NO  → send the already-reduced context to Groq for narrative/complex reasoning (existing pipeline, unchanged, now used selectively instead of unconditionally)
  → TTS (existing hybrid Sarvam/local-fallback — unchanged)
```

The Vercel proxy, Sarvam integration, voice-command parsing, and SOS system are **not being rebuilt** — they already fit this model or are out of scope.

## 7. Special case: Crossing mode

Per the migration's own explicit rule, this needs a framing change, not just a faster/local version of the same GO/WAIT prompt:
- Local traffic-light/pedestrian-signal classification and vehicle-speed estimation from a single phone camera are not reliable enough to be the sole basis for a cross/don't-cross decision — building a fake "local hazard model" for this and presenting it as authoritative would be worse than the current cloud version, not better.
- Recommended change: reframe the mode's output from a decision ("GO"/"WAIT") to an **observation** ("a vehicle is approaching from your right, no crossing signal visible") and require the user's own judgment for the final call — this is both more honest and safer, and is a defensible design decision to present to a professor precisely because it explicitly rejects over-claiming capability the system doesn't have.

## 8. Local object detection — platform-specific reality

- **Web**: already solved this session (TF.js + COCO-SSD, `mobilenet_v2` base, 0.66 confidence floor). Generalizing this beyond Guardian Mode (feeding a real Environment State / Priority Engine) is new work but low-risk — the model is already loading and running correctly, verified via headless-browser testing.
- **Flutter**: nothing exists yet. `tflite_flutter` + a mobile-exported detection model (e.g. an official TFLite export of a COCO-trained SSD/YOLO-family model) is the realistic path. This requires: adding the `camera` package's `startImageStream` (not currently used — today's Flutter app only takes discrete snapshots), a `tflite_flutter` dependency, and an actual `.tflite` model file checked into `assets/`. This is genuinely buildable now that the Android platform scaffolding has been fixed and a real emulator is available to test on — it was not buildable before this audit.

## 9. What will NOT be faked

Per the migration brief's own explicit rules, and consistent with how this project has been debugged so far this session (verify by running, not by assuming):
- No exact-metre distance claims without a real calibration method — proximity will be bucketed (`VERY_CLOSE`/`CLOSE`/`MEDIUM`/`FAR`) with an explicit confidence tag, and silence is preferred over a guess when confidence is low.
- No trained currency classifier will be invented without training data — the local approach is OCR-plus-heuristic, documented as such, not oversold as a "custom-trained AI model."
- No mAP/precision/recall numbers will be reported without an actual labeled test set — the evaluation framework (Phase 16) will be built as real, runnable infrastructure; the *numbers* will only appear once real test data exists to produce them.
- No mobile RAM/battery/FPS numbers will be reported without measuring them on the real emulator (now available) or a physical device — web-side numbers (model size, inference latency) can be measured today via the browser's own performance APIs and will be real.

## 10. Migration plan (maps to the brief's Sprint structure)

1. **Sprint 1 (this pass)**: this document, modular interface skeleton, hybrid orchestrator concept — web-first, since it's the mature and independently-verifiable surface; Flutter scaffolding fixed and build-verified as a prerequisite for any Flutter AI work.
2. **Sprint 2**: generalize the existing web object detector into a real `EnvironmentState` + `HazardEngine` + `PriorityEngine` (concrete, testable, no new API dependency — extends what already works).
3. **Sprint 3**: local OCR (Tesseract.js on web; ML Kit on Flutter once the app is verified running) and local color detection (pure canvas/pixel math, both platforms).
4. **Sprint 4**: currency OCR-plus-heuristic (honestly scoped per §9), voice-command parity on Flutter.
5. **Sprint 5**: Local/Hybrid/Cloud-enhanced mode switch (config-driven), formalizing the existing Guardian pattern into the general `hybrid_ai_orchestrator`.
6. **Sprint 6**: performance measurement (real numbers only, per §9), evaluation-framework scaffolding.

Sprints 2 onward are tracked in `IMPLEMENTATION_PROGRESS.md` as they land, each with what changed, why, how it was tested, and known limitations.
