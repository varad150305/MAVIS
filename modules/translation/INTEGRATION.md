# Host integration interface

Carried forward from the product's integration contract (Phases 5–9). **Integrating this module into MAVIS is the job of the downstream MAVIS integration phase.** Nothing in this package is a MAVIS host. There is no OCR or other concurrent module here, and concurrent operation has not been validated.

## Feature flag

`product.integration.live_translation_enabled` in `product/configs/live_m2_mt3c.yaml` is **false**.

- `product.session.create_host_session(...)` raises `FeatureDisabled` while the flag is false.
- `product.session.feature_enabled()` reads the flag.
- The standalone UI and CLI are the product itself and do not check it.

To enable it later, set the flag in a copy of the config passed as `SessionOptions(config=...)`, or in the product config. That decision belongs to the integration phase.

## Using the API

```python
from product.session import SessionOptions, create_host_session

session = create_host_session(SessionOptions(device=None, save_audio=True))   # microphone
session.start()                    # returns at once; the worker process loads models (~8 s AC, ~14-17 s battery)
for event in session.poll_events():   # from the host's event loop, e.g. every 100 ms
    handle(event)
session.stop()                     # drains ASR/MT, closes the last segment, writes the session files
summary = session.summary          # = session_summary.json
```

Other entry points:
- `iter_events()`: blocking.
- `on_event=callback`: called from the thread that reads the worker. Keep it cheap and thread-safe; its exceptions are swallowed.

`SessionOptions`:
- `wav` (replay a 16 kHz mono WAV at 1×) or microphone `device`;
- `save_audio`, `show_provisional_english` (default off);
- `session_name`, `config`, `overrides` (as CLI `--set`), `duration_s`;
- `isolation` (`"process"` default, or `"thread"`).

## Constraints

- **A worker process per session (default).**
  - `product.session_worker` runs the pipeline and streams JSON lines; a stop is forwarded on its stdin.
  - The worker also stops when the host disappears (stdin closed).
  - It exits after the session files are finalized; its memory is then returned to the OS.
  - It runs with this package as its working directory.
- **One session at a time per host.** A second `start()` raises `SessionBusy`. Create a new `TranslationSession` object for each session.
- **Models load at every start:** about 8 s on AC, 14–17 s on battery.
- **Drain-safe stop.**
  - `stop()` works at any time, including while models load; it waits up to 120 s by default and is idempotent.
  - `terminate()` is the last resort; the session then ends `failed`.
- **Failures are events, not exceptions.**
  - A crash, a kill or an early exit of the worker becomes an `error` event (`where: "worker"`, naming the session folder, with its last stderr lines) plus `session_state: failed`.
  - The UI can never stay "ready" after the worker is gone.
- **Session files** go to `product.output_root` (`runtime_data/sessions`). A host may set an absolute folder with `overrides=["product.output_root=..."]`.

## Event contract: `mavis.live_translation.v1`

Every event has:
- `schema`;
- `session_id`: **drop events of any other session**;
- `seq`: strictly increasing within a session;
- `type`;
- `t_s`: session audio seconds;
- `render_s`, on display events.

| type | meaning | rendering rule |
|---|---|---|
| `session_state` | `starting` → `ready` (`ready_in_s`, `info.on_ac_power`) → `stopping` → `stopped` (`stop_reason`, `cleanup_ok`) / `failed` (`error`) | lifecycle; never `ready` after `stopping` |
| `en_provisional` | unconfirmed hypothesis (opt-in) | label "may change"; replace per `utterance_id`; never translate |
| `en_committed` | confirmed words: `segment_id`, `utterance_id`, `new_text`, `stable_text` | `stable_text` = the segment's whole confirmed English |
| `hi_draft` | provisional Hindi for `segment_id` | show as draft; replaced by later drafts; never after the final |
| `hi_draft_late` | a draft that finished after the final | do not show |
| `hi_final` | the segment's final Hindi (`kind`: `final_mt`, `draft`, `draft_fallback`) | exactly one per segment, after `segment_closed`; never overwritten |
| `segment_closed` | `reason`, `utterance_ids`, `final_source` | — |
| `mt_error`, `final_mt_error` | a translation call failed | show as an error; not a quality measure |
| `asr_decode`, `utterance_timing`, `resources` | metrics | performance panel |
| `audio_stall` / `audio_resumed` | no microphone frames for `threshold_s` / frames again | warn (do not show plain "ready") / clear |
| `cleanup_error` | audio stream stop/close failed | show; summary `cleanup.ok=false` |
| `session_summary` | full summary incl. `user_latency` | authoritative metrics |
| `error`, `status` | exception (`where`, `traceback`) / status line | — |

**Reusable models:**
- `product.ui_model.TranscriptModel`: transcript state with these rules.
- `product.metrics_panel.LiveMetrics`: panel rows. Latencies are measured from VAD speech onset; "collecting" is shown when there are too few samples. No live WER.

## What a MAVIS host still has to provide (not done, not validated)

- **Microphone ownership.** The worker opens the input device itself. A host that captures audio elsewhere must either give this module exclusive use during a session or implement routing. Several modules must not open the microphone independently.
- **CPU and memory budget.** One worker uses about 1.6 GB peak working set and about 2–2.6 cores on average (p95 4–6 of 8) on the development laptop. **Concurrency with OCR has not been measured.**
- **Power.** Latency is worse on battery than on AC.
- **Model licensing** for the intended use: NLLB-200 is CC-BY-NC 4.0. See `LICENSES_AND_MODEL_NOTICES.md`.
