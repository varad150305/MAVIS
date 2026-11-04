"""Product launcher: live English speech -> Hindi translation on CPU.

    .venv\\Scripts\\python.exe -m product.live_translate                 # microphone, M2 + MT3-C
    .venv\\Scripts\\python.exe -m product.live_translate --wav FILE.wav  # same path, 1x WAV replay
    .venv\\Scripts\\python.exe -m product.live_translate --system m1     # frozen M1 baseline, mic
    .venv\\Scripts\\python.exe -m product.live_translate --list-devices

No server or network endpoint is involved: the application is a console
program. Each session writes everything (logs, events, transcript,
decisions, summary) to its own folder under product_outputs/sessions/.
Stop with Ctrl+C (or --duration); the session then drains outstanding ASR
and MT work, finalizes, and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from product import runtime

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "product" / "configs" / "live_m2_mt3c.yaml"
DEFAULT_M1_CONFIG = REPO_ROOT / "product" / "configs" / "live_m1.yaml"
M1_CONSOLE_PREFIXES = ("TRANSCRIPT:", "HINDI:", "ASR ERROR", "TRANSLATION ERROR", "SEGMENT READY", "FINAL SEGMENT")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live English->Hindi speech translation (CPU, console)")
    parser.add_argument("--system", choices=("m2", "m1"), default="m2",
                        help="m2 = streaming M2 + scheduled MT (default); m1 = frozen fixed-buffer baseline")
    parser.add_argument("--config", help="YAML config (default: product/configs/live_m2_mt3c.yaml or live_m1.yaml)")
    parser.add_argument("--wav", help="replay this 16 kHz WAV at 1x instead of the microphone")
    parser.add_argument("--device", help="input device index or name (default: config / system default)")
    parser.add_argument("--duration", type=float, help="stop automatically after this many seconds of audio")
    parser.add_argument("--policy", choices=("mt3c", "eager", "fixed"), help="MT invocation policy (m2 only)")
    parser.add_argument("--fixed-k", type=int, help="K for --policy fixed")
    parser.add_argument("--session-name", help="suffix for the session folder name")
    parser.add_argument("--list-devices", action="store_true", help="list audio input devices and exit")
    parser.add_argument("--save-audio", action="store_true",
                        help="save the consumed 16 kHz audio as input.wav in the session folder (replay with --wav)")
    parser.add_argument("--session-id", help="session identifier recorded in every session file "
                        "(default: a new random id; product.session passes its own)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set asr.model_size=base (YAML value; repeatable)")
    return parser.parse_args(argv)


def apply_overrides(config: Any, overrides: list[str]) -> dict[str, Any]:
    """Apply dotted KEY=VALUE overrides to the loaded config (session-local)."""

    import yaml

    applied: dict[str, Any] = {}
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        value = yaml.safe_load(raw) if raw.strip() else None
        node = config.data
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise ValueError(f"--set {key}: {part!r} is not a config section")
            node = node[part]
        if parts[-1] not in node and parts[0] != "product":
            raise ValueError(f"--set {key}: unknown key (only product.* keys may be added)")
        node[parts[-1]] = value
        applied[key.strip()] = value
    return applied


def list_devices(console) -> int:
    import sounddevice as sd

    default_in = sd.default.device[0]
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            api = sd.query_hostapis(info["hostapi"])["name"]
            mark = "*" if index == default_in else " "
            console.write(f"{mark} {index:3d}  {info['name']}  [{api}]\n")
    console.write("(* = default input)\n")
    return 0


def _device_arg(value: str | None):
    if value is None:
        return None
    return int(value) if value.strip().lstrip("-").isdigit() else value


class LineFilter:
    """stdout replacement: every line goes to `log`, selected lines also to `console`."""

    def __init__(self, console, log, prefixes: tuple[str, ...]) -> None:
        self.console, self.log, self.prefixes = console, log, prefixes
        self._buffer = ""

    def write(self, text: str) -> int:
        self.log.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.startswith(self.prefixes):
                self.console.write(line + "\n")
                self.console.flush()
        return len(text)

    def flush(self) -> None:
        self.log.flush()


def make_session_dir(root: Path, system: str, source: str, name: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = root / (f"{stamp}_{system}_{source}" + (f"_{name}" if name else ""))
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def redirect_logging(config: Any, session: Path, system: str) -> None:
    """Point every log/output path of the loaded config into the session folder."""

    config.data["logging"]["file"] = str(session / f"{system}.log")
    config.data["logging"]["json_file"] = str(session / f"{system}_events.jsonl")
    config.data["logging"]["console"] = False
    config.data.setdefault("output", {})["transcript_file"] = str(session / "transcript.jsonl")


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0], "python_machine": platform.machine(),
                            "platform": platform.platform(), "processor": platform.processor(),
                            "on_ac_power": runtime.on_ac_power()}
    for module in ("torch", "faster_whisper", "ctranslate2", "transformers", "sounddevice", "numpy"):
        try:
            info[module] = __import__(module).__version__
        except Exception:
            info[module] = None
    return info


def error_lines(log_path: Path, limit: int = 5) -> list[str]:
    if not log_path.exists():
        return []
    lines = [l.rstrip() for l in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if "| ERROR |" in l]
    return lines[-limit:]


# =======================================================================


def run_m1(args, config, session: Path, console, source) -> dict[str, Any]:
    import main as main_module
    from unittest.mock import patch

    from utils.logger import EventLogger

    lc = config.get("logging")
    logger = EventLogger(level=lc["level"], console=False, file_enabled=True, file_path=lc["file"],
                         json_enabled=True, json_path=lc["json_file"])
    with (session / "m1_console.txt").open("w", encoding="utf-8") as log_file:
        filtered = LineFilter(console, log_file, M1_CONSOLE_PREFIXES)
        old_stdout = sys.stdout
        sys.stdout = filtered
        try:
            with patch.object(main_module, "build_audio_stream", return_value=source):
                exit_code = main_module.run_m1(config, logger)
        finally:
            sys.stdout = old_stdout
    return {"run_exit_code": exit_code}


def run_m2(args, config, session: Path, console, source, product_cfg: dict[str, Any],
           event_sink=None, on_display=None) -> dict[str, Any]:
    import main as main_module
    import torch

    from product import live_pipeline as lp
    from translation.translator import TranslationEngine
    from utils.logger import EventLogger

    clock = source.clock
    display = lp.Display(console, session / "transcript.jsonl", clock, sink=event_sink)
    if on_display is not None:
        on_display(display)

    policy_name = args.policy or product_cfg.get("policy", "mt3c")
    mt3c_params = mt3c_sha = None
    if policy_name == "mt3c":
        mt3c_params, mt3c_sha = lp.load_mt3c_parameters(REPO_ROOT / product_cfg["mt3c_parameters"],
                                                        product_cfg.get("mt3c_parameters_sha256"))
    policy_factory = lp.make_policy_factory(policy_name, fixed_k=args.fixed_k or product_cfg.get("fixed_k", 8),
                                            mt3c_params=mt3c_params, mt3c_sha256=mt3c_sha)
    seg_cfg = product_cfg.get("segment", {})
    rules = lp.SegmentRules(close_after_words=seg_cfg.get("close_after_words", 20),
                            idle_close_s=seg_cfg.get("idle_close_s", 2.5),
                            hard_max_words=seg_cfg.get("hard_max_words", 60),
                            sentence_gap_s=seg_cfg.get("sentence_gap_s"),
                            sentence_min_words=seg_cfg.get("sentence_min_words", 3),
                            long_close_needs_pause=bool(seg_cfg.get("long_close_needs_pause", False)),
                            long_close_max_words=seg_cfg.get("long_close_max_words"))

    # The main thread runs the VAD: keep it at 1 torch thread (silero_vad's
    # own import-time setting, which main.py imports before us anyway). MT
    # gets its thread count inside the MT worker thread - see MTWorker.
    torch.set_num_threads(1)
    mt_threads = int(config.get("cpu", "torch_threads", default=8))
    t = config.get("translation")
    console.write("Loading translation model...\n")
    console.flush()
    load_start = time.perf_counter()
    engine = TranslationEngine(model_name=t["model"], device=t["device"], max_input_tokens=t["max_input_tokens"],
                               num_beams=t["num_beams"], max_new_tokens=t["max_new_tokens"], revision=t.get("revision"))

    worker = lp.MTWorker(engine, clock, display.translation, torch_threads=mt_threads, warm_up=True)
    worker.start()
    worker.ready.wait()
    translator = lp.AsyncSegmentTranslator(worker, clock)

    fq_cfg = product_cfg.get("final_mt") or {}
    final_worker = final_engine = None
    if fq_cfg.get("enabled"):
        from product.final_mt import DEFAULT_MODEL_DIR, NllbCt2Translator

        console.write("Loading final-quality translation model (NLLB-600M, CTranslate2 int8)...\n")
        console.flush()
        final_engine = NllbCt2Translator(REPO_ROOT / fq_cfg["model_dir"] if fq_cfg.get("model_dir") else DEFAULT_MODEL_DIR,
                                         threads=int(fq_cfg.get("threads", 4)))
        final_worker = lp.MTWorker(final_engine, clock, display.final_translation, warm_up=True)
        final_worker.name = "product-final-mt-worker"
        final_worker.start()
        final_worker.ready.wait()
    mt_load_s = time.perf_counter() - load_start
    controller = lp.SegmentController(policy_factory, translator, clock, rules, display, final_worker)
    controller.start()

    lc = config.get("logging")
    logger = EventLogger(level=lc["level"], console=False, file_enabled=True, file_path=lc["file"],
                         json_enabled=True, json_path=lc["json_file"])
    console.write(f"Loading speech recognition (Whisper {config.get('asr', 'model_size')}, VAD) and warming up...\n")
    console.flush()
    repair_cfg = product_cfg.get("boundary_repair") or {}
    repair = (lp.BoundaryRepair(tolerance_s=float(repair_cfg.get("tolerance_s", 0.3)),
                                max_ngram=int(repair_cfg.get("max_ngram", 5)))
              if repair_cfg.get("enabled") else None)
    carry = int(product_cfg.get("prompt_carry_words") or 0)
    trim_keep_s = float(product_cfg.get("trim_keep_s") or 0.0)
    prompt_min_words = int(product_cfg.get("prompt_min_words") or 0)
    final_window_full_max_s = float(product_cfg.get("final_window_full_max_s") or 0.0)
    cross_utterance_dedup = bool(product_cfg.get("cross_utterance_dedup"))
    provisional = bool(product_cfg.get("display_provisional_en"))
    guard_cfg = product_cfg.get("repetition_guard") or {}
    guard = (lp.RepetitionGuard(max_ngram=int(guard_cfg.get("max_ngram", 3)), min_run=int(guard_cfg.get("min_run", 4)))
             if guard_cfg.get("enabled") else None)
    with (session / "m2_console.txt").open("w", encoding="utf-8") as m2_console:
        exit_code, hook = lp.run_m2_with_translation(main_module, config, logger, source, controller, m2_console,
                                                     product_cfg.get("vad_reset_after_silence_s"), repair, carry,
                                                     guard, trim_keep_s, prompt_min_words=prompt_min_words,
                                                     on_provisional=display.provisional if provisional else None,
                                                     on_record=display.event if event_sink is not None else None,
                                                     final_window_full_max_s=final_window_full_max_s,
                                                     cross_utterance_dedup=cross_utterance_dedup)

    t_end = clock()
    controller.stop(t_end)
    drained = worker.shutdown()
    final_drained = final_worker.shutdown() if final_worker is not None else None

    with (session / "decisions.jsonl").open("w", encoding="utf-8") as f:
        for seg in controller.segments:
            for d in seg.decisions:
                f.write(json.dumps({"segment_id": seg.segment_id, **d}, ensure_ascii=False, default=str) + "\n")
    segments_out = [{"segment_id": s.segment_id, "opened_s": s.opened_s, "closed_s": s.closed_s,
                     "close_reason": s.close_reason, "utterance_ids": s.utterance_ids, "words": len(s.words),
                     "source": " ".join(s.words), "final_hindi": display.finals.get(s.segment_id),
                     "final_kind": display.final_kinds.get(s.segment_id),
                     "last_draft_hindi": _last_draft(display, s), "mt_calls": len(s.calls)}
                    for s in controller.segments]
    (session / "segments.json").write_text(json.dumps(segments_out, ensure_ascii=False, indent=1), encoding="utf-8")
    with (session / "asr_hypotheses.jsonl").open("w", encoding="utf-8") as f:
        for h in hook.hypotheses:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")

    return {
        "run_exit_code": exit_code,
        "policy": policy_name,
        "policy_params": controller.scheduler.policy.params(),
        "segment_rules": asdict(rules),
        "mt_model_load_and_warmup_s": mt_load_s,
        "mt_warmup_ms": worker.warm_up_ms,
        "mt_worker_torch_threads": worker.thread_torch_threads,
        "main_thread_torch_threads_after_run": torch.get_num_threads(),
        "controller_error": controller.error,
        "controller_alive_after_stop": controller.is_alive(),
        "mt_worker_drained": drained,
        "final_mt": final_engine.settings() if final_engine is not None else None,
        "final_mt_worker_drained": final_drained,
        "mt_max_queue_depth": worker.max_queue_depth,
        "deadline_timer_lateness_s": lp._dist(controller.timer_lateness_s),
        "stable_update_events": len(hook.events),
        "vad_reset_after_silence_s": product_cfg.get("vad_reset_after_silence_s"),
        "vad_resets": hook.vad_resets,
        "boundary_repair": asdict(repair) if repair is not None else None,
        "boundary_repairs": hook.repairs,
        "prompt_carry_words": carry,
        "trim_keep_s": trim_keep_s,
        "context_trims": hook.trims,
        "prompt_min_words": prompt_min_words,
        "prompts_dropped": hook.prompts_dropped,
        "final_coverage_gaps": hook.final_coverage_gaps,
        "final_window_full_max_s": final_window_full_max_s,
        "final_window_restores": hook.final_window_restores,
        "cross_utterance_dedup": cross_utterance_dedup,
        "cross_boundary_dedups": hook.cross_boundary_dedups,
        "display_provisional_en": provisional,
        "repetition_guard": {"max_ngram": guard.max_ngram, "min_run": guard.min_run, "dropped_words": guard.dropped}
        if guard is not None else None,
        "asr_model": config.get("asr", "model_size"),
        "asr_language": config.get("asr", "language"),
        "vad_minimum_silence_duration_ms": config.get("vad", "minimum_silence_duration_ms"),
        "translation": lp.translation_metrics(controller.segments, display),
        "asr": lp.asr_metrics(Path(lc["json_file"])),
        "user_latency": lp.user_latency_metrics(hook, controller.segments, display),
    }


def _last_draft(display, segment) -> str | None:
    """Hindi of the segment's last successful draft-MT call (for comparison with the final)."""

    for call in reversed(segment.calls):
        result = display.result_for(segment.segment_id, call.invocation_id)
        if result is not None and not result.error:
            return result.target_text
    return None


def main(argv: list[str] | None = None) -> int:
    runtime.prefer_offline_models()
    console = runtime.utf8_console(sys.stdout)
    args = parse_args(argv)
    if args.list_devices:
        return list_devices(console)
    code, _summary = run_session(argv, console)
    return code


def run_session(argv: list[str] | None, console, *, event_sink=None, stop_event: threading.Event | None = None,
                ready_callback=None) -> tuple[int, dict[str, Any] | None]:
    """One session of the product pipeline; returns (exit code, summary).

    The CLI calls it with the terminal as `console`. product.session calls it
    from a worker thread with `event_sink` (structured records), its own
    `stop_event` (set -> the source raises the KeyboardInterrupt run_m2
    already drains on) and `ready_callback(ready_in_s, info)`.

    Phase 6: session_start.json and lifecycle.jsonl (product.lifecycle) are
    written while the session runs, a stop request is recorded when it is
    made (if `stop_event` is a product.lifecycle.StopSignal), microphone
    stalls and audio cleanup failures are reported, and a pipeline exception
    still produces a session summary."""

    runtime.prefer_offline_models()
    args = parse_args(argv)

    import copy
    import traceback
    import uuid

    from product.lifecycle import SessionRecorder, atomic_write_json
    from product.sources import LiveMicSource, PacedWavSource, check_input_device
    from utils.config import ConfigError, load_config

    config_path = Path(args.config) if args.config else (DEFAULT_M1_CONFIG if args.system == "m1" else DEFAULT_CONFIG)
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        console.write(f"Configuration error: {exc}\n")
        return 1, {"error": f"configuration error: {exc}"}
    expected = "M1" if args.system == "m1" else "M2"
    if config.system_name != expected:
        console.write(f"Configuration error: --system {args.system} needs system.name {expected!r}, "
                      f"{config_path} has {config.system_name!r}\n")
        return 1, {"error": "configuration error: wrong system"}

    try:
        overrides = apply_overrides(config, args.set)
    except ValueError as exc:
        console.write(f"Configuration error: {exc}\n")
        return 1, {"error": f"configuration error: {exc}"}
    product_cfg = config.get("product", default={}) or {}
    root = REPO_ROOT / product_cfg.get("output_root", "runtime_data/sessions")  # handoff: package-local
    session = make_session_dir(root, args.system, "wav" if args.wav else "mic", args.session_name)
    redirect_logging(config, session, args.system)
    session_id = args.session_id or uuid.uuid4().hex[:12]
    recorder = SessionRecorder(session, session_id)

    started_wall = datetime.now().isoformat(timespec="seconds")
    # relative to the repository when inside it; a host may choose an output root elsewhere
    session_dir = str(session.relative_to(REPO_ROOT)) if session.is_relative_to(REPO_ROOT) else str(session)
    summary: dict[str, Any] = {"session_id": session_id, "session_dir": session_dir, "started": started_wall,
                               "system": args.system, "config": str(config_path), "args": vars(args),
                               "environment": environment_info(), "overrides": overrides}
    recorder.write_start({"session_dir": session_dir, "started": started_wall, "system": args.system,
                          "config": str(config_path), "argv": list(argv) if argv is not None else sys.argv[1:],
                          "args": vars(args), "overrides": overrides, "environment": summary["environment"],
                          "effective_config": copy.deepcopy(config.data)})
    recorder.record("created", session_dir=session_dir, on_ac_power=summary["environment"].get("on_ac_power"))
    console.write(f"Session folder: {summary['session_dir']}\n")

    stop_event = stop_event if stop_event is not None else threading.Event()
    if hasattr(stop_event, "subscribe"):
        stop_event.subscribe(lambda sig: recorder.record("stop_requested", reason=sig.reason,
                                                         requested_wall=sig.requested_wall))
    device = _device_arg(args.device) if args.device is not None else config.get("audio", "device")
    audio_cfg = config.get("audio")
    if args.wav:
        source = PacedWavSource(args.wav, audio_cfg["chunk_duration_ms"], stop_event, args.duration)
        summary["input"] = {"wav": str(args.wav), "audio_duration_s": source.audio_duration_s}
    else:
        try:
            device_name = check_input_device(device, audio_cfg["sample_rate"], audio_cfg["channels"])
        except RuntimeError as exc:
            console.write(f"ERROR: {exc}\n")
            summary["error"] = str(exc)
            summary["exit_code"] = 2
            atomic_write_json(session / "session_summary.json", summary, indent=1)
            recorder.record("failed", exit_code=2, error=str(exc), where="input device check")
            return 2, summary
        from audio.audio_stream import AudioStream

        stream = AudioStream(sample_rate=audio_cfg["sample_rate"], channels=audio_cfg["channels"],
                             chunk_duration_ms=audio_cfg["chunk_duration_ms"],
                             buffer_size=max(1, round(audio_cfg["buffer_size_ms"] / audio_cfg["chunk_duration_ms"])),
                             device=device)
        stall_s = product_cfg.get("audio_stall_s", 2.0)
        source = LiveMicSource(stream, stop_event, args.duration,
                               stall_after_s=float(stall_s) if stall_s is not None else None)
        summary["input"] = {"device": device, "device_name": device_name}

    display_ref: dict[str, Any] = {}

    def on_audio_health(record: dict[str, Any]) -> None:
        recorder.record(record["type"], **{k: v for k, v in record.items() if k != "type"})
        display = display_ref.get("display")
        if display is not None:              # M2: console line + transcript + event sink
            display.audio_health(record)
        else:
            console.write(f"WARNING: {record['type']}: "
                          f"{record.get('error') or record.get('no_frames_for_s') or record.get('stalled_for_s')}\n")
            if event_sink is not None:
                event_sink({k: v for k, v in record.items() if k != "traceback"})

    source.on_audio_health = on_audio_health

    t_launch = time.perf_counter()
    ready: dict[str, float] = {}

    def on_start() -> None:
        ready["s"] = time.perf_counter() - t_launch
        console.write(f"Listening ({'replaying ' + Path(args.wav).name if args.wav else summary['input']['device_name']})"
                      f" - ready after {ready['s']:.1f} s. Press Ctrl+C to stop.\n")
        if args.system != "m1":
            from product.live_pipeline import CONSOLE_LEGEND

            console.write(CONSOLE_LEGEND)
        console.flush()
        recorder.record("ready", ready_in_s=round(ready["s"], 3), stop_already_requested=stop_event.is_set())
        if ready_callback is not None:
            ready_callback(ready["s"], {"on_ac_power": summary["environment"].get("on_ac_power"),
                                        "input": summary.get("input"), "session_dir": summary["session_dir"],
                                        "asr_model": config.get("asr", "model_size")})

    source.on_start = on_start
    from product.sources import InputMonitor

    source.monitor = InputMonitor(audio_cfg["sample_rate"], session / "input.wav" if args.save_audio else None)
    summary["power_throttling_opt_out"] = runtime.disable_power_throttling()
    summary["keep_awake"] = runtime.keep_system_awake(True)
    on_sample = None
    if event_sink is not None:
        def on_sample(sample):
            event_sink({"type": "resources", "t_s": source.clock(), "cores_busy": sample["cores_busy"],
                        "working_set_mb": sample["working_set_mb"], "dropped_frames": source.dropped_frames})
    sampler = runtime.ResourceSampler(on_sample=on_sample)
    sampler.start()
    cleanup_errors: list[dict[str, Any]] = []

    def cleanup(op: str, fn) -> Any:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - recorded, never reported as success
            cleanup_errors.append({"op": op, "error": f"{type(exc).__name__}: {exc}",
                                   "traceback": traceback.format_exc()})
            recorder.record("cleanup_error", op=op, error=f"{type(exc).__name__}: {exc}")
            return None

    recorder.record("loading")
    try:
        if args.system == "m1":
            summary.update(run_m1(args, config, session, console, source))
        else:
            summary.update(run_m2(args, config, session, console, source, product_cfg, event_sink=event_sink,
                                  on_display=lambda d: display_ref.__setitem__("display", d)))
    except KeyboardInterrupt:
        console.write("Interrupted before the session could start or finish cleanly.\n")
        summary["interrupted_outside_run_loop"] = True
    except Exception as exc:  # noqa: BLE001 - finish the session record instead of losing it
        console.write(f"ERROR: the pipeline failed: {type(exc).__name__}: {exc}\n")
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["error_traceback"] = traceback.format_exc()
        recorder.record("pipeline_error", error=summary["error"])
    finally:
        display_ref.clear()                  # the pipeline (and its Display) is finished
        cleanup("close_log_handlers", _close_session_log_handlers)
        summary["resources"] = cleanup("resource_sampler.stop", sampler.stop) or {}
        cleanup("keep_system_awake(False)", lambda: runtime.keep_system_awake(False))
        cleanup("input_monitor.close", source.monitor.close)
        summary["input_level"] = source.monitor.summary()
    recorder.record("pipeline_finished", run_exit_code=summary.get("run_exit_code"),
                    stop_reason=source.stop_reason, error=summary.get("error"))

    # steady state = samples taken after audio capture began (no model loading)
    listening_at = ready.get("s")
    steady = [s["cores_busy"] for s in sampler.samples if listening_at is not None and s["t_s"] > listening_at + 1.0]
    if steady:
        steady.sort()
        summary["resources"]["listening_mean_cores_busy"] = sum(steady) / len(steady)
        summary["resources"]["listening_p95_cores_busy"] = steady[int(0.95 * (len(steady) - 1))]
    (session / "resource_samples.json").write_text(
        json.dumps({"listening_at_s": listening_at, "samples": sampler.samples}, indent=0), encoding="utf-8")

    summary["startup_to_listening_s"] = ready.get("s")
    summary["stop_reason"] = source.stop_reason or ("Ctrl+C" if summary.get("run_exit_code") == 0 else None)
    summary["audio"] = {"frames_consumed": source.frames_read, "audio_consumed_s": source.frames_read * source.frame_s,
                        "dropped_frames": source.dropped_frames,
                        "frame_consume_lag_s": _summarize(source.consume_lag_s)}
    if hasattr(source, "stall_report"):
        summary["audio"]["stall_threshold_s"] = source.stall_after_s
        summary["audio"]["stalls"] = source.stall_report()
    # every cleanup step that failed (audio stream stop/close, monitor, sampler, ...): never silently "ok"
    all_cleanup = list(getattr(source, "cleanup_errors", [])) + cleanup_errors
    summary["cleanup"] = {"ok": not all_cleanup, "errors": all_cleanup}
    summary["errors_logged"] = error_lines(Path(config.get("logging", "file")))
    code = 0 if summary.get("run_exit_code") == 0 and not summary.get("controller_error") \
        and not summary.get("error") else 1
    summary["exit_code"] = code
    summary["lifecycle_write_errors"] = recorder.write_errors
    try:
        atomic_write_json(session / "session_summary.json", summary, indent=1)
    except Exception as exc:  # noqa: BLE001
        recorder.record("failed", exit_code=code, error=f"session_summary.json not written: {exc}")
        raise
    recorder.record("finalized" if code == 0 else "failed", exit_code=code, cleanup_ok=summary["cleanup"]["ok"],
                    error=summary.get("error") or summary.get("controller_error"))
    print_summary(console, summary)
    return code, summary


def _close_session_log_handlers() -> None:
    """Close the research EventLogger's file handlers after a session. It
    clears (but does not close) old handlers when the next one is created,
    so in a long-lived host process every session's m2.log would stay open
    (and its folder locked on Windows). The logger itself is not modified."""

    import logging

    logger = logging.getLogger("live_translation")
    for handler in list(logger.handlers):
        try:
            handler.close()
        finally:
            logger.removeHandler(handler)


def _summarize(values: list[float]) -> dict[str, Any]:
    from product.live_pipeline import _dist

    return _dist(values)


def print_summary(console, s: dict[str, Any]) -> None:
    def fmt(d, key, scale=1.0, unit=""):
        v = (d or {}).get(key)
        return "n/a" if v is None else f"{v * scale:.2f}{unit}"

    r = s.get("resources", {})
    lines = ["", "=== Session summary ===",
             f"exit code {s.get('run_exit_code')} | stop: {s.get('stop_reason')} | on AC power: {s['environment'].get('on_ac_power')}",
             f"startup to listening {fmt(s, 'startup_to_listening_s', unit=' s')} | audio {fmt(s['audio'], 'audio_consumed_s', unit=' s')}"
             f" | dropped frames {s['audio']['dropped_frames']}",
             f"input level: peak {s.get('input_level', {}).get('peak_dbfs')} dBFS, frame RMS p50/p90 "
             f"{(s.get('input_level', {}).get('frame_rms_dbfs') or {}).get('p50')}/"
             f"{(s.get('input_level', {}).get('frame_rms_dbfs') or {}).get('p90')} dBFS"
             + (f" | audio saved: {s['input_level']['saved_wav']}" if s.get('input_level', {}).get('saved_wav') else ""),
             f"CPU mean {fmt(r, 'mean_cores_busy')} cores (while listening {fmt(r, 'listening_mean_cores_busy')}),"
             f" p95 {fmt(r, 'p95_cores_busy')} of {r.get('logical_cpus')} | peak memory {fmt(r, 'peak_working_set_mb', unit=' MB')}"]
    if "asr" in s:
        a, tr = s["asr"], s["translation"]
        lines += [f"ASR decodes {a['decodes']} (failed {a['failed_decodes']}) | inference p50 {fmt(a['inference_ms'], 'p50', unit=' ms')}"
                  f" p95 {fmt(a['inference_ms'], 'p95', unit=' ms')} | RTF p50 {fmt(a['rtf'], 'p50')}",
                  f"segments {tr['segments']} | stable words {tr['stable_words']} | MT calls {tr['mt_calls']} {tr['mt_calls_by_reason']}"
                  f" | MT errors {tr['mt_errors']}",
                  *([f"final-quality MT: {tr['final_mt']['calls']} calls, {tr['final_mt']['errors']} errors | "
                     f"inference p50 {fmt(tr['final_mt']['inference_ms'], 'p50', unit=' ms')} | close->final p50 "
                     f"{fmt(tr['final_mt']['close_to_final_s'], 'p50', unit=' s')} p95 "
                     f"{fmt(tr['final_mt']['close_to_final_s'], 'p95', unit=' s')}"] if tr.get("final_mt") else []),
                  f"MT inference p50 {fmt(tr['mt_inference_ms'], 'p50', unit=' ms')} p95 {fmt(tr['mt_inference_ms'], 'p95', unit=' ms')}"
                  f" | word->Hindi p50 {fmt(tr['word_to_hindi_s'], 'p50', unit=' s')} p95 {fmt(tr['word_to_hindi_s'], 'p95', unit=' s')}"
                  f" max {fmt(tr['word_to_hindi_s'], 'max', unit=' s')}"]
        if s.get("controller_error"):
            lines.append(f"CONTROLLER ERROR: {s['controller_error']}")
    for st in (s.get("audio") or {}).get("stalls") or []:
        lines.append(f"AUDIO STALL: no microphone frames from {st['start_s']:.1f} s for "
                     + (f"{st['duration_s']:.1f} s" if st.get("duration_s") is not None else "the rest of the session"))
    for err in (s.get("cleanup") or {}).get("errors") or []:
        lines.append(f"CLEANUP FAILED: {err['op']}: {err['error']}")
    if s.get("error"):
        lines.append(f"PIPELINE ERROR: {s['error']}")
    for line in s.get("errors_logged", []):
        lines.append(f"logged error: {line}")
    lines.append(f"details: {s['session_dir']}")
    console.write("\n".join(lines) + "\n")
    console.flush()


if __name__ == "__main__":
    raise SystemExit(main())
