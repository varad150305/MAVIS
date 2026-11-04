import statistics
import threading
import time


TIMING_METRIC_NAMES = (
    "camera_read_ms", "total_localization_ms", "selection_ms", "sharpness_ms",
    "stability_ms", "duplicate_check_ms", "queue_wait_ms",
    "ocr_preprocessing_ms", "ocr_inference_ms", "layout_reconstruction_ms",
    "total_capture_latency_ms", "candidate_detection_to_ocr_result_ms",
    "total_processing_ms", "postprocessing_ms",
)

# Counts, ratios, and other non-duration magnitudes (e.g. the
# duplicate-detection fingerprint distance) - keep them out of the
# millisecond report so they never get mislabeled as "ms".
COUNT_METRIC_NAMES = (
    "duplicate_distance",
)


class PrototypeProfiler:
    """Thread-safe rolling baseline measurements for the prototype."""

    def __init__(self):
        self.lock = threading.Lock()
        self.started_at = time.perf_counter()
        self.counters = {}
        self.samples = {}

    def count(self, name, amount=1):
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + amount

    def sample(self, name, value):
        with self.lock:
            self.samples.setdefault(name, []).append(float(value))

    def snapshot(self):
        with self.lock:
            return {
                "counters": dict(self.counters),
                "samples": {name: list(values) for name, values in self.samples.items()},
                "elapsed_s": time.perf_counter() - self.started_at,
            }

    @staticmethod
    def summary(values):
        if not values:
            return {"avg": None, "p50": None, "p95": None, "count": 0}
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return {
            "avg": statistics.fmean(ordered),
            "p50": statistics.median(ordered),
            "p95": ordered[p95_index],
            "count": len(ordered),
        }

    def report(self, reset=False):
        snapshot = self.snapshot()
        counters = snapshot["counters"]
        samples = snapshot["samples"]
        elapsed = max(snapshot["elapsed_s"], 1e-9)
        frames = counters.get("frames_read", 0)
        displayed = counters.get("frames_displayed", 0)
        print("\n----------------------------------------")
        print("LATENCY REPORT")
        print("----------------------------------------")
        print(f"Camera read FPS: {frames / elapsed:.2f}")
        print(f"Displayed FPS: {displayed / elapsed:.2f}")
        for name in TIMING_METRIC_NAMES:
            result = self.summary(samples.get(name, []))
            if result["count"]:
                print(
                    f"{name}: avg={result['avg']:.2f} ms "
                    f"p50={result['p50']:.2f} ms p95={result['p95']:.2f} ms "
                    f"n={result['count']}"
                )
        print("Counts/dimensions (not durations):")
        for name in COUNT_METRIC_NAMES:
            result = self.summary(samples.get(name, []))
            if result["count"]:
                print(
                    f"{name}: avg={result['avg']:.2f} "
                    f"p50={result['p50']:.2f} p95={result['p95']:.2f} "
                    f"n={result['count']}"
                )
        print("Quality gates:")
        for name in (
            "frames_checked", "candidates_rejected_duplicate",
            "candidates_rejected_low_confidence",
            "candidates_accepted", "ocr_calls",
            "ocr_queue_full", "frames_skipped_by_capture",
            "selector_commit_dropped",
        ):
            print(f"  {name}: {counters.get(name, 0)}")
        queue_depth = self.summary(samples.get("ocr_queue_depth", []))
        if queue_depth["count"]:
            print(
                f"  ocr_queue_depth: avg={queue_depth['avg']:.2f} "
                f"p50={queue_depth['p50']:.2f} p95={queue_depth['p95']:.2f}"
            )
        print("----------------------------------------\n")
        if reset:
            with self.lock:
                self.started_at = time.perf_counter()
                self.counters.clear()
                self.samples.clear()
        return snapshot


profiler = PrototypeProfiler()
