"""Background ASR worker for the M1 baseline."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from queue import Empty, Full, Queue
import threading
import time

from asr.whisper_engine import ASRResult, WhisperEngine
from streaming.buffer import SpeechSegment


@dataclass(frozen=True)
class ASRTask:
    """One speech segment submitted to the ASR worker."""

    segment: SpeechSegment
    submitted_at: float


@dataclass(frozen=True)
class ASRTaskResult:
    """Result produced for one submitted speech segment."""

    segment: SpeechSegment
    result: ASRResult | None

    submitted_at: float
    started_at: float
    completed_at: float

    queue_wait_ms: float
    segment_to_transcript_ms: float

    error: str | None = None


class ASRWorker:
    """Single-threaded background ASR worker.

    The worker owns ASR inference so Whisper never blocks
    the Audio -> VAD -> Buffer processing loop.

    A bounded input queue prevents unbounded memory growth.
    Submission is non-blocking; overloaded segments are rejected
    and reported to the caller.
    """

    def __init__(
        self,
        engine: WhisperEngine,
        queue_size: int = 2,
        logger: logging.Logger | None = None,
    ) -> None:
        if queue_size <= 0:
            raise ValueError(
                "queue_size must be greater than zero."
            )

        self.engine = engine
        self.queue_size = queue_size

        self.logger = (
            logger
            if logger is not None
            else logging.getLogger(__name__)
        )

        self._input_queue: Queue[
            ASRTask | None
        ] = Queue(maxsize=queue_size)

        self._result_queue: Queue[
            ASRTaskResult
        ] = Queue()

        self._thread: threading.Thread | None = None
        self._started = False
        self._stop_requested = False

        self._submitted_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._dropped_count = 0

        self._lock = threading.Lock()

    @property
    def started(self) -> bool:
        """Return whether the worker thread is running."""

        return self._started

    @property
    def submitted_count(self) -> int:
        """Return number of segments accepted by the worker."""

        with self._lock:
            return self._submitted_count

    @property
    def completed_count(self) -> int:
        """Return number of successfully completed ASR tasks."""

        with self._lock:
            return self._completed_count

    @property
    def failed_count(self) -> int:
        """Return number of ASR tasks that failed."""

        with self._lock:
            return self._failed_count

    @property
    def dropped_count(self) -> int:
        """Return number of segments rejected because the queue was full."""

        with self._lock:
            return self._dropped_count

    @property
    def pending_count(self) -> int:
        """Return number of tasks currently waiting in the input queue."""

        return self._input_queue.qsize()

    def start(self) -> None:
        """Start the background worker."""

        if self._started:
            return

        self._stop_requested = False

        self._thread = threading.Thread(
            target=self._run,
            name="m1-asr-worker",
            daemon=True,
        )

        self._thread.start()
        self._started = True

        self.logger.info(
            "M1 ASR worker started."
        )

    def submit(
        self,
        segment: SpeechSegment,
    ) -> bool:
        """Submit one SpeechSegment without blocking.

        Returns:
            True if accepted.
            False if the bounded queue is full.
        """

        if not self._started:
            raise RuntimeError(
                "ASR worker must be started before submission."
            )

        if self._stop_requested:
            raise RuntimeError(
                "ASR worker is stopping; new work cannot be submitted."
            )

        if segment.samples.size == 0:
            raise ValueError(
                "Cannot submit an empty SpeechSegment."
            )

        task = ASRTask(
            segment=segment,
            submitted_at=time.perf_counter(),
        )

        try:
            self._input_queue.put_nowait(task)
        except Full:
            with self._lock:
                self._dropped_count += 1

            self.logger.warning(
                "M1 ASR queue full; segment rejected | "
                "start_seq=%d | end_seq=%d | "
                "queue_size=%d",
                segment.start_sequence_number,
                segment.end_sequence_number,
                self.queue_size,
            )

            return False

        with self._lock:
            self._submitted_count += 1

        self.logger.info(
            "M1 ASR segment submitted | "
            "start_seq=%d | end_seq=%d | "
            "duration_ms=%.1f | pending=%d",
            segment.start_sequence_number,
            segment.end_sequence_number,
            segment.duration_ms,
            self.pending_count,
        )

        return True

    def poll_results(
        self,
    ) -> list[ASRTaskResult]:
        """Return all currently available ASR results."""

        results: list[ASRTaskResult] = []

        while True:
            try:
                results.append(
                    self._result_queue.get_nowait()
                )
            except Empty:
                break

        return results

    def wait_until_idle(
        self,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Wait until queued/in-progress ASR work finishes."""

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        deadline = (
            time.perf_counter()
            + timeout_seconds
        )

        while time.perf_counter() < deadline:
            if (
                self.pending_count == 0
                and self._input_queue.unfinished_tasks == 0
            ):
                return

            time.sleep(0.01)

        self.logger.warning(
            "Timed out waiting for ASR worker to become idle | "
            "pending=%d | unfinished=%d",
            self.pending_count,
            self._input_queue.unfinished_tasks,
        )

    def stop(
        self,
        drain: bool = True,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Stop the worker.

        Args:
            drain:
                When True, process all queued segments before exit.
                When False, queued work is abandoned.
            timeout_seconds:
                Maximum join time.
        """

        if not self._started:
            return

        self._stop_requested = True

        if not drain:
            while True:
                try:
                    self._input_queue.get_nowait()
                    self._input_queue.task_done()
                except Empty:
                    break

        self._input_queue.put(None)

        if self._thread is not None:
            self._thread.join(
                timeout=timeout_seconds
            )

            if self._thread.is_alive():
                self.logger.warning(
                    "M1 ASR worker did not stop within "
                    "%.1f seconds.",
                    timeout_seconds,
                )

        self._started = False

        self.logger.info(
            "M1 ASR worker stopped."
        )

    def _run(self) -> None:
        """Worker thread loop."""

        while True:
            task = self._input_queue.get()

            if task is None:
                self._input_queue.task_done()
                break

            try:
                self._process_task(task)

            finally:
                self._input_queue.task_done()

    def _process_task(
        self,
        task: ASRTask,
    ) -> None:
        """Run ASR for one SpeechSegment."""

        segment = task.segment

        started_at = time.perf_counter()

        queue_wait_ms = (
            started_at
            - task.submitted_at
        ) * 1000.0

        self.logger.info(
            "M1 ASR inference started | "
            "start_seq=%d | end_seq=%d | "
            "queue_wait_ms=%.1f",
            segment.start_sequence_number,
            segment.end_sequence_number,
            queue_wait_ms,
        )

        try:
            # getattr, not a required field: M1's SpeechSegment has no
            # initial_prompt attribute, so this is always None there -
            # byte-identical to today's behavior. Only M2's
            # ProcessingWindow-backed adapter (streaming/streaming_asr.py)
            # ever sets it.
            initial_prompt = getattr(segment, "initial_prompt", None)

            result = self.engine.transcribe(
                audio=segment.samples,
                sample_rate=segment.sample_rate,
                initial_prompt=initial_prompt,
            )

            completed_at = time.perf_counter()

            segment_to_transcript_ms = (
                completed_at
                - segment.end_timestamp
            ) * 1000.0

            with self._lock:
                self._completed_count += 1

            task_result = ASRTaskResult(
                segment=segment,
                result=result,
                submitted_at=task.submitted_at,
                started_at=started_at,
                completed_at=completed_at,
                queue_wait_ms=queue_wait_ms,
                segment_to_transcript_ms=(
                    segment_to_transcript_ms
                ),
                error=None,
            )

            self._result_queue.put(
                task_result
            )

            self.logger.info(
                "M1 ASR inference completed | "
                "start_seq=%d | end_seq=%d | "
                "inference_ms=%.1f | rtf=%.3f | "
                "queue_wait_ms=%.1f | "
                "segment_to_transcript_ms=%.1f",
                segment.start_sequence_number,
                segment.end_sequence_number,
                result.inference_time_ms,
                result.rtf,
                queue_wait_ms,
                segment_to_transcript_ms,
            )

        except Exception as exc:
            completed_at = time.perf_counter()

            with self._lock:
                self._failed_count += 1

            error_message = (
                f"{type(exc).__name__}: {exc}"
            )

            task_result = ASRTaskResult(
                segment=segment,
                result=None,
                submitted_at=task.submitted_at,
                started_at=started_at,
                completed_at=completed_at,
                queue_wait_ms=queue_wait_ms,
                segment_to_transcript_ms=(
                    (completed_at - segment.end_timestamp)
                    * 1000.0
                ),
                error=error_message,
            )

            self._result_queue.put(
                task_result
            )

            self.logger.error(
                "M1 ASR inference failed | "
                "start_seq=%d | end_seq=%d | %s",
                segment.start_sequence_number,
                segment.end_sequence_number,
                error_message,
            )