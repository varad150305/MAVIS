"""Background translation worker for the M1 baseline."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from queue import Empty, Full, Queue
import threading
import time

from translation.translator import TranslationEngine, TranslationResult


@dataclass(frozen=True)
class TranslationTask:
    """One transcript submitted for translation."""

    text: str
    submitted_at: float

    # Completion time of the upstream ASR stage.
    upstream_completed_at: float

    # End time of the original speech segment.
    segment_end_timestamp: float

    start_sequence_number: int | None = None
    end_sequence_number: int | None = None


@dataclass(frozen=True)
class TranslationTaskResult:
    """Translation result plus pipeline timing metadata."""

    task: TranslationTask
    result: TranslationResult | None

    started_at: float
    completed_at: float

    queue_wait_ms: float

    # ASR completion -> translation completion.
    asr_to_translation_ms: float

    # Original speech-segment end -> translation completion.
    segment_to_translation_ms: float

    error: str | None = None


class TranslationWorker:
    """Single-threaded background translation worker.

    Translation never executes inside the microphone/VAD/ASR loop.

    The input queue is bounded so that slow translation cannot
    cause unbounded memory growth.
    """

    def __init__(
        self,
        engine: TranslationEngine,
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
            TranslationTask | None
        ] = Queue(maxsize=queue_size)

        self._result_queue: Queue[
            TranslationTaskResult
        ] = Queue()

        self._thread: threading.Thread | None = None
        self._started = False
        self._stop_requested = False

        self._submitted_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._dropped_count = 0
        self._skipped_empty_count = 0

        self._lock = threading.Lock()

    @property
    def started(self) -> bool:
        """Return whether the worker is running."""

        return self._started

    @property
    def submitted_count(self) -> int:
        """Return accepted non-empty translation tasks."""

        with self._lock:
            return self._submitted_count

    @property
    def completed_count(self) -> int:
        """Return successfully translated tasks."""

        with self._lock:
            return self._completed_count

    @property
    def failed_count(self) -> int:
        """Return translation failures."""

        with self._lock:
            return self._failed_count

    @property
    def dropped_count(self) -> int:
        """Return tasks rejected because the queue was full."""

        with self._lock:
            return self._dropped_count

    @property
    def skipped_empty_count(self) -> int:
        """Return empty transcripts skipped without inference."""

        with self._lock:
            return self._skipped_empty_count

    @property
    def pending_count(self) -> int:
        """Return number of tasks waiting in the queue."""

        return self._input_queue.qsize()

    def start(self) -> None:
        """Start the translation worker."""

        if self._started:
            return

        self._stop_requested = False

        self._thread = threading.Thread(
            target=self._run,
            name="m1-translation-worker",
            daemon=True,
        )

        self._thread.start()
        self._started = True

        self.logger.info(
            "M1 translation worker started."
        )

    def submit(
        self,
        text: str,
        upstream_completed_at: float,
        segment_end_timestamp: float,
        start_sequence_number: int | None = None,
        end_sequence_number: int | None = None,
    ) -> bool:
        """Submit one transcript without blocking.

        Empty transcripts are ignored without entering the queue.

        Returns:
            True when a non-empty transcript was accepted.
            False when the bounded queue was full or text was empty.
        """

        if not self._started:
            raise RuntimeError(
                "Translation worker must be started before submission."
            )

        if self._stop_requested:
            raise RuntimeError(
                "Translation worker is stopping; "
                "new work cannot be submitted."
            )

        if text is None:
            raise ValueError(
                "Translation text cannot be None."
            )

        normalized_text = " ".join(
            text.strip().split()
        )

        if not normalized_text:
            with self._lock:
                self._skipped_empty_count += 1

            self.logger.info(
                "M1 translation skipped empty transcript."
            )

            return False

        task = TranslationTask(
            text=normalized_text,
            submitted_at=time.perf_counter(),
            upstream_completed_at=upstream_completed_at,
            segment_end_timestamp=segment_end_timestamp,
            start_sequence_number=start_sequence_number,
            end_sequence_number=end_sequence_number,
        )

        try:
            self._input_queue.put_nowait(task)

        except Full:
            with self._lock:
                self._dropped_count += 1

            self.logger.warning(
                "M1 translation queue full; "
                "transcript rejected | "
                "start_seq=%s | end_seq=%s | "
                "queue_size=%d",
                start_sequence_number,
                end_sequence_number,
                self.queue_size,
            )

            return False

        with self._lock:
            self._submitted_count += 1

        self.logger.info(
            "M1 translation submitted | "
            "start_seq=%s | end_seq=%s | "
            "pending=%d",
            start_sequence_number,
            end_sequence_number,
            self.pending_count,
        )

        return True

    def poll_results(
        self,
    ) -> list[TranslationTaskResult]:
        """Return all currently available translation results."""

        results: list[TranslationTaskResult] = []

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
        """Wait until all queued translation work is finished."""

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
            "Timed out waiting for translation worker | "
            "pending=%d | unfinished=%d",
            self.pending_count,
            self._input_queue.unfinished_tasks,
        )

    def stop(
        self,
        drain: bool = True,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Stop the translation worker."""

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
                    "M1 translation worker did not stop within "
                    "%.1f seconds.",
                    timeout_seconds,
                )

        self._started = False

        self.logger.info(
            "M1 translation worker stopped."
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
        task: TranslationTask,
    ) -> None:
        """Translate one transcript."""

        started_at = time.perf_counter()

        queue_wait_ms = (
            started_at
            - task.submitted_at
        ) * 1000.0

        asr_to_translation_start_ms = (
            started_at
            - task.upstream_completed_at
        ) * 1000.0

        self.logger.info(
            "M1 translation started | "
            "start_seq=%s | end_seq=%s | "
            "queue_wait_ms=%.1f",
            task.start_sequence_number,
            task.end_sequence_number,
            queue_wait_ms,
        )

        try:
            result = self.engine.translate(
                task.text
            )

            completed_at = time.perf_counter()

            asr_to_translation_ms = (
                completed_at
                - task.upstream_completed_at
            ) * 1000.0

            segment_to_translation_ms = (
                completed_at
                - task.segment_end_timestamp
            ) * 1000.0

            with self._lock:
                self._completed_count += 1

            task_result = TranslationTaskResult(
                task=task,
                result=result,
                started_at=started_at,
                completed_at=completed_at,
                queue_wait_ms=queue_wait_ms,
                asr_to_translation_ms=(
                    asr_to_translation_ms
                ),
                segment_to_translation_ms=(
                    segment_to_translation_ms
                ),
                error=None,
            )

            self._result_queue.put(
                task_result
            )

            self.logger.info(
                "M1 translation completed | "
                "start_seq=%s | end_seq=%s | "
                "inference_ms=%.1f | "
                "queue_wait_ms=%.1f | "
                "asr_to_translation_ms=%.1f | "
                "segment_to_translation_ms=%.1f",
                task.start_sequence_number,
                task.end_sequence_number,
                result.inference_time_ms,
                queue_wait_ms,
                asr_to_translation_ms,
                segment_to_translation_ms,
            )

        except Exception as exc:
            completed_at = time.perf_counter()

            with self._lock:
                self._failed_count += 1

            error_message = (
                f"{type(exc).__name__}: {exc}"
            )

            task_result = TranslationTaskResult(
                task=task,
                result=None,
                started_at=started_at,
                completed_at=completed_at,
                queue_wait_ms=queue_wait_ms,
                asr_to_translation_ms=(
                    completed_at
                    - task.upstream_completed_at
                ) * 1000.0,
                segment_to_translation_ms=(
                    completed_at
                    - task.segment_end_timestamp
                ) * 1000.0,
                error=error_message,
            )

            self._result_queue.put(
                task_result
            )

            self.logger.error(
                "M1 translation failed | "
                "start_seq=%s | end_seq=%s | %s",
                task.start_sequence_number,
                task.end_sequence_number,
                error_message,
            )