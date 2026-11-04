"""M2 ASR window queue backpressure primitives (Phase 1).

Phase 1 only defines what a full/near-full window queue looks like and
the vocabulary for why a window might later be dropped. It does NOT
implement a window queue, a dispatcher, or any coalescing/staleness
decision logic - those require the adaptive buffer (Phase 2) and
incremental Whisper (Phase 3) work this module is preparing for.

Why this is not just "increase asr.queue_size":
M1's `asr.worker.ASRWorker` submits at most one `SpeechSegment` per
utterance, so a full queue means "ASR fell behind" and dropping the
newest segment (tracked via `ASRWorker.dropped_count`) is an
acceptable, rare fallback - the existing bounded `queue.Queue` and
`put_nowait`/`Full` handling in `asr/worker.py` already covers this
correctly and is left unchanged.

M2's incremental decoding is expected to submit many
`streaming.streaming_state.ProcessingWindow`s per utterance. A full
queue there more often means several windows for the *same* utterance
are queued and a newer one already makes the older ones obsolete.
Simply enlarging the queue (e.g. to 50) would let stale windows pile
up and be processed in submission order, adding latency without
adding value - the newest audio/context would sit behind work that no
longer matters. `QueueBackpressureStatus` lets a future dispatcher
detect "the queue is full" as a distinct, structured condition from
"per-utterance staleness", which `WindowDropReason` gives a name to.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WindowDropReason(str, Enum):
    """Why a pending ProcessingWindow was not (or should not be) processed.

    Not yet raised or consumed by any queue/dispatcher in Phase 1;
    this enum only fixes the vocabulary a future ASR window dispatcher
    will log, matching CLAUDE.md's requirement that every M2 decision
    be logged in structured form.
    """

    # The bounded window queue had no free capacity.
    QUEUE_FULL = "QUEUE_FULL"

    # A newer ProcessingWindow for the same utterance made this one
    # obsolete before it was processed.
    SUPERSEDED_BY_NEWER_WINDOW = "SUPERSEDED_BY_NEWER_WINDOW"

    # The window waited longer than a future policy's staleness bound.
    STALE_TIMEOUT = "STALE_TIMEOUT"


@dataclass(frozen=True)
class QueueBackpressureStatus:
    """Point-in-time bounded-queue occupancy for M2 window dispatch.

    A plain, dependency-free snapshot so a future window dispatcher
    can report/log backpressure without being coupled to any specific
    queue implementation (`queue.Queue`, a list, etc.).

    Attributes:
        capacity: Maximum number of ProcessingWindows the queue may
            hold at once.
        current_size: Number of windows currently queued/in-flight.
        is_full: Whether `current_size >= capacity`.
    """

    capacity: int
    current_size: int
    is_full: bool

    @classmethod
    def evaluate(
        cls,
        capacity: int,
        current_size: int,
    ) -> "QueueBackpressureStatus":
        """Compute backpressure status for a bounded window queue.

        Args:
            capacity: Maximum queue capacity.
            current_size: Current number of queued/in-flight windows.

        Returns:
            The computed QueueBackpressureStatus.

        Raises:
            ValueError: If capacity is not positive or current_size is
                negative.
        """

        if capacity <= 0:
            raise ValueError("capacity must be greater than zero.")

        if current_size < 0:
            raise ValueError("current_size cannot be negative.")

        return cls(
            capacity=capacity,
            current_size=current_size,
            is_full=current_size >= capacity,
        )

    @property
    def has_capacity(self) -> bool:
        """Return whether at least one more window could be enqueued."""

        return not self.is_full
