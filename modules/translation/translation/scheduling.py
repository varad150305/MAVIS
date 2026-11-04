"""Downstream MT invocation scheduling: policy interface + driver.

A policy decides, for each stable source update from M2, whether to
invoke MT now. The driver applies the decision through the MT-2
`IncrementalTranslator` (frozen MT-1 engine, full-prefix re-translation)
and logs every decision with its reason.

Implemented policies:

  EagerPolicy           MT3-A - translate whenever the stable source gained
                        new words (baseline)
  FixedThresholdPolicy  MT3-B - translate once >= K new stable source words
                        have accumulated since the last MT call (baseline)
  AdaptiveDeadlinePolicy MT3-C - stable-debt threshold raised by an
                        arrival-rate x estimated-MT-cost term, bounded by a
                        hard deadline (frozen spec:
                        evaluation/mt3c_prep/mt3c_specification.md)

Finalization is policy-independent: `StreamingMTScheduler.finalize()`
always ensures the final stable source is translated (one call only if
it has untranslated words) and logs that as its own decision record.

Deadline timers: a policy may ask for a timer after a WAIT (decision
details `arm_timer_at`). Timers fire as their own 'deadline' events on the
stream time axis, independently of ASR updates: before an update at time t
every timer < t fires; before finalization every timer <= the finalization
time fires (same-time order UPDATE, DEADLINE, FINALIZE). MT3-A/B never arm
timers, so their behaviour is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from translation.incremental import FinalizeResult, IncrementalTranslator, MTInvocation


@dataclass(frozen=True)
class SourceUpdate:
    """One stable-source update from M2 (full current stable prefix)."""

    sample_id: str
    update_id: int
    stable_source_text: str
    utterance_id: int | None = None
    source_available_time: float | None = None


@dataclass(frozen=True)
class SchedulerState:
    """Observations available to a policy at an update (read-only)."""

    stable_source_words: int
    words_already_translated: int
    new_stable_words: int
    translation_debt: int
    source_available_time: float | None
    invocations_so_far: int
    # Event-time observations (None when the caller supplies no times).
    now: float | None = None                       # stream time of this event
    delta_words: int | None = None                 # new stable words in this update (0 for timer/finalization)
    t_pending: float | None = None                 # availability time of the oldest untranslated stable word
    oldest_pending_age_s: float | None = None      # now - t_pending
    rate_origin: float | None = None               # first update's VAD onset (rate-window start bound)
    arrivals: tuple[tuple[float, int], ...] = ()   # causal (update time, new words) history so far
    is_finalization: bool = False


@dataclass(frozen=True)
class PolicyDecision:
    translate: bool
    reason: str
    details: dict[str, Any] | None = None

    @property
    def label(self) -> str:
        return "MT" if self.translate else "WAIT"


class SchedulingPolicy(Protocol):
    name: str

    def params(self) -> dict[str, Any]: ...

    def decide(self, state: SchedulerState) -> PolicyDecision: ...


NO_NEW_WORDS = "no new stable source words (duplicate/empty update)"


class EagerPolicy:
    """MT3-A: every update with new stable words triggers MT. No K, no
    deadline, no load or debt gate; debt is observed only."""

    name = "MT3-A eager"

    def params(self) -> dict[str, Any]:
        return {}

    def decide(self, state: SchedulerState) -> PolicyDecision:
        if state.new_stable_words > 0:
            return PolicyDecision(True, "new stable source words")
        return PolicyDecision(False, NO_NEW_WORDS)


class FixedThresholdPolicy:
    """MT3-B: translate the full stable prefix once the stable source has
    grown by >= K words since the last MT call.

    `state.new_stable_words` is exactly the accumulated new stable words
    (current stable words - last translated source words), reset by every
    MT call. No deadline, no time/CPU/stability signal.
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("K must be >= 1")
        self.k = k
        self.name = f"MT3-B fixed threshold (K={k})"

    def params(self) -> dict[str, Any]:
        return {"K": self.k}

    def decide(self, state: SchedulerState) -> PolicyDecision:
        if state.new_stable_words <= 0:
            return PolicyDecision(False, NO_NEW_WORDS)
        if state.new_stable_words >= self.k:
            return PolicyDecision(True, "threshold_reached")
        return PolicyDecision(False, "insufficient_new_words")


class AdaptiveDeadlinePolicy:
    """MT3-C (frozen specification; parameters from evaluation/mt3c_parameters.json).

        D          = stable words - source position of the latest scheduled MT call
        A          = now - t_pending (oldest untranslated stable word)
        R          = words arrived in (now - W_R, now] / (now - max(now - W_R, rate_origin))
        C_hat(W)   = a + b*W seconds (frozen AC dev-set fit)
        K_adaptive = min(max(K_base, ceil(R * C_hat(W) / rho)), K_max)

    Priority: D = 0 -> WAIT; A >= T_max -> MT (deadline); D >= K_adaptive -> MT
    (threshold_adaptive); else WAIT and arm a timer at t_pending + T_max.
    Finalization (handled by the scheduler): one call iff D > 0.

    Inputs are the stable-source event history only. Measured MT runtime,
    MT worker busy state, OS CPU state and target-side output are never read.
    """

    REASON_FINAL = "finalization"
    REASON_FINAL_NOOP = "finalization_already_translated"
    REASON_NO_DEBT = "no_untranslated_source"
    REASON_DEADLINE = "deadline"
    REASON_THRESHOLD = "threshold_adaptive"
    REASON_BELOW = "below_adaptive_threshold"

    def __init__(self, *, k_base: int, k_max: int, t_max_s: float, w_r_s: float, rho: float,
                 cost_a_s: float, cost_b_s_per_word: float, compute_term: bool = True,
                 parameters_sha256: str | None = None, variant: str = "primary") -> None:
        if not 1 <= k_base <= k_max:
            raise ValueError("require 1 <= K_base <= K_max")
        if t_max_s <= 0 or w_r_s <= 0 or rho <= 0:
            raise ValueError("T_max, W_R and rho must be > 0")
        self.k_base, self.k_max, self.t_max_s, self.w_r_s, self.rho = k_base, k_max, t_max_s, w_r_s, rho
        self.cost_a_s, self.cost_b_s_per_word, self.compute_term = cost_a_s, cost_b_s_per_word, compute_term
        self.parameters_sha256, self.variant = parameters_sha256, variant
        self.name = f"MT3-C adaptive+deadline ({variant})"

    @classmethod
    def from_frozen(cls, params: dict[str, Any], parameters_sha256: str, *, t_max_s: float | None = None,
                    compute_term: bool = True, variant: str = "primary") -> "AdaptiveDeadlinePolicy":
        p = params["parameters"]
        return cls(k_base=p["K_base"], k_max=p["K_max"], t_max_s=p["T_max_s"] if t_max_s is None else t_max_s,
                   w_r_s=p["W_R_s"], rho=p["rho"], cost_a_s=p["cost_model"]["a_s"],
                   cost_b_s_per_word=p["cost_model"]["b_s_per_word"], compute_term=compute_term,
                   parameters_sha256=parameters_sha256, variant=variant)

    def params(self) -> dict[str, Any]:
        return {"K_base": self.k_base, "K_max": self.k_max,
                "T_max_s": None if math.isinf(self.t_max_s) else self.t_max_s,
                "W_R_s": self.w_r_s, "rho": self.rho, "cost_a_s": self.cost_a_s,
                "cost_b_s_per_word": self.cost_b_s_per_word, "compute_term": self.compute_term,
                "variant": self.variant, "parameters_sha256": self.parameters_sha256}

    def cost_estimate_s(self, words: int) -> float:
        return self.cost_a_s + self.cost_b_s_per_word * words

    def signals(self, state: SchedulerState) -> dict[str, Any]:
        t = state.now
        if t is None or state.rate_origin is None:
            raise ValueError("MT3-C needs event times and the rate origin (first update's VAD onset)")
        lo = max(t - self.w_r_s, state.rate_origin)
        span = t - lo
        if span <= 0:
            raise ValueError(f"non-positive arrival-rate window at t={t}")
        window_words = sum(dw for tu, dw in state.arrivals if t - self.w_r_s < tu <= t)
        rate = window_words / span
        c_hat = self.cost_estimate_s(state.stable_source_words)
        compute_threshold = rate * c_hat / self.rho
        k_adaptive = (min(max(self.k_base, math.ceil(compute_threshold)), self.k_max)
                      if self.compute_term else self.k_base)
        debt = state.translation_debt
        deadline_at = (state.t_pending + self.t_max_s) if state.t_pending is not None else None
        return {
            "arrival_rate_wps": rate, "rate_window_start_s": lo, "rate_window_words": window_words,
            "est_mt_cost_s": c_hat, "compute_threshold": compute_threshold, "rho": self.rho,
            "k_base": self.k_base, "k_adaptive": k_adaptive, "k_max": self.k_max,
            "compute_raised": k_adaptive > self.k_base, "t_max_s": None if math.isinf(self.t_max_s) else self.t_max_s,
            "deadline_at_s": deadline_at,
            "deadline_met": deadline_at is not None and t >= deadline_at,
            "threshold_met": debt > 0 and debt >= k_adaptive,
            "arm_timer_at": None,
        }

    def decide(self, state: SchedulerState) -> PolicyDecision:
        s = self.signals(state)
        if state.translation_debt == 0:
            return PolicyDecision(False, self.REASON_NO_DEBT, s)
        if s["deadline_met"]:
            return PolicyDecision(True, self.REASON_DEADLINE, s)
        if s["threshold_met"]:
            return PolicyDecision(True, self.REASON_THRESHOLD, s)
        if s["deadline_at_s"] is not None and math.isfinite(s["deadline_at_s"]):
            s["arm_timer_at"] = s["deadline_at_s"]
        return PolicyDecision(False, self.REASON_BELOW, s)

    def finalization_decision(self, state: SchedulerState, invoked_mt: bool) -> PolicyDecision:
        return PolicyDecision(invoked_mt, self.REASON_FINAL if invoked_mt else self.REASON_FINAL_NOOP,
                              self.signals(state))


@dataclass
class DecisionRecord:
    sample_id: str
    update_id: int | None
    state: SchedulerState
    decision: PolicyDecision
    invocation_id: int | None
    kind: str = "update"  # "update" | "deadline" | "finalization"
    policy_params: dict[str, Any] = field(default_factory=dict)
    utterance_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["state"].pop("arrivals", None)  # full history is reconstructible; keep records compact
        out["decision"]["label"] = self.decision.label
        return out


@dataclass
class _SampleLog:
    sample_id: str
    decisions: list[DecisionRecord] = field(default_factory=list)
    last_update_words: list[str] = field(default_factory=list)
    word_times: list[float | None] = field(default_factory=list)  # availability time of each stable word
    arrivals: list[tuple[float, int]] = field(default_factory=list)
    rate_origin: float | None = None
    timer: float | None = None
    last_utterance_id: int | None = None
    last_source_time: float | None = None


class StreamingMTScheduler:
    """Feeds one sample's stable-source updates through a policy."""

    def __init__(self, translator: IncrementalTranslator, policy: SchedulingPolicy) -> None:
        self.translator = translator
        self.policy = policy
        self._log: _SampleLog | None = None

    def start(self, sample_id: str, *, rate_origin: float | None = None) -> None:
        """`rate_origin`: VAD onset of the sample's first update (needed by MT3-C only)."""

        self.translator.reset(sample_id)
        self._log = _SampleLog(sample_id, rate_origin=rate_origin)

    @property
    def decisions(self) -> list[DecisionRecord]:
        return list(self._log.decisions) if self._log else []

    @property
    def pending_timer(self) -> float | None:
        return self._log.timer if self._log else None

    def on_update(self, update: SourceUpdate) -> MTInvocation | None:
        if self._log is None or update.sample_id != self._log.sample_id:
            raise RuntimeError("start(sample_id) must be called for this sample first")

        t = update.source_available_time
        if t is not None:
            self._fire_timers(t, inclusive=False)  # UPDATE precedes DEADLINE at equal times

        words = update.stable_source_text.split()
        previous = self._log.last_update_words
        if words[: len(previous)] != previous:
            raise ValueError(f"non-monotonic stable source in {update.sample_id!r} at update {update.update_id}")
        delta = len(words) - len(previous)
        self._log.last_update_words = words
        self._log.word_times += [t] * delta
        if delta and t is not None:
            self._log.arrivals.append((t, delta))
        self._log.last_utterance_id = update.utterance_id
        self._log.last_source_time = t

        state = self._state(len(words), t, now=t, delta=delta)
        decision = self.policy.decide(state)

        invocation = None
        if decision.translate:
            invocation = self.translator.update(
                update.stable_source_text,
                utterance_id=update.utterance_id,
                source_available_time=t,
                trigger_time=t,
            )
        self._after_decision(decision)
        self._log.decisions.append(DecisionRecord(
            update.sample_id, update.update_id, state, decision,
            invocation.invocation_id if invocation else None,
            kind="update", policy_params=self.policy.params(), utterance_id=update.utterance_id,
        ))
        return invocation

    def _after_decision(self, decision: PolicyDecision) -> None:
        if decision.translate:
            self._log.timer = None  # the pending source is now scheduled
        elif decision.details and decision.details.get("arm_timer_at") is not None:
            self._log.timer = decision.details["arm_timer_at"]

    def _fire_timers(self, until: float, *, inclusive: bool) -> None:
        """Deadline events, independent of ASR updates."""

        log = self._log
        while log.timer is not None and (log.timer <= until if inclusive else log.timer < until):
            t = log.timer
            log.timer = None
            state = self._state(len(log.last_update_words), log.last_source_time, now=t, delta=0)
            decision = self.policy.decide(state)
            invocation = None
            if decision.translate:
                invocation = self.translator.update(
                    " ".join(log.last_update_words),
                    utterance_id=log.last_utterance_id,
                    source_available_time=log.last_source_time,
                    trigger_time=t,
                )
            self._after_decision(decision)
            log.decisions.append(DecisionRecord(
                log.sample_id, None, state, decision, invocation.invocation_id if invocation else None,
                kind="deadline", policy_params=self.policy.params(),
            ))

    def _state(self, stable_words: int, source_available_time: float | None, *, now: float | None = None,
               delta: int | None = None, is_finalization: bool = False) -> SchedulerState:
        invocations = self.translator.invocations
        translated = invocations[-1].source_word_count if invocations else 0
        log = self._log
        t_pending = log.word_times[translated] if stable_words > translated and len(log.word_times) > translated else None
        return SchedulerState(
            stable_source_words=stable_words,
            words_already_translated=translated,
            new_stable_words=stable_words - translated,
            translation_debt=stable_words - translated,
            source_available_time=source_available_time,
            invocations_so_far=len(invocations),
            now=now,
            delta_words=delta,
            t_pending=t_pending,
            oldest_pending_age_s=(now - t_pending) if now is not None and t_pending is not None else None,
            rate_origin=log.rate_origin,
            arrivals=tuple(log.arrivals),
            is_finalization=is_finalization,
        )

    def finalize(self, final_source_text: str, *, source_available_time: float | None = None,
                 finalization_time: float | None = None) -> FinalizeResult:
        """Ensure the final stable source is translated (policy-independent);
        no duplicate call if it was already translated. Logged as a
        'finalization' decision record.

        `finalization_time` (default: `source_available_time`) is the stream
        time at which the end of the sample is known; timers due by then fire
        first, any later timer is cancelled."""

        if self._log is None:
            raise RuntimeError("start(sample_id) must be called first")
        t_fin = source_available_time if finalization_time is None else finalization_time
        if t_fin is not None:
            self._fire_timers(t_fin, inclusive=True)  # DEADLINE precedes FINALIZE at equal times
        self._log.timer = None
        state = self._state(len(final_source_text.split()), source_available_time, now=t_fin, delta=0,
                            is_finalization=True)
        result = self.translator.finalize(final_source_text, source_available_time=source_available_time,
                                          trigger_time=t_fin)
        final_decision = getattr(self.policy, "finalization_decision", None)
        if final_decision is not None:
            decision = final_decision(state, result.invoked_mt)
        else:
            decision = PolicyDecision(result.invoked_mt, "finalization" if result.invoked_mt
                                      else "finalization: final source already translated (no extra call)")
        self._log.decisions.append(DecisionRecord(
            self._log.sample_id, None, state, decision,
            result.invocation.invocation_id if result.invocation else None,
            kind="finalization", policy_params=self.policy.params(),
        ))
        return result
