"""Explicit task states mapped onto existing phase names (log overlay only)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskState(str, Enum):
    SCAN = "SCAN"
    LOCALIZE_DRILL = "LOCALIZE_DRILL"
    APPROACH_DRILL = "APPROACH_DRILL"
    CLOSE_HAND = "CLOSE_HAND"
    VERIFY_GRASP = "VERIFY_GRASP"
    LIFT_AND_VERIFY = "LIFT_AND_VERIFY"
    RELOCALIZE_BOARD = "RELOCALIZE_BOARD"
    PLAN_TRANSFER = "PLAN_TRANSFER"
    RETREAT_OR_REPOSITION = "RETREAT_OR_REPOSITION"
    APPROACH_BOARD = "APPROACH_BOARD"
    FEED_AND_MONITOR = "FEED_AND_MONITOR"
    PLACE_DRILL = "PLACE_DRILL"
    COMPLETE = "COMPLETE"
    ABORT_OR_SAFE_RECOVERY = "ABORT_OR_SAFE_RECOVERY"


PHASE_TO_STATE = {
    "1/8 look: top-down scan": TaskState.SCAN,
    "2/8 look: side view": TaskState.LOCALIZE_DRILL,
    "3/8 approach (sideways, thumb pre-opposed)": TaskState.APPROACH_DRILL,
    "4/8 grasp": TaskState.CLOSE_HAND,
    "5/8 locate board": TaskState.RELOCALIZE_BOARD,
    "6/8 approach by tip-to-board distance": TaskState.APPROACH_BOARD,
    "7/8 drill": TaskState.FEED_AND_MONITOR,
    "8/8 place": TaskState.PLACE_DRILL,
    "done": TaskState.COMPLETE,
}


@dataclass
class TransitionRecord:
    from_state: TaskState
    to_state: TaskState
    reason: str = ""
    ok: bool = True


@dataclass
class TaskStateMachine:
    state: TaskState = TaskState.SCAN
    history: list[TransitionRecord] = field(default_factory=list)
    fail_reason: str = ""
    retries: dict = field(default_factory=dict)

    def enter(self, nxt: TaskState, reason: str = "", ok: bool = True) -> TaskState:
        rec = TransitionRecord(self.state, nxt, reason=reason, ok=ok)
        self.history.append(rec)
        self.state = nxt
        if not ok:
            self.fail_reason = reason or self.fail_reason
        return self.state

    def abort(self, reason: str) -> TaskState:
        self.fail_reason = reason
        return self.enter(TaskState.ABORT_OR_SAFE_RECOVERY, reason=reason, ok=False)

    def sync_phase(self, phase_name: str) -> TaskState | None:
        st = PHASE_TO_STATE.get(str(phase_name))
        if st is not None and st != self.state:
            self.enter(st, reason=f"phase:{phase_name}")
            return st
        return None

    def bump_retry(self, key: str, limit: int) -> bool:
        n = int(self.retries.get(key, 0)) + 1
        self.retries[key] = n
        return n <= int(limit)

    def to_log(self) -> dict:
        return {
            "state": self.state.value,
            "fail_reason": self.fail_reason,
            "retries": dict(self.retries),
            "n_transitions": len(self.history),
            "last": (self.history[-1].to_state.value if self.history else None),
        }
