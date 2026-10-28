"""Shared runtime bag passed between task phases."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np


@dataclass
class TaskCtx:
    args: object
    calib: object
    model: object
    data: object
    robot: object
    hand: object
    cam: object
    views: object
    det: object
    board_det: object
    gates: object
    st: dict
    metrics: dict
    runlog: object
    here: str
    board_n_hint: np.ndarray
    hole_hint: np.ndarray
    allow_angled: bool
    sim_t: object
    xy: np.ndarray
    yaw: float
    trig_q: int = -1
    tip_site: int = -1
    # filled by phases
    gt_grip: np.ndarray | None = None
    est: object = None
    tip_in_hand: np.ndarray | None = None
    pg: np.ndarray | None = None
    Rg: np.ndarray | None = None
    a: np.ndarray | None = None
    p_slide: np.ndarray | None = None
    board: object = None
    board_n: np.ndarray | None = None
    hole: np.ndarray | None = None
    d: np.ndarray | None = None
    Rd: np.ndarray | None = None
    p_aim: np.ndarray | None = None
    feed: float = 0.0
    extras: SimpleNamespace = field(default_factory=SimpleNamespace)

    def log(self, msg):
        print(f"[{self.sim_t():6.2f}s] {msg}", flush=True)

    def phase(self, name, **fields):
        self.st["phase"] = name
        self.log(f"== {name}")
        self.runlog.event(name, "start", **fields)

    def phase_end(self, name, **fields):
        self.runlog.event(name, "end", **fields)

    def show_detection(self, img, secs=2.5):
        self.st["overlay"] = img
        self.st["overlay_until"] = self.sim_t() + secs
