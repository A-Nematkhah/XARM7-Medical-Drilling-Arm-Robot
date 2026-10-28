"""Structured JSONL log for each task phase (sim vs hardware comparison)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (float, int, str, bool)) or v is None:
        return v
    return str(v)


class RunLog:
    def __init__(self, path, sim_t=lambda: 0.0, run_meta=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sim_t = sim_t
        self._fh = self.path.open("w", encoding="utf-8")
        self.event("run", "start", **(run_meta or {}),
                   utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    def event(self, phase, event, **fields):
        rec = {"t": round(float(self.sim_t()), 3), "phase": phase, "event": event}
        rec.update({k: _jsonable(v) for k, v in fields.items()})
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self, **fields):
        if self._fh is None:
            return
        self.event("run", "end", **fields)
        self._fh.close()
        self._fh = None
