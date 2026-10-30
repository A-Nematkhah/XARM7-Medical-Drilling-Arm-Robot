"""Per-phase TCP / joint speed profiles.

Phases pick a named profile instead of hard-coded T / vmax. Numbers live in
`config/motion.yaml` so hardware can retune without editing task code.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION = ROOT / "config" / "motion.yaml"

# Built-in defaults if YAML is missing.
_DEFAULTS = {
    "scan": dict(vmax=0.12, t_min=1.5, settle=0.20, joint_T=2.5),
    "grasp": dict(vmax=0.07, t_min=0.6, settle=0.15, joint_T=2.5,
                  joint_T_first=3.0, joint_T_step=0.8),
    "lift": dict(vmax=0.10, t_min=1.2, settle=0.20),
    "relocate": dict(vmax=0.08, t_min=1.2, settle=0.20, joint_T=2.5),
    "approach": dict(vmax=0.035, t_min=0.35, settle=0.08),
    "feed": dict(vmax=0.012, t_min=0.22, settle=0.03,
                  kp=0.55, ki=0.10, vmin=0.0025, max_step=0.004,
                  f_nom=3.5, f_hi=8.0, rpm_nom=1100.0, rpm_lo=400.0,
                  load_scale_min=0.25),
    "retract": dict(vmax=0.06, t_min=0.8, settle=0.10),
    "place": dict(vmax=0.08, t_min=0.8, settle=0.15, joint_T=2.5,
                  joint_T_first=3.0, joint_T_step=0.8),
}


@dataclass(frozen=True)
class MotionProfile:
    name: str
    vmax: float
    t_min: float
    settle: float = 0.15
    joint_T: float = 2.5
    joint_T_first: float = 3.0
    joint_T_step: float = 0.8
    # Depth-loop gains (used by feed only; others ignore them).
    kp: float = 0.0
    ki: float = 0.0
    vmin: float = 0.0
    max_step: float = 0.004
    # Load-aware feed derate (N / chuck rpm). 0 disables that axis.
    f_nom: float = 0.0
    f_hi: float = 0.0
    rpm_nom: float = 0.0
    rpm_lo: float = 0.0
    load_scale_min: float = 0.25

    def duration(self, dist_m: float) -> float:
        """Wall-clock lower bound for a Cartesian hop of length `dist_m`."""
        return max(float(self.t_min), abs(float(dist_m)) / max(float(self.vmax), 1e-6))

    def cart_kwargs(self, dist_m: float | None = None) -> dict:
        """Args for `ArmDriver.move_to_pose` / `move_along`."""
        T = self.t_min if dist_m is None else self.duration(dist_m)
        return dict(T=float(T), vmax=float(self.vmax), settle=float(self.settle))


class MotionBook:
    """Named profiles. `book["approach"]` → MotionProfile."""

    def __init__(self, profiles: dict[str, MotionProfile], source: str = "<defaults>"):
        self._profiles = dict(profiles)
        self.source = source

    def __getitem__(self, name: str) -> MotionProfile:
        if name not in self._profiles:
            raise KeyError(f"unknown motion profile {name!r}; have {sorted(self._profiles)}")
        return self._profiles[name]

    def get(self, name: str, default: str | None = None) -> MotionProfile:
        if name in self._profiles:
            return self._profiles[name]
        if default is not None:
            return self[default]
        raise KeyError(name)

    def names(self):
        return sorted(self._profiles)


def _profile_from_dict(name: str, raw: dict) -> MotionProfile:
    base = dict(_DEFAULTS.get(name, dict(vmax=0.08, t_min=1.0, settle=0.15)))
    keys = ("vmax", "t_min", "settle", "joint_T", "joint_T_first", "joint_T_step",
            "kp", "ki", "vmin", "max_step",
            "f_nom", "f_hi", "rpm_nom", "rpm_lo", "load_scale_min")
    base.update({k: raw[k] for k in raw if k in keys})
    return MotionProfile(
        name=name,
        vmax=float(base["vmax"]),
        t_min=float(base["t_min"]),
        settle=float(base.get("settle", 0.15)),
        joint_T=float(base.get("joint_T", 2.5)),
        joint_T_first=float(base.get("joint_T_first", 3.0)),
        joint_T_step=float(base.get("joint_T_step", 0.8)),
        kp=float(base.get("kp", 0.0)),
        ki=float(base.get("ki", 0.0)),
        vmin=float(base.get("vmin", 0.0)),
        max_step=float(base.get("max_step", 0.004)),
        f_nom=float(base.get("f_nom", 0.0)),
        f_hi=float(base.get("f_hi", 0.0)),
        rpm_nom=float(base.get("rpm_nom", 0.0)),
        rpm_lo=float(base.get("rpm_lo", 0.0)),
        load_scale_min=float(base.get("load_scale_min", 0.25)),
    )


def load_motion_profiles(path: str | Path | None = None, *,
                         feed_vmax_cap: float | None = None) -> MotionBook:
    """Load YAML profiles; missing file → built-in defaults.

    `feed_vmax_cap` (usually gates.MAX_FEED_SPEED) clamps the feed profile.
    """
    p = Path(path) if path is not None else DEFAULT_MOTION
    raw = {}
    source = "<defaults>"
    if p.is_file() and yaml is not None:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        source = str(p)
    names = sorted(set(_DEFAULTS) | set(raw))
    profiles = {}
    for name in names:
        block = raw.get(name) if isinstance(raw.get(name), dict) else {}
        profiles[name] = _profile_from_dict(name, block or {})
    if feed_vmax_cap is not None and "feed" in profiles:
        feed = profiles["feed"]
        cap = float(feed_vmax_cap)
        if feed.vmax > cap + 1e-12:
            profiles["feed"] = replace(feed, vmax=cap)
    return MotionBook(profiles, source=source)
