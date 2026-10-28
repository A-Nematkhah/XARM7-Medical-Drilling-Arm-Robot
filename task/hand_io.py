"""RH56 finger readout used by the live loop (no MuJoCo trigger qpos)."""
import numpy as np

from task.constants import INDEX_TRIGGER_OFF, INDEX_TRIGGER_ON

INDEX = 3  # little, ring, middle, index, thumb_bend, thumb_rot
# Sim trigger slide is 0..5 mm; treat >1 mm as physically engaged.
TRIGGER_TRAVEL_ON = 0.001


def scale_finger_forces(raw, scale=1.0):
    """Map a hand force register into gates.finger_load units (N*m in sim)."""
    return np.asarray(raw, float) * float(scale)


def trigger_physically_engaged(ctx, travel_on=TRIGGER_TRAVEL_ON):
    """True when the physical trigger has moved (sim) or index is the switch (real)."""
    model, data, adr = ctx.model, ctx.data, getattr(ctx, "trig_q", None)
    if model is None or data is None or adr is None or int(adr) < 0:
        return True
    return float(data.qpos[int(adr)]) > float(travel_on)


def index_angle(hand):
    """Index finger 0..1000 (0 = closed / trigger pressed)."""
    try:
        ang = hand.get_angles()
    except Exception:
        ang = getattr(hand, "target", None)
    if ang is None:
        return 1000
    return int(ang[INDEX])


def trigger_pressed(hand, on=INDEX_TRIGGER_ON):
    return index_angle(hand) <= int(on)


def trigger_released(hand, off=INDEX_TRIGGER_OFF):
    return index_angle(hand) >= int(off)
