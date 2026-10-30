"""Score IK branches: continuity + wrap + rest, not just joint-limit margin.

`plan_path_ik` used to keep the chain with the largest min-limit margin. That
prefers an elbow-flip / wrist-wrap if it sits farther from the stops. Cost
below is minimized so a nearby, unwrapped branch wins when margin is adequate.
"""
from __future__ import annotations

import numpy as np

from hw.defaults import REST_Q, XARM7_HI, XARM7_LO


def limit_margin(q, lo=XARM7_LO, hi=XARM7_HI):
    """Smallest distance (rad) to a real (not ±2π) joint limit."""
    q = np.asarray(q, float).reshape(-1)
    lo = np.asarray(lo, float).reshape(-1)
    hi = np.asarray(hi, float).reshape(-1)
    lim = (hi - lo) < 6.0
    if not np.any(lim):
        return 0.0
    return float(np.min(np.minimum(q - lo, hi - q)[lim]))


def chain_path_length(chain, q_cur=None):
    qs = []
    if q_cur is not None:
        qs.append(np.asarray(q_cur, float).reshape(-1))
    qs.extend(np.asarray(c, float).reshape(-1) for c in chain)
    if len(qs) < 2:
        return 0.0
    return float(sum(np.linalg.norm(qs[i + 1] - qs[i]) for i in range(len(qs) - 1)))


def wrap_penalty(q, rest=REST_Q, lo=XARM7_LO, hi=XARM7_HI):
    """How far unlimited (±2π) joints have spun away from rest."""
    q = np.asarray(q, float).reshape(-1)
    rest = np.asarray(rest, float).reshape(-1)[: q.size]
    lo = np.asarray(lo, float).reshape(-1)[: q.size]
    hi = np.asarray(hi, float).reshape(-1)[: q.size]
    unlimited = (hi - lo) >= 6.0
    return float(np.sum(np.abs(q - rest)[unlimited]))


def branch_cost(chain, q_cur, rest=REST_Q, lo=XARM7_LO, hi=XARM7_HI, *,
                w_path=1.0, w_wrap=0.35, w_rest=0.15, w_margin=1.5):
    """Lower is more stable. `chain` is one or more 7-vectors."""
    chain = [np.asarray(c, float).reshape(-1) for c in chain]
    if not chain:
        return float("inf")
    margin = min(limit_margin(c, lo, hi) for c in chain)
    path = chain_path_length(chain, q_cur)
    q0 = chain[0]
    rest = np.asarray(rest, float).reshape(-1)[: q0.size]
    wrap = wrap_penalty(q0, rest, lo, hi)
    rest_d = float(np.linalg.norm(q0 - rest))
    return (float(w_path) * path
            + float(w_wrap) * wrap
            + float(w_rest) * rest_d
            - float(w_margin) * margin)


def rank_qs(sols, q_cur, rest=REST_Q, lo=XARM7_LO, hi=XARM7_HI):
    """Sort distinct IK solutions, most stable first."""
    return sorted(
        sols, key=lambda q: branch_cost([q], q_cur, rest=rest, lo=lo, hi=hi))


def pick_best_chain(chains, q_cur, rest=REST_Q, lo=XARM7_LO, hi=XARM7_HI,
                    min_margin=0.0):
    """Return `(chain, margin)` of the lowest-cost chain that clears `min_margin`.

    `None` if every chain is infeasible.
    """
    best = None
    for chain in chains:
        if not chain:
            continue
        m = min(limit_margin(c, lo, hi) for c in chain)
        if m < float(min_margin):
            continue
        cost = branch_cost(chain, q_cur, rest=rest, lo=lo, hi=hi)
        if best is None or cost < best[0]:
            best = (cost, chain, m)
    if best is None:
        return None
    return best[1], best[2]
