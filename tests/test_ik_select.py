import numpy as np
import pytest

from hw.defaults import REST_Q, XARM7_HI, XARM7_LO
from hw.ik_select import (
    branch_cost, chain_path_length, limit_margin, pick_best_chain, rank_qs, wrap_penalty,
)


def test_limit_margin_matches_real_limits():
    assert limit_margin(REST_Q) > 0.5
    near = REST_Q.copy()
    near[1] = XARM7_LO[1] + 0.05
    assert limit_margin(near) == pytest.approx(0.05, abs=1e-6)


def test_wrap_penalty_grows_when_wrist_spins():
    q = REST_Q.copy()
    q[6] = REST_Q[6] + 3.0  # joint7 unlimited
    assert wrap_penalty(q) > wrap_penalty(REST_Q) + 2.9


def test_path_length_includes_start_jump():
    q0 = REST_Q
    q1 = REST_Q + 0.5
    assert chain_path_length([q0, q1], q_cur=REST_Q) == pytest.approx(0.5 * np.sqrt(7))


def test_pick_prefers_nearby_unwrapped_over_max_margin():
    cur = REST_Q.copy()
    near = [REST_Q.copy(), REST_Q.copy() + 0.05]
    far = [REST_Q.copy() + 2.5, REST_Q.copy() + 2.6]
    # Push the far chain off limited joints so its min-margin looks better.
    far[0][1] = 0.0
    far[1][1] = 0.0
    far[0][6] = 4.0
    far[1][6] = 4.0
    picked, margin = pick_best_chain([far, near], cur, min_margin=0.15)
    assert np.allclose(picked[0], near[0])
    assert margin > 0.15
    assert branch_cost(near, cur) < branch_cost(far, cur)


def test_rank_qs_puts_rest_ahead_of_wrapped():
    wrapped = REST_Q.copy()
    wrapped[4] = 5.0
    ordered = rank_qs([wrapped, REST_Q], REST_Q)
    assert np.allclose(ordered[0], REST_Q)


def test_min_margin_rejects_all():
    cur = REST_Q.copy()
    tight = REST_Q.copy()
    tight[1] = XARM7_LO[1] + 0.02
    assert pick_best_chain([[tight]], cur, min_margin=0.2) is None
