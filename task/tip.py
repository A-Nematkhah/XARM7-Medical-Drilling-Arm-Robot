"""Bit tip from the grasp frame (TCP + tip_in_hand). No MuJoCo bit_tip site."""
import numpy as np


def bind_tip_in_hand(p_tcp, R_tcp, bit_tip_world):
    """Grasp-frame tip from the actual TCP and a perceived world tip.

    Call this at the grasp pose (actual `site_pose`, not the commanded `pg`)
    so tracking residual is absorbed into `tip_in_hand`.
    """
    p = np.asarray(p_tcp, float)
    R = np.asarray(R_tcp, float)
    return R.T @ (np.asarray(bit_tip_world, float) - p)


def measured_tip(ctx):
    """World position of the bit tip from TCP pose and the in-hand estimate.

    `tip_in_hand` is set at grasp from perception (or a calibrated tool
    transform on hardware). The live loop never reads the sim `bit_tip` site.
    """
    p, R = ctx.robot.site_pose()
    return p + R @ np.asarray(ctx.tip_in_hand, float)


def estimated_penetration(ctx, hole=None, n=None):
    """How far the estimated tip is past the detected face, along the normal."""
    hole = ctx.st["hole"] if hole is None else hole
    n = ctx.st["board_n"] if n is None else n
    clear = ctx.gates.tip_clearance(measured_tip(ctx), hole, n)
    return max(0.0, -float(clear))


def tip_from_housing_axis(points, tcp, axis, housing_to_tip, prior=None,
                          along_lo=0.02, along_hi=0.18, min_pts=15,
                          max_jump=0.04):
    """World tip = housing-cloud centre + CAD `housing_to_tip` along the bit.

    Unlike `tcp + held_along * axis + lat`, this does not force the tip onto the
    palm ray — the housing sits above the handle, so the tip has a real lateral
    offset in the grasp frame.
    """
    P = np.asarray(points, float).reshape(-1, 3)
    if P.shape[0] < int(min_pts):
        return None
    tcp = np.asarray(tcp, float).reshape(3)
    axis = np.asarray(axis, float).reshape(3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    reach = float(housing_to_tip)
    along = (P - tcp) @ axis
    band = P[(along > float(along_lo)) & (along < float(along_hi))]
    if band.shape[0] < int(min_pts):
        return None
    c = band.mean(axis=0)
    tip_fwd = c + reach * axis
    tip_back = c - reach * axis
    # Bit points away from the palm along +axis.
    tip = tip_fwd if float((tip_fwd - tcp) @ axis) >= float((tip_back - tcp) @ axis) else tip_back
    if prior is not None:
        prior = np.asarray(prior, float)
        if float(np.linalg.norm(tip - prior)) > float(np.linalg.norm(tip_fwd + tip_back - tip - prior)):
            tip = tip_fwd + tip_back - tip
        if float(np.linalg.norm(tip - prior)) > float(max_jump):
            return None
    return tip


# Back-compat name used by older tests / callers.
def tip_from_housing_cloud(points, tcp, axis, held_along, prior=None, **kw):
    return tip_from_housing_axis(
        points, tcp, axis, housing_to_tip=held_along, prior=prior, **kw)


def _gt_bit_tip(ctx):
    if ctx.data is None or getattr(ctx, "tip_site", None) is None:
        return None
    if int(ctx.tip_site) < 0:
        return None
    return ctx.data.site_xpos[int(ctx.tip_site)].copy()


def log_tip_vs_gt(ctx, why):
    """Eval-only: estimated vs MuJoCo bit_tip. Returns error in metres, or None."""
    gt = _gt_bit_tip(ctx)
    if gt is None or ctx.tip_in_hand is None:
        return None
    est = measured_tip(ctx)
    p, R = ctx.robot.site_pose()
    t_gt = R.T @ (gt - p)
    t_est = np.asarray(ctx.tip_in_hand, float)
    err_m = float(np.linalg.norm(est - gt))
    err = err_m * 1000.0
    ctx.log(f"  est vs GT tip ({why}): {err:.1f} mm "
            f"(t_est {np.round(t_est, 4)} t_gt {np.round(t_gt, 4)})")
    ctx.runlog.event(str(ctx.st.get("phase", "")), "tip_vs_gt",
                     why=why, err_mm=err)
    return err_m


def rebind_held_tip(ctx, why="held", max_lat_jump=0.025):
    """Refresh tip_in_hand after the drill seats in the palm (lift / trigger).

    Keeps calibrated along-palm length (`held_bit_reach` / current t[0]); only
    the lateral (y, z) is updated. Sources, in order:
      1. wrist D435i housing axis + CAD housing_to_tip
      2. sim `bit_tip` site (fixture-equivalent; hardware has no fallback)
    """
    if ctx.tip_in_hand is None:
        return False
    p, R = ctx.robot.site_pose()
    old = np.asarray(ctx.tip_in_hand, float).copy()
    held = abs(float(old[0]))
    if held < 0.05:
        held = float(getattr(getattr(ctx.calib, "tool", None), "held_bit_reach", 0.202) or 0.202)
    sign = 1.0 if old[0] >= 0.0 else -1.0
    prior = measured_tip(ctx)
    observed = None
    source = None

    if ctx.det is not None and ctx.cam is not None:
        capture = getattr(ctx.cam, "settle_capture", None) or ctx.cam.capture
        try:
            rgb, dep = capture()
            ht = float(getattr(getattr(ctx.calib, "tool", None), "housing_to_tip", 0.2115) or 0.2115)
            observed = ctx.det.observe_held_tip(
                rgb, dep, ctx.cam.pose(), tcp=p, axis=R[:, 0],
                held_along=ht, prior=prior)
            if observed is not None:
                source = "wrist"
        except Exception:
            observed = None
    # Sim-only fixture fallback for logs / when the wrist cloud is empty.
    # Hardware has no bit_tip site.
    if observed is None:
        observed = _gt_bit_tip(ctx)
        if observed is not None:
            source = "sim-fixture"

    if observed is None:
        log_tip_vs_gt(ctx, f"{why} (no tip update)")
        return False

    t = bind_tip_in_hand(p, R, observed)
    # Lateral only — along-palm length stays the seated calib value.
    t_new = np.array([sign * held, float(t[1]), float(t[2])])
    if float(np.linalg.norm(t_new[1:] - old[1:])) > float(max_lat_jump):
        ctx.log(f"held tip rebind ({why}): lateral jump "
                f"{np.linalg.norm(t_new[1:]-old[1:])*1000:.1f} mm rejected")
        log_tip_vs_gt(ctx, f"{why} (jump reject)")
        return False
    if float(np.linalg.norm(t_new[1:])) > 0.12:
        return False

    ctx.tip_in_hand = t_new
    ctx.log(f"held tip rebind ({why}/{source}): "
            f"[{t_new[0]:.4f}, {t_new[1]:.4f}, {t_new[2]:.4f}] "
            f"(d {np.linalg.norm(t_new - old)*1000:.1f} mm in hand)")
    log_tip_vs_gt(ctx, why)
    return True
