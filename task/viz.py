"""Unified sim+sensor window, optional legacy previews, and mp4 recorder."""
import time

import cv2
import mujoco
import numpy as np

from task.constants import (
    SPAWN_DRILL_HI, SPAWN_DRILL_LO, SPAWN_FACE_EXTENT_XY, SPAWN_FACE_XY,
)


def render_lidar_image(scan, cmap=None, size=(640, 400)):
    """BEV (left) + azimuth×elevation range image (right) for a 360° scan."""
    w, h = int(size[0]), int(size[1])
    img = np.zeros((h, w, 3), np.uint8)
    bev_w = min(h, w * 3 // 5)
    pan_w = max(1, w - bev_w)
    origin = np.asarray(scan.origin, float).reshape(3)
    pts = np.asarray(scan.points, float).reshape(-1, 3)
    ok = np.isfinite(pts).all(axis=1)
    img[:, :bev_w] = _bev_panel(pts[ok], origin, cmap, bev_w, h)
    img[:, bev_w:] = _range_panel(scan, pan_w, h)
    cv2.line(img, (bev_w, 0), (bev_w, h), (80, 80, 80), 1)
    cv2.putText(img, "BEV", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, "range", (bev_w + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    n = int(ok.sum())
    cv2.putText(img, f"{n} hits", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    return img


def _height_bgr(z, lo=0.0, hi=0.45):
    t = float(np.clip((z - lo) / (hi - lo + 1e-9), 0.0, 1.0))
    # blue (low) -> cyan -> yellow -> red (high)
    if t < 0.5:
        u = t * 2.0
        return (int(255 * (1.0 - u)), int(180 + 75 * u), int(40 + 80 * u))
    u = (t - 0.5) * 2.0
    return (int(40 * (1.0 - u)), int(255 * (1.0 - 0.4 * u)), int(255))


def _bev_panel(P, origin, cmap, w, h):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (18, 18, 22)
    span = 1.6
    def xy_px(xy):
        u = int(round((xy[0] / span + 0.5) * (w - 1)))
        v = int(round((0.5 - xy[1] / span) * (h - 1)))
        return u, v
    # table workspace box (approx) + drill spawn patch (left of the board)
    cv2.rectangle(img, xy_px([0.15, -0.55]), xy_px([0.78, 0.42]), (40, 40, 48), 1)
    lo = SPAWN_DRILL_LO
    hi = SPAWN_DRILL_HI
    cv2.rectangle(img, xy_px(lo), xy_px(hi), (220, 200, 40), 1)
    u, v = xy_px(0.5 * (lo + hi))
    cv2.putText(img, "spawn", (u - 18, v - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (220, 200, 40), 1, cv2.LINE_AA)
    ox, oy = xy_px(origin[:2])
    cv2.drawMarker(img, (ox, oy), (80, 80, 255), cv2.MARKER_CROSS, 12, 1)
    for p in P:
        u, v = xy_px(p[:2])
        if 0 <= u < w and 0 <= v < h:
            img[v, u] = _height_bgr(p[2])
            if 0 <= v + 1 < h:
                img[v + 1, u] = img[v, u]
    if cmap is not None:
        if getattr(cmap, "drill_xy", None) is not None:
            u, v = xy_px(cmap.drill_xy)
            cv2.circle(img, (u, v), 9, (80, 220, 80), 2)
            cv2.putText(img, "drill", (u + 10, v - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (80, 220, 80), 1, cv2.LINE_AA)
        if getattr(cmap, "board_xy", None) is not None:
            u, v = xy_px(cmap.board_xy)
            cv2.circle(img, (u, v), 11, (80, 200, 255), 2)
            cv2.putText(img, "board", (u + 10, v + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (80, 200, 255), 1, cv2.LINE_AA)
    return img


def _range_panel(scan, w, h):
    rng = np.asarray(scan.ranges, float).reshape(-1)
    n_az = int(getattr(scan, "n_azim", 180) or 180)
    n_el = int(getattr(scan, "n_elev", 0) or max(1, len(rng) // max(n_az, 1)))
    n_el = max(1, min(n_el, max(1, len(rng) // max(n_az, 1))))
    use = n_el * n_az
    grid = rng[:use].reshape(n_el, n_az)
    vis = np.zeros_like(grid, np.float32)
    finite = np.isfinite(grid)
    vis[finite] = np.clip(grid[finite] / 1.6, 0.0, 1.0)
    vis = np.flipud(vis)  # steepest downward ring at the bottom
    u8 = (vis * 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    color[~np.flipud(finite)] = (20, 20, 24)
    return cv2.resize(color, (w, h), interpolation=cv2.INTER_NEAREST)


def paint_lidar_scene(viewer, scan, cmap=None, max_pts=360):
    """Draw LiDAR hits in the MuJoCo viewer user scene (spheres + a few rays)."""
    scn = getattr(viewer, "user_scn", None)
    if scn is None:
        return
    scn.ngeom = 0
    origin = np.asarray(scan.origin, float).reshape(3)
    pts = np.asarray(scan.points, float).reshape(-1, 3)
    ok = np.isfinite(pts).all(axis=1)
    P = pts[ok]
    if len(P) > max_pts:
        P = P[::int(np.ceil(len(P) / max_pts))]
    _add_sphere(scn, origin, 0.012, (0.95, 0.25, 0.15, 0.95))
    for p in P:
        t = float(np.clip(p[2] / 0.45, 0.0, 1.0))
        rgba = (0.15 + 0.85 * t, 0.75 * (1.0 - t) + 0.2, 1.0 - 0.7 * t, 0.85)
        _add_sphere(scn, p, 0.007, rgba)
    # a sparse set of rays so the sweep is readable
    step = max(1, len(P) // 36)
    for p in P[::step]:
        _add_line(scn, origin, p, (0.35, 0.75, 1.0, 0.25))
    if cmap is not None:
        if getattr(cmap, "drill_xy", None) is not None:
            d = np.array([cmap.drill_xy[0], cmap.drill_xy[1], 0.08], float)
            _add_sphere(scn, d, 0.022, (0.2, 1.0, 0.35, 0.55))
        if getattr(cmap, "board_xy", None) is not None:
            b = np.array([cmap.board_xy[0], cmap.board_xy[1], 0.30], float)
            _add_sphere(scn, b, 0.025, (1.0, 0.85, 0.2, 0.5))


def _add_sphere(scn, pos, r, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    scn.ngeom += 1
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([r, 0.0, 0.0], float),
        np.asarray(pos, float),
        np.eye(3).ravel(),
        np.asarray(rgba, float),
    )


def _add_line(scn, a, b, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    scn.ngeom += 1
    g = scn.geoms[scn.ngeom - 1]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, float))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 2.0,
                         np.asarray(a, float), np.asarray(b, float))


def _default_scene_cam():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.40, -0.10, 0.26]
    cam.distance = 1.35
    cam.azimuth = 62
    cam.elevation = -24
    return cam


def overlay_spawn_map(img, drill_xy=None):
    """Bottom-left top-down map: fixed board + left-of-board drill spawn box."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    H, W = img.shape[:2]
    mw, mh = min(210, W // 4), min(170, H // 3)
    x0, y0 = 8, H - 38 - mh
    if y0 < 8:
        return img
    panel = img[y0:y0 + mh, x0:x0 + mw]
    overlay = panel.copy()
    overlay[:] = (18, 22, 28)
    # World XY window around the table (x right, y up on the panel).
    x_lo, x_hi = 0.22, 0.78
    y_lo, y_hi = -0.52, 0.18

    def xy_px(xy):
        u = int(round((float(xy[0]) - x_lo) / (x_hi - x_lo) * (mw - 1)))
        v = int(round((1.0 - (float(xy[1]) - y_lo) / (y_hi - y_lo)) * (mh - 1)))
        return u, v

    # board footprint (vertical face → thin in X, long in Y). Overlay-only.
    bp = SPAWN_FACE_XY
    bh = SPAWN_FACE_EXTENT_XY
    p1 = xy_px([bp[0] - bh[0], bp[1] - bh[1]])
    p2 = xy_px([bp[0] + bh[0], bp[1] + bh[1]])
    cv2.rectangle(overlay, p1, p2, (90, 170, 255), -1)
    cv2.putText(overlay, "board", (p2[0] + 4, (p1[1] + p2[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 170, 255), 1, cv2.LINE_AA)
    lo, hi = SPAWN_DRILL_LO, SPAWN_DRILL_HI
    q1, q2 = xy_px(lo), xy_px(hi)
    cv2.rectangle(overlay, q1, q2, (40, 90, 70), -1)
    cv2.rectangle(overlay, q1, q2, (60, 230, 255), 1)
    cv2.putText(overlay, "spawn", (q1[0] + 4, q1[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 230, 255), 1, cv2.LINE_AA)
    if drill_xy is not None:
        cv2.circle(overlay, xy_px(drill_xy), 5, (80, 255, 120), -1)
    cv2.addWeighted(overlay, 0.72, panel, 0.28, 0, panel)
    img[y0:y0 + mh, x0:x0 + mw] = panel
    cv2.rectangle(img, (x0, y0), (x0 + mw, y0 + mh), (60, 230, 255), 1)
    label = (f"drill spawn  x {lo[0]:.2f}-{hi[0]:.2f} m  "
             f"y {lo[1]:+.2f}-{hi[1]:+.2f} m  (left of board)")
    cv2.putText(img, label, (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (60, 230, 255), 1, cv2.LINE_AA)
    return img


def compose_task_frame(scene_bgr, *, wrist_bgr=None, lidar_bgr=None, status="",
                       drill_xy=None):
    """Paste optional LiDAR (TL) / wrist (TR) insets onto a BGR scene frame."""
    img = np.ascontiguousarray(scene_bgr, dtype=np.uint8)
    H, W = img.shape[:2]
    IW, IH = W // 3, H // 3
    if lidar_bgr is not None:
        limg = cv2.resize(lidar_bgr, (IW, IH))
        img[8:8 + IH, 8:8 + IW] = limg
        cv2.rectangle(img, (7, 7), (9 + IW, 9 + IH), (180, 220, 255), 1)
        cv2.putText(img, "wrist LiDAR", (8, IH + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 220, 255), 1, cv2.LINE_AA)
    if wrist_bgr is not None:
        wimg = cv2.resize(wrist_bgr, (IW, IH))
        img[8:8 + IH, W - IW - 8:W - 8] = wimg
        cv2.rectangle(img, (W - IW - 9, 7), (W - 7, 9 + IH), (255, 255, 255), 1)
        cv2.putText(img, "wrist D435i", (W - IW - 2, IH + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    overlay_spawn_map(img, drill_xy=drill_xy)
    if status:
        cv2.rectangle(img, (0, H - 30), (W, H), (0, 0, 0), -1)
        cv2.putText(img, status, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _wrist_bgr(ctx, renderer, iw, ih):
    st = ctx.st
    if st["overlay"] is not None and ctx.sim_t() < st["overlay_until"]:
        return cv2.resize(st["overlay"][:, :, ::-1], (iw, ih))
    renderer.update_scene(ctx.data, camera="d435i_color")
    return renderer.render()[:, :, ::-1]


def _status_line(ctx):
    st = ctx.st
    return (f"{st['phase']}   t={ctx.sim_t():5.1f}s   "
            f"motor={'ON ' if st['motor'] else 'off'}   hole={st['depth']*1000:4.1f} mm")


def attach_video(ctx, path, size, fps):
    model, data, robot, st = ctx.model, ctx.data, ctx.robot, ctx.st

    class Recorder:
        def __init__(self):
            self.W, self.H = size
            self.IW, self.IH = self.W // 3, self.H // 3
            self.FPS = fps
            self.r = mujoco.Renderer(model, self.H, self.W)
            self.ri = mujoco.Renderer(model, self.IH, self.IW)
            self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (self.W, self.H))
            self.cam = _default_scene_cam()
            self.every = int(round(1.0 / self.FPS / robot.ctrl_dt)); self.k = 0

        def __call__(self):
            self.k += 1
            if self.k % self.every:
                return
            self.r.update_scene(data, self.cam)
            scene = self.r.render()[:, :, ::-1]
            wrist = _wrist_bgr(ctx, self.ri, self.IW, self.IH)
            lidar = None
            lscan = st.get("lidar_scan")
            if lscan is not None:
                lidar = render_lidar_image(lscan, st.get("lidar_map"), size=(self.IW, self.IH))
            img = compose_task_frame(scene, wrist_bgr=wrist, lidar_bgr=lidar,
                                     status=_status_line(ctx),
                                     drill_xy=getattr(ctx, "xy", None))
            self.vw.write(img)

        def close(self):
            self.vw.release()

    rec = Recorder()
    robot.callbacks.append(rec)
    return rec


def attach_viewer(ctx, *, wrist=True, lidar=True, size=(960, 540)):
    """One OpenCV window: MuJoCo scene + optional LiDAR / D435i insets (same layout as video)."""
    model, data, robot = ctx.model, ctx.data, ctx.robot
    W, H = int(size[0]), int(size[1])
    IW, IH = W // 3, H // 3
    win = "Smooty"
    r = mujoco.Renderer(model, H, W)
    ri = mujoco.Renderer(model, IH, IW) if wrist else None
    cam = _default_scene_cam()
    every = max(1, int(round(1.0 / 12.0 / robot.ctrl_dt)))
    k = [0]
    t_wall0 = [time.time(), data.time]
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)

    def sync():
        k[0] += 1
        if k[0] % every == 0:
            r.update_scene(data, cam)
            scene = r.render()[:, :, ::-1]
            wrist_img = _wrist_bgr(ctx, ri, IW, IH) if ri is not None else None
            lidar_img = None
            if lidar:
                lscan = ctx.st.get("lidar_scan")
                if lscan is not None:
                    lidar_img = render_lidar_image(
                        lscan, ctx.st.get("lidar_map"), size=(IW, IH))
            img = compose_task_frame(scene, wrist_bgr=wrist_img, lidar_bgr=lidar_img,
                                     status=_status_line(ctx),
                                     drill_xy=getattr(ctx, "xy", None))
            cv2.imshow(win, img)
            cv2.waitKey(1)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                raise SystemExit
        lag = (data.time - t_wall0[1]) - (time.time() - t_wall0[0])
        if lag > 0:
            time.sleep(lag)

    robot.callbacks.append(sync)
    return win


def attach_wrist_preview(ctx):
    """Legacy separate D435i window; prefer attach_viewer insets."""
    class WristPreview:
        def __init__(self):
            self.W, self.H = 640, 360
            self.r = mujoco.Renderer(ctx.model, self.H, self.W)
            self.every = max(1, int(round(1.0 / 12.0 / ctx.robot.ctrl_dt)))
            self.k = 0
            cv2.namedWindow("wrist D435i", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("wrist D435i", self.W, self.H)

        def __call__(self):
            self.k += 1
            if self.k % self.every:
                return
            if ctx.st["overlay"] is not None and ctx.sim_t() < ctx.st["overlay_until"]:
                img = cv2.resize(ctx.st["overlay"], (self.W, self.H))
            else:
                self.r.update_scene(ctx.data, camera="d435i_color")
                img = self.r.render()
            img = np.ascontiguousarray(img[:, :, ::-1], dtype=np.uint8)
            cv2.putText(img, str(ctx.st["phase"]), (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("wrist D435i", img)
            cv2.waitKey(1)

    ctx.robot.callbacks.append(WristPreview())


def attach_lidar_preview(ctx, *, window=False):
    """Refresh lidar_scan / lidar_map on ctx.st; optional legacy separate window."""
    lidar = getattr(ctx, "lidar", None)
    if lidar is None:
        return None

    class LidarPreview:
        def __init__(self):
            self.W, self.H = 640, 400
            self.every = max(1, int(round(1.0 / 6.0 / ctx.robot.ctrl_dt)))
            self.k = 0
            if window:
                cv2.namedWindow("wrist LiDAR", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("wrist LiDAR", self.W, self.H)

        def __call__(self):
            self.k += 1
            if self.k % self.every:
                return
            try:
                scan = lidar.scan()
                cmap = lidar.locate(scan)
            except Exception:
                return
            ctx.st["lidar_scan"] = scan
            ctx.st["lidar_map"] = cmap
            if getattr(ctx, "coarse_map", None) is None:
                ctx.coarse_map = cmap
            if not window:
                return
            img = render_lidar_image(scan, cmap, size=(self.W, self.H))
            cv2.putText(img, str(ctx.st["phase"]), (8, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("wrist LiDAR", img)
            cv2.waitKey(1)

    preview = LidarPreview()
    ctx.robot.callbacks.append(preview)
    return preview
