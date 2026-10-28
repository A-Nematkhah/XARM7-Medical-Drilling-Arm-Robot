"""Optional MuJoCo viewer, wrist preview, and mp4 recorder."""
import time

import cv2
import mujoco
import numpy as np


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
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = [0.40, -0.10, 0.26]; self.cam.distance = 1.35
            self.cam.azimuth = 62; self.cam.elevation = -24
            self.every = int(round(1.0 / self.FPS / robot.ctrl_dt)); self.k = 0

        def __call__(self):
            self.k += 1
            if self.k % self.every:
                return
            self.r.update_scene(data, self.cam)
            img = self.r.render()[:, :, ::-1].copy()
            if st["overlay"] is not None and ctx.sim_t() < st["overlay_until"]:
                ins = cv2.resize(st["overlay"], (self.IW, self.IH))[:, :, ::-1]
            else:
                self.ri.update_scene(data, camera="d435i_color")
                ins = self.ri.render()[:, :, ::-1]
            img[8:8 + self.IH, self.W - self.IW - 8:self.W - 8] = ins
            cv2.rectangle(img, (self.W - self.IW - 9, 7), (self.W - 7, 9 + self.IH), (255, 255, 255), 1)
            cv2.putText(img, "wrist D435i", (self.W - self.IW - 2, self.IH + 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            txt = (f"{st['phase']}   t={ctx.sim_t():5.1f}s   "
                   f"motor={'ON ' if st['motor'] else 'off'}   hole={st['depth']*1000:4.1f} mm")
            cv2.rectangle(img, (0, self.H - 30), (self.W, self.H), (0, 0, 0), -1)
            cv2.putText(img, txt, (10, self.H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            self.vw.write(img)

        def close(self):
            self.vw.release()

    rec = Recorder()
    robot.callbacks.append(rec)
    return rec


def attach_viewer(ctx):
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(ctx.model, ctx.data)
    t_wall0 = [time.time(), ctx.data.time]

    def sync():
        viewer.sync()
        lag = (ctx.data.time - t_wall0[1]) - (time.time() - t_wall0[0])
        if lag > 0:
            time.sleep(lag)
        if not viewer.is_running():
            raise SystemExit

    ctx.robot.callbacks.append(sync)
    return viewer


def attach_wrist_preview(ctx):
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
