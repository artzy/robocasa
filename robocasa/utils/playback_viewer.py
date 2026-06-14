"""On-screen playback viewer for platforms without OpenCV GUI (e.g. Windows + headless cv2)."""

from __future__ import annotations

import sys

import numpy as np


def _stdin_ready():
    """Return True when Enter or q was pressed in the terminal (non-blocking)."""
    if sys.platform == "win32":
        import msvcrt

        if not msvcrt.kbhit():
            return False
        ch = msvcrt.getwch()
        if ch in ("\r", "\n", "q", "Q"):
            return True
        # Drain remaining chars until Enter for pasted/multi-key input.
        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\r", "\n", "q", "Q"):
                return True
        return False

    import select

    if not select.select([sys.stdin], [], [], 0)[0]:
        return False
    ch = sys.stdin.read(1)
    if ch in ("\r", "\n", "q", "Q"):
        return True
    while select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        if ch in ("\r", "\n"):
            return True
    return False


def resolve_layout_id(env):
    cur = env
    while cur is not None:
        layout_id = getattr(cur, "layout_id", None)
        if layout_id is not None:
            return layout_id
        cur = getattr(cur, "env", None)
    return None


def get_layout_camera_config(env):
    from robocasa.utils import camera_utils as CamUtils

    layout_id = resolve_layout_id(env)
    return dict(CamUtils.LAYOUT_CAMS.get(layout_id, CamUtils.DEFAULT_LAYOUT_CAM))


def render_free_camera(sim, width, height, cam_config):
    import mujoco

    ctx = sim._render_context_offscreen
    if ctx is None:
        raise RuntimeError("offscreen renderer is required for free-camera rendering")

    ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    ctx.cam.lookat[:] = cam_config["lookat"]
    ctx.cam.distance = float(cam_config["distance"])
    ctx.cam.azimuth = float(cam_config["azimuth"])
    ctx.cam.elevation = float(cam_config["elevation"])
    ctx.render(width=width, height=height, camera_id=-1)
    frame = ctx.read_pixels(width, height)
    return np.ascontiguousarray(frame)


def apply_mjviewer_camera_config(env, cam_config):
    viewer = getattr(env, "viewer", None)
    if viewer is None:
        return
    viewer.camera_config = dict(cam_config)
    passive = getattr(viewer, "viewer", None)
    if passive is None:
        return
    passive.cam.lookat[:] = cam_config["lookat"]
    passive.cam.distance = float(cam_config["distance"])
    passive.cam.azimuth = float(cam_config["azimuth"])
    passive.cam.elevation = float(cam_config["elevation"])
    passive.cam.type = 0


class PygamePlaybackViewer:
    def __init__(
        self,
        camera_name="robot0_agentview_center",
        width=768,
        height=512,
        title="RoboCasa",
        free_cam_config=None,
    ):
        import pygame

        pygame.init()
        self._pygame = pygame
        self.camera_name = camera_name
        self.free_cam_config = free_cam_config
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self._should_close = False
        self._last_surface = None

    def _handle_events(self):
        pg = self._pygame
        pg.event.pump()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self._should_close = True
            elif event.type == pg.KEYDOWN and event.key in (
                pg.K_ESCAPE,
                pg.K_RETURN,
                pg.K_KP_ENTER,
                pg.K_q,
            ):
                self._should_close = True

    def pump_events(self):
        """Process window events without rendering (call before slow sim work)."""
        self._handle_events()
        return not self._should_close

    def update(self, sim):
        self._handle_events()
        if self._should_close:
            return False

        if self.free_cam_config is not None:
            frame = render_free_camera(
                sim, self.width, self.height, self.free_cam_config
            )
        else:
            frame = sim.render(
                camera_name=self.camera_name,
                width=self.width,
                height=self.height,
            )
        frame = np.ascontiguousarray(np.flipud(frame))
        surf = self._pygame.image.frombuffer(
            frame.tobytes(), (self.width, self.height), "RGB"
        )
        self._last_surface = surf
        self.screen.blit(surf, (0, 0))
        self._pygame.display.flip()
        self.clock.tick(60)
        return True

    def wait_until_closed(self, sim=None):
        """Keep the window responsive until the user closes it."""
        pg = self._pygame
        if not sys.stdin.isatty():
            # Piped/non-interactive run: show the last frame briefly, then exit.
            if sim is not None:
                self.update(sim)
            elif self._last_surface is not None:
                self.screen.blit(self._last_surface, (0, 0))
                pg.display.flip()
            pg.time.wait(500)
            return

        while not self._should_close:
            self._handle_events()
            if _stdin_ready():
                self._should_close = True
                break
            if sim is not None:
                if not self.update(sim):
                    break
            elif self._last_surface is not None:
                self.screen.blit(self._last_surface, (0, 0))
                pg.display.flip()
                self.clock.tick(30)
            else:
                pg.time.wait(16)

    def close(self):
        if getattr(self, "_pygame", None) is not None:
            self._pygame.display.quit()
            self._pygame.quit()
            self._pygame = None


def opencv_gui_available():
    try:
        import cv2

        return "GUI:                           NONE" not in cv2.getBuildInformation()
    except Exception:
        return False


def onscreen_renderer_name():
    if sys.platform == "win32":
        return "mujoco" if opencv_gui_available() else "pygame"
    return "mjviewer"


def robosuite_viewer_kwargs(onscreen_renderer=None, render_camera="robot0_frontview"):
    if onscreen_renderer is None:
        onscreen_renderer = onscreen_renderer_name()
    kwargs = {
        "has_renderer": onscreen_renderer == "mujoco",
        "has_offscreen_renderer": onscreen_renderer in ("mujoco", "pygame"),
        "renderer": onscreen_renderer if onscreen_renderer != "pygame" else "mjviewer",
    }
    if onscreen_renderer == "mujoco":
        kwargs["render_camera"] = render_camera
    return onscreen_renderer, kwargs


def find_pygame_viewer(env):
    cur = env
    while cur is not None:
        viewer = getattr(cur, "_pygame_viewer", None)
        if viewer is not None:
            return viewer
        cur = getattr(cur, "env", None)
    return None


def render_onscreen(env, pygame_viewer=None):
    if pygame_viewer is None:
        pygame_viewer = find_pygame_viewer(env)
    if pygame_viewer is not None:
        return pygame_viewer.update(env.sim)
    env.render()
    return True
