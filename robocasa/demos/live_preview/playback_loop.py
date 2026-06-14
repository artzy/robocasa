"""Unified state-sequence playback with pacing for home-wrapped demos."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from termcolor import colored

from robocasa.utils.playback_viewer import (
    DEFAULT_VIEWER_HEIGHT,
    DEFAULT_VIEWER_WIDTH,
    PygamePlaybackViewer,
    apply_mjviewer_camera_config,
    get_layout_camera_config,
)


DEFAULT_PLAYBACK_FPS = 60
DEFAULT_HOME_DWELL_SEC = 1.5
HOME_PLAYBACK_VIEWER_WIDTH = 1280
HOME_PLAYBACK_VIEWER_HEIGHT = 720


def fast_reset_state(env, flat_state: np.ndarray) -> None:
    """Lightweight sim restore for playback (no model/ep_meta reload)."""
    env.sim.set_state_from_flattened(np.asarray(flat_state))
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()


def hold_at_state(
    env,
    state: np.ndarray,
    *,
    pygame_viewer=None,
    cam_config=None,
    dwell_sec: float = DEFAULT_HOME_DWELL_SEC,
) -> None:
    fast_reset_state(env, state)
    if pygame_viewer is not None:
        if cam_config is not None:
            pygame_viewer.free_cam_config = cam_config
        pygame_viewer.update(env.sim, pace=False)
    elif env.renderer == "mjviewer":
        if env.viewer is None:
            env.initialize_renderer()
        if cam_config is not None:
            apply_mjviewer_camera_config(env, cam_config)
        env.viewer.update()
    elif getattr(env, "renderer", None) not in (None, "offscreen"):
        env.render()
    if dwell_sec > 0:
        time.sleep(dwell_sec)


def play_state_sequence(
    env,
    states: Sequence[np.ndarray],
    *,
    pygame_viewer=None,
    onscreen_renderer: str = "mjviewer",
    cam_config=None,
    max_fps: float = DEFAULT_PLAYBACK_FPS,
    dwell_home_start: bool = True,
    dwell_home_end: bool = True,
    dwell_sec: float = DEFAULT_HOME_DWELL_SEC,
) -> None:
    """Play flattened sim states with single pacing loop and fast reset."""
    if len(states) == 0:
        return

    frame_budget = 1.0 / max(max_fps, 1.0)

    if dwell_home_start:
        print(colored("Home - starting position", "green"))
        hold_at_state(
            env,
            states[0],
            pygame_viewer=pygame_viewer,
            cam_config=cam_config,
            dwell_sec=dwell_sec,
        )

    for state in states:
        start = time.time()
        if pygame_viewer is not None and not pygame_viewer.pump_events():
            break

        fast_reset_state(env, state)

        if pygame_viewer is not None:
            if not pygame_viewer.update(env.sim, pace=False):
                break
        elif env.renderer == "mjviewer":
            if env.viewer is None:
                env.initialize_renderer()
            if cam_config is not None:
                apply_mjviewer_camera_config(env, cam_config)
            env.viewer.update()
        else:
            env.render()

        elapsed = time.time() - start
        remaining = frame_budget - elapsed
        if remaining > 0:
            time.sleep(remaining)

    if dwell_home_end:
        print(colored("Home - task complete", "green"))
        hold_at_state(
            env,
            states[-1],
            pygame_viewer=pygame_viewer,
            cam_config=cam_config,
            dwell_sec=dwell_sec,
        )

    print(colored("Playback finished.", "green"))


def make_pygame_viewer_for_home(
    env,
    *,
    title: str,
    width: int = HOME_PLAYBACK_VIEWER_WIDTH,
    height: int = HOME_PLAYBACK_VIEWER_HEIGHT,
    cam_config=None,
) -> PygamePlaybackViewer:
    if cam_config is None:
        cam_config = get_layout_camera_config(env)
    return PygamePlaybackViewer(
        camera_name="robot0_agentview_center",
        width=width,
        height=height,
        title=title,
        free_cam_config=cam_config,
    )
