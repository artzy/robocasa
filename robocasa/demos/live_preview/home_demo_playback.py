"""Human demo playback with Home start/end for demo_tasks."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import robosuite
from termcolor import colored

import robocasa  # noqa: F401
import robocasa.utils.lerobot_utils as LU
from robocasa.demos.live_preview.home_pose import RobotHomePose
from robocasa.demos.live_preview.home_wrap_cache import (
    cache_path,
    load_cached_states,
    save_cached_states,
)
from robocasa.demos.live_preview.playback_loop import (
    DEFAULT_HOME_DWELL_SEC,
    DEFAULT_PLAYBACK_FPS,
    HOME_PLAYBACK_VIEWER_HEIGHT,
    HOME_PLAYBACK_VIEWER_WIDTH,
    fast_reset_state,
    make_pygame_viewer_for_home,
    play_state_sequence,
)
from robocasa.demos.live_preview.registry import (
    home_demo_spec_for_task,
    pre_return_fn_for_task,
)
from robocasa.demos.live_preview.robot_bridge import (
    build_home_wrapped_demo_states,
)
from robocasa.demos.live_preview.robot_session import (
    resolve_home_pose,
    resolve_robot,
)
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.scripts.dataset_scripts.playback_utils import resolve_instruction_from_ep_meta
from robocasa.utils.playback_viewer import get_layout_camera_config

PREVIEW_FPS = DEFAULT_PLAYBACK_FPS
PREVIEW_HOME_DWELL_SEC = DEFAULT_HOME_DWELL_SEC
DEMO_TAIL_EXTEND = 50
DEFAULT_HOME_RENDERER = "mjviewer"


def _make_env_from_dataset(
    dataset: Path,
    *,
    render: bool,
    renderer: str | None = None,
):
    env_meta = LU.get_env_metadata(dataset)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    onscreen_renderer = (
        renderer
        if renderer is not None
        else (DEFAULT_HOME_RENDERER if render else "mjviewer")
    )
    env_kwargs["has_renderer"] = render and onscreen_renderer == "mujoco"
    env_kwargs["renderer"] = (
        onscreen_renderer if onscreen_renderer != "pygame" else "mjviewer"
    )
    env_kwargs["has_offscreen_renderer"] = render and onscreen_renderer in (
        "mujoco",
        "pygame",
    )
    env_kwargs["use_camera_obs"] = False
    env_kwargs["render_camera"] = (
        None if render and onscreen_renderer != "mujoco" else "robot0_agentview_center"
    )
    env = robosuite.make(**env_kwargs)
    env._playback_onscreen_renderer = onscreen_renderer
    return env, onscreen_renderer


def _load_episode(dataset: Path, episode_index: int = 0):
    states = LU.get_episode_states(dataset, episode_index)
    initial_state = {
        "states": states[0],
        "model": LU.get_episode_model_xml(dataset, episode_index),
        "ep_meta": json.dumps(LU.get_episode_meta(dataset, episode_index)),
    }
    return states, initial_state


def build_home_wrapped_states_for_task(
    env,
    *,
    task_name: str,
    demo_states: np.ndarray,
    initial_state: dict,
    home_preset: Path | str | None = None,
    layout: int | None = None,
    style: int | None = None,
    seed: int | None = None,
    extend_demo_tail: int | None = None,
    demo_stride: int | None = None,
    rebuild_wrap: bool = False,
    episode_index: int = 0,
) -> tuple[list[np.ndarray], RobotHomePose]:
    spec = home_demo_spec_for_task(task_name)
    if demo_stride is None:
        demo_stride = spec.demo_stride if spec is not None else 1
    if extend_demo_tail is None:
        extend_demo_tail = spec.demo_tail_extend if spec is not None else DEMO_TAIL_EXTEND

    preset_path = Path(home_preset) if home_preset is not None else None
    cache_file = cache_path(
        task_name,
        episode_index,
        home_preset=preset_path,
        demo_stride=demo_stride,
        extend_demo_tail=extend_demo_tail,
    )
    if not rebuild_wrap:
        cached = load_cached_states(cache_file)
        if cached is not None:
            print(colored(f"Loaded wrapped states from cache: {cache_file}", "cyan"))
            reset_to(env, initial_state)
            home = resolve_home_pose(
                env,
                task_name,
                home_preset,
                layout=layout,
                style=style,
                seed=seed,
            )
            return cached, home

    reset_to(env, initial_state)
    ep_meta = json.loads(initial_state["ep_meta"])
    lang = resolve_instruction_from_ep_meta(ep_meta)
    if lang:
        print(colored(f"Instruction: {lang}", "green"))
    print(colored("Spawning environment...", "yellow"))

    open_gripper_qpos = resolve_robot(env).get_gripper_joint_positions("right").copy()
    home = resolve_home_pose(
        env,
        task_name,
        home_preset,
        layout=layout or ep_meta.get("layout_id"),
        style=style or ep_meta.get("style_id"),
        seed=seed or ep_meta.get("seed"),
    )

    demo_subset = np.asarray(demo_states)
    if demo_stride > 1:
        demo_subset = demo_subset[::demo_stride]
        print(
            colored(
                f"Demo stride {demo_stride}: {len(demo_states)} -> {len(demo_subset)} frames",
                "cyan",
            )
        )

    wrapped = build_home_wrapped_demo_states(
        env,
        demo_subset,
        home,
        open_gripper_qpos,
        extend_demo_tail=extend_demo_tail,
        pre_return_fn=pre_return_fn_for_task(task_name),
    )
    save_cached_states(
        cache_file,
        wrapped,
        meta={
            "task": task_name,
            "episode": episode_index,
            "demo_stride": demo_stride,
            "frames": len(wrapped),
        },
    )
    print(colored(f"Cached wrapped states: {cache_file}", "cyan"))
    return wrapped, home


def play_human_demo_with_home(
    task_name: str,
    *,
    dataset: Path | str,
    episode_index: int = 0,
    render_offscreen: bool = False,
    video_path: str | bool = False,
    home_preset: Path | str | None = None,
    layout: int | None = None,
    style: int | None = None,
    seed: int | None = None,
    extend_demo_tail: int | None = None,
    demo_stride: int | None = None,
    playback_fps: float = DEFAULT_PLAYBACK_FPS,
    renderer: str | None = None,
    viewer_width: int = HOME_PLAYBACK_VIEWER_WIDTH,
    viewer_height: int = HOME_PLAYBACK_VIEWER_HEIGHT,
    dwell_sec: float = DEFAULT_HOME_DWELL_SEC,
    no_dwell: bool = False,
    rebuild_wrap: bool = False,
) -> None:
    """Play one human demo episode with Home hold -> demo -> return Home."""
    dataset = Path(dataset)
    render = not render_offscreen and video_path is False
    effective_dwell = 0.0 if no_dwell else dwell_sec

    print(
        colored(
            f"Building '{task_name}' demo with Home (start -> task -> return)...",
            "yellow",
        )
    )
    build_start = time.time()
    env, onscreen_renderer = _make_env_from_dataset(
        dataset, render=render, renderer=renderer
    )
    demo_states, initial_state = _load_episode(dataset, episode_index)

    wrapped, home = build_home_wrapped_states_for_task(
        env,
        task_name=task_name,
        demo_states=demo_states,
        initial_state=initial_state,
        home_preset=home_preset,
        layout=layout,
        style=style,
        seed=seed,
        extend_demo_tail=extend_demo_tail,
        demo_stride=demo_stride,
        rebuild_wrap=rebuild_wrap,
        episode_index=episode_index,
    )
    build_elapsed = time.time() - build_start
    print(
        colored(
            f"Preview ready: {len(wrapped)} frames "
            f"(demo {len(demo_states)} source), home='{home.name or 'preset'}', "
            f"build {build_elapsed:.1f}s, renderer={onscreen_renderer}, fps={playback_fps}",
            "cyan",
        )
    )

    cam_config = get_layout_camera_config(env)
    pygame_viewer = None
    if render and onscreen_renderer == "pygame":
        pygame_viewer = make_pygame_viewer_for_home(
            env,
            title=f"RoboCasa {task_name}",
            width=viewer_width,
            height=viewer_height,
            cam_config=cam_config,
        )
        print(colored("Opening viewer (pygame window)...", "yellow"))
    elif render and onscreen_renderer == "mjviewer":
        print(
            colored(
                "Opening MuJoCo viewer (GPU)... check taskbar if hidden",
                "yellow",
            )
        )

    print(colored(f"Playing back episode: {task_name} (home-wrapped demo)", "yellow"))
    play_state_sequence(
        env,
        wrapped,
        pygame_viewer=pygame_viewer,
        onscreen_renderer=onscreen_renderer,
        cam_config=cam_config,
        max_fps=playback_fps,
        dwell_home_start=not no_dwell,
        dwell_home_end=not no_dwell,
        dwell_sec=effective_dwell,
    )

    if pygame_viewer is not None:
        print(
            colored(
                "Close the window, press Esc/Enter/q in the viewer, "
                "or press Enter/q in the terminal.",
                "green",
            )
        )
        pygame_viewer.wait_until_closed(env.sim)
        pygame_viewer.close()
    elif render and onscreen_renderer == "mjviewer":
        try:
            input("Press Enter to close the viewer...")
        except EOFError:
            pass
        if env.viewer is not None:
            env.viewer.close()
            env.viewer = None

    env.close()
