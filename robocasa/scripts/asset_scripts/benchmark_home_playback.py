"""Benchmark HotDogSetup home-wrapped playback build and frame timing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import robocasa  # noqa: F401
from robocasa.demos.live_preview.home_demo_playback import (
    _load_episode,
    _make_env_from_dataset,
    build_home_wrapped_states_for_task,
)
from robocasa.demos.live_preview.playback_loop import fast_reset_state, play_state_sequence
from robocasa.demos.live_preview.registry import home_demo_spec_for_task
from robocasa.utils.dataset_registry_utils import get_ds_path


def benchmark(
    task_name: str = "HotDogSetup",
    *,
    demo_stride: int | None = None,
    rebuild_wrap: bool = False,
    playback_fps: float = 60,
    renderer: str | None = "mjviewer",
    sample_frames: int = 100,
) -> None:
    spec = home_demo_spec_for_task(task_name)
    if demo_stride is None:
        demo_stride = spec.demo_stride if spec is not None else 1

    dataset = Path(get_ds_path(task_name, source="human", split="pretrain"))
    render = renderer is not None

    build_start = time.time()
    env, onscreen_renderer = _make_env_from_dataset(
        dataset, render=render, renderer=renderer
    )
    demo_states, initial_state = _load_episode(dataset, 0)
    ep_meta = json.loads(initial_state["ep_meta"])

    wrapped, _ = build_home_wrapped_states_for_task(
        env,
        task_name=task_name,
        demo_states=demo_states,
        initial_state=initial_state,
        layout=ep_meta.get("layout_id"),
        style=ep_meta.get("style_id"),
        demo_stride=demo_stride,
        rebuild_wrap=rebuild_wrap,
    )
    build_sec = time.time() - build_start

    n = min(sample_frames, len(wrapped))
    indices = np.linspace(0, len(wrapped) - 1, n, dtype=int)

    reset_times: list[float] = []
    for idx in indices:
        t0 = time.time()
        fast_reset_state(env, wrapped[idx])
        reset_times.append(time.time() - t0)

    avg_reset_ms = 1000.0 * float(np.mean(reset_times))
    theoretical_play_sec = len(wrapped) / playback_fps

    print(f"Task: {task_name}")
    print(f"Renderer: {onscreen_renderer}")
    print(f"Demo stride: {demo_stride}")
    print(f"Wrapped frames: {len(wrapped)} (source demo {len(demo_states)})")
    print(f"Build time: {build_sec:.1f}s")
    print(f"Avg fast_reset ({n} samples): {avg_reset_ms:.1f}ms")
    print(f"Theoretical playback @ {playback_fps}fps: {theoretical_play_sec:.1f}s")

    if render and renderer == "mjviewer":
        print("Running short mjviewer sample (first 30 frames)...")
        play_start = time.time()
        play_state_sequence(
            env,
            wrapped[:30],
            onscreen_renderer=onscreen_renderer,
            max_fps=playback_fps,
            dwell_home_start=False,
            dwell_home_end=False,
        )
        play_sec = time.time() - play_start
        print(f"30-frame sample playback: {play_sec:.1f}s ({play_sec / 30 * 1000:.1f}ms/frame)")

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="HotDogSetup")
    parser.add_argument("--demo-stride", type=int, default=None)
    parser.add_argument("--rebuild-wrap", action="store_true")
    parser.add_argument("--playback-fps", type=float, default=60)
    parser.add_argument("--renderer", default="mjviewer", choices=("mjviewer", "pygame", "mujoco"))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--sample-frames", type=int, default=100)
    args = parser.parse_args()

    benchmark(
        args.task,
        demo_stride=args.demo_stride,
        rebuild_wrap=args.rebuild_wrap,
        playback_fps=args.playback_fps,
        renderer=None if args.no_render else args.renderer,
        sample_frames=args.sample_frames,
    )


if __name__ == "__main__":
    main()
