"""Capture robot home pose (base + EEF) into a JSON preset for live preview demos."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import robocasa  # noqa: F401
from robocasa.demos.live_preview.home_pose import (
    default_home_preset_path,
    resolve_home_preset_path,
    save_home_preset,
)
from robocasa.demos.live_preview.registry import HOME_DEMO_REGISTRY, LIVE_DEMO_REGISTRY
from robocasa.demos.live_preview.robot_session import capture_home_from_env


def _load_make_env(task_name: str):
    spec = LIVE_DEMO_REGISTRY.get(task_name)
    if spec is None:
        raise ValueError(f"Unknown live demo task: {task_name}")
    mod = importlib.import_module(spec.module)
    make_env = getattr(mod, f"make_{task_name.lower()}_env", None)
    if make_env is None:
        make_env = getattr(mod, "make_move_pan_env", None)
    if make_env is None:
        raise ValueError(f"No make_env helper for task {task_name}")
    return make_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture robot home pose preset.")
    parser.add_argument(
        "--task",
        type=str,
        default="MovePan",
        choices=sorted(LIVE_DEMO_REGISTRY.keys() | HOME_DEMO_REGISTRY.keys()),
    )
    parser.add_argument("--source-fixture", type=str, default="counter")
    parser.add_argument("--target-fixture", type=str, default="stove")
    parser.add_argument("--layout", type=int, default=15)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: task default home preset).",
    )
    args = parser.parse_args()

    make_env = _load_make_env(args.task)
    env, _, _ = make_env(
        source_fixture=args.source_fixture,
        target_fixture=args.target_fixture,
        layout=args.layout,
        style=args.style,
        seed=args.seed,
        render_offscreen=True,
    )
    env.reset()

    default_out = default_home_preset_path(args.task)
    out_path = resolve_home_preset_path(args.task, args.output or default_out)
    if out_path is None:
        raise RuntimeError(f"No default home preset path for task {args.task}")

    home = capture_home_from_env(
        env,
        task_name=args.task,
        name=(
            f"layout{args.layout}_style{args.style}_seed{args.seed}_"
            f"{args.source_fixture}_{args.target_fixture}"
        ),
        layout=args.layout,
        style=args.style,
        seed=args.seed,
        source_fixture=args.source_fixture,
        target_fixture=args.target_fixture,
    )
    saved = save_home_preset(home, out_path)
    env.close()
    print(f"Saved {args.task} home preset to {saved}")


if __name__ == "__main__":
    main()
