"""
Teleoperate MovePan: pick coppelia_frying_pan from source fixture to target fixture.

Usage:
    python -m robocasa.demos.demo_move_pan
    python -m robocasa.demos.demo_move_pan --source-fixture counter --target-fixture sink
    python -m robocasa.demos.demo_move_pan --source-fixture stove --target-fixture sink --layout 15
"""

from __future__ import annotations

import argparse

from robocasa.demos.move_pan_live import (
    make_input_device,
    make_move_pan_env,
    parse_registries,
)
from robocasa.scripts.collect_demos import collect_human_trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teleop MovePan with CoppeliaSim pan.")
    parser.add_argument(
        "--source-fixture",
        type=str,
        default="counter",
        help="Source fixture: counter, sink, stove, cabinet, microwave, fridge",
    )
    parser.add_argument(
        "--target-fixture",
        type=str,
        default="sink",
        help="Target fixture for success check",
    )
    parser.add_argument(
        "--obj-registries",
        type=str,
        default="coppelia_edu",
        help="Comma-separated object registries (default: coppelia_edu)",
    )
    parser.add_argument("--layout", type=int, default=None, help="Kitchen layout id")
    parser.add_argument("--style", type=int, default=None, help="Kitchen style id")
    parser.add_argument(
        "--device",
        type=str,
        default="keyboard",
        choices=["keyboard", "spacemouse"],
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env, pygame_viewer, onscreen_renderer = make_move_pan_env(
        source_fixture=args.source_fixture,
        target_fixture=args.target_fixture,
        obj_registries=parse_registries(args.obj_registries),
        layout=args.layout,
        style=args.style,
        seed=args.seed,
        teleop=True,
    )
    device = make_input_device(env, args.device)

    while True:
        collect_human_trajectory(
            env,
            device,
            "right",
            "single-arm-opposed",
            mirror_actions=True,
            render=(onscreen_renderer != "mjviewer"),
            max_fr=30,
            pygame_viewer=pygame_viewer,
        )
        print()
