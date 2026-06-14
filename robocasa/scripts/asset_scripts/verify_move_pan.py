"""Smoke tests for MovePan task and pan move sandbox."""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np
import robocasa
from robocasa.demos.demo_pan_move_sandbox import DEFAULT_PAN_MJCF, PanMoveSandboxEnv
from robocasa.demos.move_pan_live import _generate_preview_states, _get_eef_pose, make_move_pan_env
from robocasa.environments import REGISTERED_KITCHEN_ENVS
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.utils.env_utils import create_env
from robosuite.controllers import load_composite_controller_config


def verify_kitchen_move_pan(layout: int = 15, style: int = 34, seed: int = 0) -> None:
    assert "MovePan" in REGISTERED_KITCHEN_ENVS

    combos = [
        ("counter", "sink"),
        ("stove", "sink"),
        ("counter", "stove"),
    ]
    for src, tgt in combos:
        t0 = time.time()
        env = create_env(
            "MovePan",
            source_fixture=src,
            target_fixture=tgt,
            obj_registries=("coppelia_edu",),
            split="pretrain",
            seed=seed,
            layout_ids=layout,
            style_ids=style,
        )
        env.reset()
        mjcf_path = env.object_cfgs[0]["info"]["mjcf_path"]
        assert "coppelia_frying_pan" in mjcf_path.replace("\\", "/"), mjcf_path
        assert not env._check_success()
        env.close()
        print(f"OK MovePan {src}->{tgt} ({time.time() - t0:.1f}s)")

    env = gym.make(
        "robocasa/MovePan",
        source_fixture="counter",
        target_fixture="sink",
        obj_registries=("coppelia_edu",),
        split="pretrain",
        seed=seed,
        layout_ids=layout,
        style_ids=style,
    )
    env.reset()
    env.close()
    print("OK gym.make(robocasa/MovePan)")


def verify_sandbox() -> None:
    env = PanMoveSandboxEnv(
        robots="Panda",
        controller_configs=load_composite_controller_config(robot="Panda"),
        obj_mjcf_path=str(DEFAULT_PAN_MJCF),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
    )
    env.reset()
    assert env._pan_obj is not None
    assert not env._check_success()
    env.close()
    print("OK PanMoveSandboxEnv")


def verify_preview_arm_motion(layout: int = 15, style: int = 34, seed: int = 0) -> None:
    env, _, _ = make_move_pan_env(
        teleop=False,
        layout=layout,
        style=style,
        seed=seed,
        render_offscreen=True,
    )
    env.reset()
    home_eef, _ = _get_eef_pose(env)

    states = _generate_preview_states(env)
    assert len(states) >= 100

    reset_to(env, {"states": states[-1]})
    final_eef, _ = _get_eef_pose(env)
    dist = float(np.linalg.norm(final_eef - home_eef))
    assert dist > 0.05, f"expected arm motion, got {dist:.3f}m"
    env.close()
    print(f"OK preview arm motion (eef delta {dist:.2f}m, {len(states)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=int, default=15)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    verify_kitchen_move_pan(layout=args.layout, style=args.style, seed=args.seed)
    verify_preview_arm_motion(layout=args.layout, style=args.style, seed=args.seed)
    verify_sandbox()
    print("ALL OK")


if __name__ == "__main__":
    main()
