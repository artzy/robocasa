"""Smoke tests for MovePan task and pan move sandbox."""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np
import robocasa
from robocasa.demos.demo_pan_move_sandbox import DEFAULT_PAN_MJCF, PanMoveSandboxEnv
from robocasa.demos.live_preview.home_pose import (
    HOME_BASE_TOLERANCE,
    HOME_EEF_TOLERANCE,
    load_home_preset,
    resolve_home_preset_path,
)
from robocasa.demos.move_pan_live import (
    MAX_GRASP_SITE_ATTACH_DIST,
    PAN_GRASP_SITE,
    _generate_preview_states,
    _get_eef_pose,
    _get_pan_grasp_pose,
    _get_pan_pose,
    _get_robot_base_pose,
    make_move_pan_env,
)
from robocasa.environments import REGISTERED_KITCHEN_ENVS
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.utils.env_utils import create_env
from robosuite.controllers import load_composite_controller_config

DEFAULT_HOME_PRESET = resolve_home_preset_path("MovePan", None)


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


def verify_grasp_site(layout: int = 15, style: int = 34, seed: int = 0) -> None:
    env, _, _ = make_move_pan_env(
        teleop=False,
        layout=layout,
        style=style,
        seed=seed,
        render_offscreen=True,
    )
    env.reset()
    site_id = env.sim.model.site_name2id(PAN_GRASP_SITE)
    assert site_id >= 0

    pan_pos, _ = _get_pan_pose(env)
    grasp_pos, _, _ = _get_pan_grasp_pose(env)
    grasp_body_dist = float(np.linalg.norm(grasp_pos - pan_pos))
    assert grasp_body_dist > 0.05, (
        f"grasp site should differ from body center ({grasp_body_dist:.3f}m)"
    )
    env.close()
    print(f"OK pan_grasp_site (grasp-body dist {grasp_body_dist:.3f}m)")


def verify_home_start_end(layout: int = 15, style: int = 34, seed: int = 0) -> None:
    home = load_home_preset(DEFAULT_HOME_PRESET)
    env, _, _ = make_move_pan_env(
        teleop=False,
        layout=layout,
        style=style,
        seed=seed,
        render_offscreen=True,
    )
    states, _, _ = _generate_preview_states(
        env,
        home_preset=DEFAULT_HOME_PRESET,
        layout=layout,
        style=style,
        seed=seed,
        source_fixture="counter",
        target_fixture="stove",
    )

    for label, state in (("start", states[0]), ("end", states[-1])):
        reset_to(env, {"states": state})
        base_pos, _ = _get_robot_base_pose(env)
        eef_pos, _, _ = _get_eef_pose(env)
        base_dist = float(np.linalg.norm(base_pos - home.base_pos))
        eef_dist = float(np.linalg.norm(eef_pos - home.eef_pos))
        assert base_dist < HOME_BASE_TOLERANCE, (
            f"{label} base should match home ({base_dist:.3f}m)"
        )
        assert eef_dist < HOME_EEF_TOLERANCE, (
            f"{label} eef should match home ({eef_dist:.3f}m)"
        )

    env.close()
    print(
        f"OK home start/end ({len(states)} frames, "
        f"preset {DEFAULT_HOME_PRESET.name})"
    )


def verify_preview_arm_motion(layout: int = 15, style: int = 34, seed: int = 0) -> None:
    env, _, _ = make_move_pan_env(
        teleop=False,
        layout=layout,
        style=style,
        seed=seed,
        render_offscreen=True,
    )
    states, home, timeline = _generate_preview_states(
        env,
        home_preset=DEFAULT_HOME_PRESET,
        layout=layout,
        style=style,
        seed=seed,
        source_fixture="counter",
        target_fixture="stove",
    )

    assert len(states) >= 200
    assert len(states) == len(timeline)

    max_grasp_site_dist = 0.0
    for state, frame in zip(states, timeline):
        if not frame.attach:
            continue
        reset_to(env, {"states": state})
        eef_pos, _, _ = _get_eef_pose(env)
        grasp_pos, _, _ = _get_pan_grasp_pose(env)
        max_grasp_site_dist = max(
            max_grasp_site_dist, float(np.linalg.norm(grasp_pos - eef_pos))
        )

    assert max_grasp_site_dist < MAX_GRASP_SITE_ATTACH_DIST, (
        f"eef should align with grasp site during attach "
        f"(max dist {max_grasp_site_dist:.3f}m)"
    )

    reset_to(env, {"states": states[-1]})
    final_eef, _, _ = _get_eef_pose(env)
    dist = float(np.linalg.norm(final_eef - home.eef_pos))
    assert dist < HOME_EEF_TOLERANCE, (
        f"preview should end at home eef (dist {dist:.3f}m)"
    )
    env.close()
    print(
        f"OK preview arm motion (eef-home delta {dist:.3f}m, "
        f"max grasp-site dist {max_grasp_site_dist:.3f}m, "
        f"{len(states)} frames)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=int, default=15)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    verify_kitchen_move_pan(layout=args.layout, style=args.style, seed=args.seed)
    verify_grasp_site(layout=args.layout, style=args.style, seed=args.seed)
    verify_home_start_end(layout=args.layout, style=args.style, seed=args.seed)
    verify_preview_arm_motion(layout=args.layout, style=args.style, seed=args.seed)
    verify_sandbox()
    print("ALL OK")


if __name__ == "__main__":
    main()
