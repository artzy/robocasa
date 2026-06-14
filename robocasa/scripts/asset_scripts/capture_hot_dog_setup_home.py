"""Capture HotDogSetup home preset from human demo episode 0."""

from __future__ import annotations

import json
from pathlib import Path

import robocasa  # noqa: F401
from robocasa.demos.live_preview.home_demo_playback import _load_episode, _make_env_from_dataset
from robocasa.demos.live_preview.home_pose import default_home_preset_path, save_home_preset
from robocasa.demos.live_preview.robot_session import (
    capture_home_from_env,
    get_eef_pose,
    get_robot_base_pose,
    set_gripper,
    set_robot_base_pose,
    solve_eef_pose,
)
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.utils.dataset_registry_utils import get_ds_path


def main() -> None:
    dataset = get_ds_path("HotDogSetup", source="human", split="pretrain")
    if dataset is None:
        raise RuntimeError("HotDogSetup pretrain dataset not found")
    dataset = Path(dataset)

    env, _ = _make_env_from_dataset(dataset, render=False)
    _, initial_state = _load_episode(dataset, 0)
    reset_to(env, initial_state)
    ep_meta = json.loads(initial_state["ep_meta"])

    open_gripper = env.robots[0].get_gripper_joint_positions("right").copy()
    base_pos, base_ori = get_robot_base_pose(env)
    eef_pos, eef_quat, _ = get_eef_pose(env)

    # Home: step back from the counter work area and raise the arm slightly.
    home_base = base_pos.copy()
    home_base[0] -= 0.8
    home_eef = eef_pos.copy()
    home_eef[2] += 0.15

    set_robot_base_pose(env, home_base, base_ori)
    solve_eef_pose(env, home_eef, eef_quat)
    set_gripper(env, 0.0, open_gripper)
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()

    home = capture_home_from_env(
        env,
        task_name="HotDogSetup",
        name="default_from_demo_ep0",
        layout=ep_meta.get("layout_id"),
        style=ep_meta.get("style_id"),
        seed=ep_meta.get("seed"),
    )
    out = default_home_preset_path("HotDogSetup")
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = save_home_preset(home, out)
    env.close()
    print(f"Saved HotDogSetup home preset to {saved}")


if __name__ == "__main__":
    main()
