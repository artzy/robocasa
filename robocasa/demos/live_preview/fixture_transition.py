"""Fixture joint transitions for home-wrapped demo playback."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from termcolor import colored

from robocasa.demos.live_preview.robot_bridge import RobotSnapshot, apply_robot_snapshot
from robocasa.demos.live_preview.robot_session import get_eef_pose, resolve_base_env
from robocasa.models.fixtures.fridge import Fridge, FridgeBottomFreezer
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.utils import env_utils as EnvUtils

if TYPE_CHECKING:
    from robocasa.models.fixtures.fixture import Fixture


def resolve_env_fixture(env, ref_name: str):
    """Return a fixture by attribute name or fixture_refs entry."""
    if hasattr(env, ref_name):
        fixture = getattr(env, ref_name)
        if fixture is not None:
            return fixture
    refs = getattr(env, "fixture_refs", {})
    if ref_name in refs:
        ref_value = refs[ref_name]
        if isinstance(ref_value, tuple):
            return ref_value[0]
        return ref_value
    raise AttributeError(f"Fixture '{ref_name}' not found on environment")


def _fridge_door_joint_names(fridge: Fridge) -> list[str]:
    return list(fridge._fridge_door_joint_names)


def needs_fridge_close(env, fridge: Fridge, *, compartment: str = "fridge") -> bool:
    return not fridge.is_closed(env, compartment=compartment)


def compute_fridge_approach_pose(env, fridge: Fridge) -> tuple[np.ndarray, np.ndarray]:
    """Base pose in front of the fridge (matches CloseFridge placement)."""
    base_env = resolve_base_env(env)
    if isinstance(fridge, FridgeBottomFreezer):
        offset = (-0.30, -0.30)
    else:
        offset = (0.0, -0.30)
    base_pos, base_ori = EnvUtils.compute_robot_base_placement_pose(
        base_env, ref_fixture=fridge, offset=offset
    )
    return np.asarray(base_pos, dtype=float), np.asarray(base_ori, dtype=float)


def build_fridge_approach_snapshot(
    env,
    anchor_state: np.ndarray,
    fridge: Fridge,
    open_gripper_qpos: np.ndarray,
) -> RobotSnapshot:
    """Robot snapshot at fridge approach pose with current EEF orientation."""
    reset_to(env, {"states": np.asarray(anchor_state)})
    eef_pos, eef_quat, _ = get_eef_pose(env)
    base_pos, base_ori = compute_fridge_approach_pose(env, fridge)
    return RobotSnapshot(
        base_pos=base_pos,
        base_ori=base_ori,
        eef_pos=eef_pos.copy(),
        eef_quat=eef_quat.copy(),
        gripper=0.0,
    )


def _set_fixture_joints_normalized(
    env,
    fixture: Fixture,
    joint_names: list[str],
    normalized: dict[str, float],
) -> None:
    for j_name in joint_names:
        norm = float(np.clip(normalized[j_name], 0.0, 1.0))
        fixture.set_joint_state(norm, norm, env, [j_name])


def build_door_joint_close_states(
    env,
    anchor_state: np.ndarray,
    fixture: Fridge,
    open_gripper_qpos: np.ndarray,
    n_frames: int,
    *,
    compartment: str = "fridge",
    robot: RobotSnapshot | None = None,
) -> list[np.ndarray]:
    """Animate fridge door joints from current pose to closed while robot holds pose."""
    if n_frames <= 0:
        return []

    reset_to(env, {"states": np.asarray(anchor_state)})
    joint_names = _fridge_door_joint_names(fixture)
    if not joint_names:
        return []

    start_norm = fixture.get_joint_state(env, joint_names)
    if robot is None:
        robot = RobotSnapshot.from_env(env)

    states: list[np.ndarray] = []
    for i in range(n_frames):
        t = (i + 1) / n_frames
        lerped = {
            j_name: start_norm[j_name] * (1.0 - t) for j_name in joint_names
        }
        _set_fixture_joints_normalized(env, fixture, joint_names, lerped)
        apply_robot_snapshot(env, robot, open_gripper_qpos)
        states.append(np.array(env.sim.get_state().flatten()))

    if states and not fixture.is_closed(env, compartment=compartment):
        print(
            colored(
                "warning: fridge door may not be fully closed after transition",
                "yellow",
            )
        )
    return states


def capture_anchor_state(env) -> np.ndarray:
    return np.array(env.sim.get_state().flatten())
