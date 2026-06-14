"""Robot base/EEF pose helpers shared by live preview demos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.controllers.parts.arm.ik import InverseKinematicsController
from termcolor import colored

from robocasa.demos.live_preview.home_pose import (
    RobotHomePose,
    load_home_preset,
    resolve_home_preset_path,
    save_home_preset,
    warn_home_mismatch,
)
from robocasa.utils import env_utils as EnvUtils

IK_POS_TOL = 0.015
IK_MAX_ITERS = 40


def resolve_base_env(env):
    cur = env
    while cur is not None:
        if hasattr(cur, "robots") and hasattr(cur, "sim"):
            return cur
        cur = getattr(cur, "env", None)
    return env


def resolve_robot(env):
    return resolve_base_env(env).robots[0]


def get_robot_base_pose(env):
    base_env = resolve_base_env(env)
    body_id = base_env.sim.model.body_name2id("mobilebase0_base")
    pos = np.array(base_env.sim.data.body_xpos[body_id], dtype=float)
    ori = T.mat2euler(base_env.sim.data.body_xmat[body_id].reshape(3, 3))
    return pos, ori


def set_robot_base_pose(env, global_pos: np.ndarray, global_ori: np.ndarray):
    base_env = resolve_base_env(env)
    EnvUtils.set_robot_to_position(base_env, global_pos)
    yaw_addr = base_env.sim.model.get_joint_qpos_addr("mobilebase0_joint_mobile_yaw")
    base_env.sim.data.qpos[yaw_addr] = (
        float(global_ori[2]) - float(base_env.init_robot_base_ori_anchor[2])
    )
    base_env.sim.forward()


def compute_robot_base_near_fixture(env, fixture):
    base_env = resolve_base_env(env)
    return EnvUtils.compute_robot_base_placement_pose(
        base_env, fixture, ref_object="pan"
    )


def get_eef_pose(env):
    robot = resolve_robot(env)
    site_id = robot.eef_site_id["right"]
    pos = np.array(env.sim.data.site_xpos[site_id], dtype=float)
    mat = env.sim.data.site_xmat[site_id].reshape(3, 3)
    quat_xyzw = T.mat2quat(mat)
    quat = T.convert_quat(quat_xyzw, to="wxyz")
    return pos, quat, mat


def _arm_qpos_indices(env):
    robot = resolve_robot(env)
    split = robot._joint_split_idx
    return np.array(robot._ref_arm_joint_pos_indexes[:split], dtype=int)


def _eef_site_name(env) -> str:
    robot = resolve_robot(env)
    site_id = robot.eef_site_id["right"]
    return env.sim.model.site(site_id).name


def set_gripper(env, closed: float, open_qpos: np.ndarray):
    robot = resolve_robot(env)
    idx = robot._ref_gripper_joint_pos_indexes["right"]
    closed = float(np.clip(closed, 0.0, 1.0))
    env.sim.data.qpos[idx] = open_qpos * (1.0 - closed)


def solve_eef_pose(
    env,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    *,
    iters: int = IK_MAX_ITERS,
    pos_tol: float = IK_POS_TOL,
) -> bool:
    sim = env.sim
    arm_qpos_idx = _arm_qpos_indices(env)
    ref_name = _eef_site_name(env)
    site_id = resolve_robot(env).eef_site_id["right"]

    tgt_mat = T.quat2mat(T.convert_quat(target_quat, to="xyzw"))

    for _ in range(iters):
        cur_pos = np.array(sim.data.site_xpos[site_id], dtype=float)
        dpos = target_pos - cur_pos
        if np.linalg.norm(dpos) < pos_tol:
            return True

        cur_quat_xyzw = T.mat2quat(sim.data.site_xmat[site_id].reshape(3, 3))
        cur_mat = T.quat2mat(cur_quat_xyzw)
        drot = tgt_mat @ cur_mat.T

        q0 = sim.data.qpos[arm_qpos_idx].copy()
        q_des = InverseKinematicsController.compute_joint_positions(
            sim=sim,
            initial_joint=q0,
            joint_indices=arm_qpos_idx,
            ref_name=ref_name,
            control_freq=20.0,
            use_delta=True,
            dpos=dpos * 0.35,
            drot=drot.reshape(-1),
            integration_dt=0.05,
            Kpos=0.95,
            Kori=0.95,
        )
        sim.data.qpos[arm_qpos_idx] = q_des
        sim.forward()

    return np.linalg.norm(target_pos - sim.data.site_xpos[site_id]) < pos_tol * 3.0


def restore_robot_qpos(env, qpos: np.ndarray) -> None:
    robot = resolve_robot(env)
    for addr, val in zip(robot._ref_joint_pos_indexes, qpos):
        env.sim.data.qpos[addr] = val
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()


def snapshot_robot_qpos(env) -> np.ndarray:
    robot = resolve_robot(env)
    return np.array(
        [env.sim.data.qpos[addr] for addr in robot._ref_joint_pos_indexes],
        dtype=float,
    )


def capture_home_from_env(
    env,
    *,
    task_name: str = "",
    name: str = "",
    layout: int | None = None,
    style: int | None = None,
    seed: int | None = None,
    source_fixture: str | None = None,
    target_fixture: str | None = None,
    extra: dict[str, Any] | None = None,
) -> RobotHomePose:
    base_pos, base_ori = get_robot_base_pose(env)
    eef_pos, eef_quat, _ = get_eef_pose(env)
    meta = dict(extra or {})
    if source_fixture is not None:
        meta["source_fixture"] = source_fixture
    if target_fixture is not None:
        meta["target_fixture"] = target_fixture
    return RobotHomePose(
        task_name=task_name,
        name=name,
        layout=layout,
        style=style,
        seed=seed,
        base_pos=base_pos.copy(),
        base_ori=base_ori.copy(),
        eef_pos=eef_pos.copy(),
        eef_quat=eef_quat.copy(),
        gripper=0.0,
        extra=meta,
    )


def apply_home_pose(
    env,
    home: RobotHomePose,
    open_gripper_qpos: np.ndarray,
) -> bool:
    set_robot_base_pose(env, home.base_pos, home.base_ori)
    ik_ok = solve_eef_pose(env, home.eef_pos, home.eef_quat)
    set_gripper(env, home.gripper, open_gripper_qpos)
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()
    return ik_ok


def resolve_home_pose(
    env,
    task_name: str,
    preset_path: Path | str | None,
    *,
    layout: int | None = None,
    style: int | None = None,
    seed: int | None = None,
    source_fixture: str | None = None,
    target_fixture: str | None = None,
) -> RobotHomePose:
    path = resolve_home_preset_path(task_name, preset_path)
    extra_checks: dict[str, Any] = {}
    if source_fixture is not None:
        extra_checks["source_fixture"] = source_fixture
    if target_fixture is not None:
        extra_checks["target_fixture"] = target_fixture

    if path is not None and path.exists():
        home = load_home_preset(path)
        print(colored(f"Loaded home preset: {path}", "cyan"))
        warn_home_mismatch(
            home,
            task_name=task_name,
            layout=layout,
            style=style,
            seed=seed,
            extra_checks=extra_checks or None,
        )
        return home

    display = path if path is not None else preset_path
    print(
        colored(
            f"warning: home preset not found ({display}); using reset pose snapshot",
            "yellow",
        )
    )
    return capture_home_from_env(
        env,
        task_name=task_name,
        name="reset_snapshot",
        layout=layout,
        style=style,
        seed=seed,
        source_fixture=source_fixture,
        target_fixture=target_fixture,
    )


__all__ = [
    "IK_MAX_ITERS",
    "IK_POS_TOL",
    "apply_home_pose",
    "capture_home_from_env",
    "compute_robot_base_near_fixture",
    "get_eef_pose",
    "get_robot_base_pose",
    "resolve_base_env",
    "resolve_home_pose",
    "resolve_robot",
    "save_home_preset",
    "set_gripper",
    "set_robot_base_pose",
    "solve_eef_pose",
]
