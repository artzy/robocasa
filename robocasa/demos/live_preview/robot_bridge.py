"""Robot pose snapshots and bridge states for home-wrapped demo playback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from robocasa.demos.live_preview.base_path_planning import (
    plan_base_waypoints,
    sample_base_poses_along_waypoints,
)
from robocasa.demos.live_preview.return_home import (
    PHASE_HOLD_END,
    PHASE_RETURN_ARM,
    PHASE_RETURN_BASE,
)
from robocasa.demos.live_preview.robot_session import (
    apply_home_pose,
    get_eef_pose,
    get_robot_base_pose,
    set_gripper,
    set_robot_base_pose,
    solve_eef_pose,
)
from robocasa.demos.live_preview.timeline_utils import lerp_scalar, lerp_vec, lerp_yaw
from robocasa.demos.live_preview.home_pose import RobotHomePose
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

PHASE_HOLD_START = 30
PHASE_LEAVE_BASE = PHASE_RETURN_BASE
PHASE_LEAVE_ARM = PHASE_RETURN_ARM

PreReturnFn = Callable[
    [object, np.ndarray, np.ndarray],
    tuple[list[np.ndarray], np.ndarray],
]


@dataclass
class RobotSnapshot:
    base_pos: np.ndarray
    base_ori: np.ndarray
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    gripper: float = 0.0

    @classmethod
    def from_home(cls, home: RobotHomePose) -> RobotSnapshot:
        return cls(
            base_pos=home.base_pos.copy(),
            base_ori=home.base_ori.copy(),
            eef_pos=home.eef_pos.copy(),
            eef_quat=home.eef_quat.copy(),
            gripper=float(home.gripper),
        )

    @classmethod
    def from_env(cls, env) -> RobotSnapshot:
        base_pos, base_ori = get_robot_base_pose(env)
        eef_pos, eef_quat, _ = get_eef_pose(env)
        return cls(
            base_pos=base_pos.copy(),
            base_ori=base_ori.copy(),
            eef_pos=eef_pos.copy(),
            eef_quat=eef_quat.copy(),
            gripper=0.0,
        )


def apply_robot_snapshot(
    env,
    robot: RobotSnapshot,
    open_gripper_qpos: np.ndarray,
) -> None:
    set_robot_base_pose(env, robot.base_pos, robot.base_ori)
    solve_eef_pose(env, robot.eef_pos, robot.eef_quat)
    set_gripper(env, robot.gripper, open_gripper_qpos)
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()


def capture_state_at_robot_pose(
    env,
    anchor_state: np.ndarray,
    robot: RobotSnapshot,
    open_gripper_qpos: np.ndarray,
    *,
    robot_snapshot: RobotSnapshot | None = None,
) -> np.ndarray:
    reset_to(env, {"states": np.asarray(anchor_state)})
    target = robot_snapshot if robot_snapshot is not None else robot
    apply_robot_snapshot(env, target, open_gripper_qpos)
    return np.array(env.sim.get_state().flatten())


def build_hold_states(
    env,
    anchor_state: np.ndarray,
    robot: RobotSnapshot,
    open_gripper_qpos: np.ndarray,
    n_frames: int,
    *,
    robot_snapshot: RobotSnapshot | None = None,
) -> list[np.ndarray]:
    if n_frames <= 0:
        return []
    state = capture_state_at_robot_pose(
        env,
        anchor_state,
        robot,
        open_gripper_qpos,
        robot_snapshot=robot_snapshot,
    )
    return [state.copy() for _ in range(n_frames)]


def build_robot_transition_states(
    env,
    anchor_state: np.ndarray,
    start: RobotSnapshot,
    end: RobotSnapshot,
    open_gripper_qpos: np.ndarray,
    *,
    n_base: int = PHASE_LEAVE_BASE,
    n_arm: int = PHASE_LEAVE_ARM,
    avoid_obstacles: bool = True,
) -> list[np.ndarray]:
    """Interpolate robot from ``start`` to ``end`` while keeping the scene anchor fixed."""
    states: list[np.ndarray] = []

    if avoid_obstacles and n_base > 0:
        waypoints = plan_base_waypoints(
            env,
            anchor_state,
            start.base_pos,
            end.base_pos,
            start.base_ori,
            end.base_ori,
            start,
            open_gripper_qpos,
        )
        base_poses = sample_base_poses_along_waypoints(
            waypoints, start.base_ori, end.base_ori, n_base
        )
        for base_pos, base_ori in base_poses:
            robot = RobotSnapshot(
                base_pos=base_pos,
                base_ori=base_ori,
                eef_pos=start.eef_pos.copy(),
                eef_quat=start.eef_quat.copy(),
                gripper=start.gripper,
            )
            states.append(
                capture_state_at_robot_pose(
                    env, anchor_state, robot, open_gripper_qpos
                )
            )
    else:
        for i in range(n_base):
            t = (i + 1) / n_base
            robot = RobotSnapshot(
                base_pos=lerp_vec(start.base_pos, end.base_pos, t),
                base_ori=lerp_yaw(start.base_ori, end.base_ori, t),
                eef_pos=start.eef_pos.copy(),
                eef_quat=start.eef_quat.copy(),
                gripper=lerp_scalar(start.gripper, end.gripper, t),
            )
            states.append(
                capture_state_at_robot_pose(
                    env, anchor_state, robot, open_gripper_qpos
                )
            )

    for i in range(n_arm):
        t = (i + 1) / n_arm
        robot = RobotSnapshot(
            base_pos=end.base_pos.copy(),
            base_ori=end.base_ori.copy(),
            eef_pos=lerp_vec(start.eef_pos, end.eef_pos, t),
            eef_quat=end.eef_quat.copy(),
            gripper=lerp_scalar(start.gripper, end.gripper, t),
        )
        states.append(
            capture_state_at_robot_pose(
                env, anchor_state, robot, open_gripper_qpos
            )
        )

    return states


def build_home_wrapped_demo_states(
    env,
    demo_states: np.ndarray,
    home: RobotHomePose,
    open_gripper_qpos: np.ndarray,
    *,
    extend_demo_tail: int = 0,
    pre_return_fn: PreReturnFn | None = None,
) -> list[np.ndarray]:
    """Home hold -> leave home -> human demo -> [pre-return] -> return home -> home hold."""
    demo_states = np.asarray(demo_states)
    if demo_states.ndim != 2 or demo_states.shape[0] == 0:
        raise ValueError("demo_states must be a non-empty 2D array")

    start_anchor = demo_states[0]
    end_anchor = demo_states[-1]

    reset_to(env, {"states": start_anchor})
    task_start = RobotSnapshot.from_env(env)
    home_robot = RobotSnapshot.from_home(home)

    reset_to(env, {"states": start_anchor})
    apply_robot_snapshot(env, home_robot, open_gripper_qpos)
    home_robot_resolved = RobotSnapshot.from_env(env)

    wrapped: list[np.ndarray] = []
    wrapped.extend(
        build_hold_states(
            env,
            start_anchor,
            home_robot,
            open_gripper_qpos,
            PHASE_HOLD_START,
        )
    )
    wrapped.extend(
        build_robot_transition_states(
            env,
            start_anchor,
            home_robot,
            task_start,
            open_gripper_qpos,
        )
    )
    wrapped.extend(list(demo_states))
    if extend_demo_tail > 0:
        tail = demo_states[-1]
        wrapped.extend([tail.copy() for _ in range(extend_demo_tail)])

    return_anchor = end_anchor
    if pre_return_fn is not None:
        pre_return_states, return_anchor = pre_return_fn(
            env, end_anchor, open_gripper_qpos
        )
        wrapped.extend(pre_return_states)

    reset_to(env, {"states": return_anchor})
    task_end_for_return = RobotSnapshot.from_env(env)
    return_states = build_robot_transition_states(
        env,
        return_anchor,
        task_end_for_return,
        home_robot_resolved,
        open_gripper_qpos,
    )
    wrapped.extend(return_states)
    if return_states:
        reset_to(env, {"states": return_states[-1]})
        apply_home_pose(env, home, open_gripper_qpos)
        hold_state = np.array(env.sim.get_state().flatten())
        wrapped.extend([hold_state.copy() for _ in range(PHASE_HOLD_END)])
    else:
        reset_to(env, {"states": return_anchor})
        apply_home_pose(env, home, open_gripper_qpos)
        home_at_return = RobotSnapshot.from_env(env)
        wrapped.extend(
            build_hold_states(
                env,
                return_anchor,
                home_robot,
                open_gripper_qpos,
                PHASE_HOLD_END,
                robot_snapshot=home_at_return,
            )
        )
    return wrapped


def apply_home_to_env(
    env,
    home: RobotHomePose,
    anchor_state: np.ndarray,
    open_gripper_qpos: np.ndarray,
) -> None:
    reset_to(env, {"states": np.asarray(anchor_state)})
    apply_home_pose(env, home, open_gripper_qpos)
