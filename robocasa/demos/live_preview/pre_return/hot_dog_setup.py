"""HotDogSetup: navigate to fridge, close door, then return home."""

from __future__ import annotations

import numpy as np
from termcolor import colored

from robocasa.demos.live_preview.fixture_transition import (
    build_door_joint_close_states,
    build_fridge_approach_snapshot,
    capture_anchor_state,
    needs_fridge_close,
    resolve_env_fixture,
)
from robocasa.demos.live_preview.robot_bridge import (
    RobotSnapshot,
    build_hold_states,
    build_robot_transition_states,
)
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

PHASE_GOTO_FRIDGE_BASE = 30
PHASE_CLOSE_FRIDGE = 20
PHASE_HOLD_AFTER_CLOSE = 10


def append_hot_dog_pre_return_states(
    env,
    end_anchor: np.ndarray,
    open_gripper_qpos: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    After demo: goto fridge -> close door -> brief hold.
    Returns extra states and the anchor state for return-home.
    """
    reset_to(env, {"states": np.asarray(end_anchor)})
    fridge = resolve_env_fixture(env, "fridge")

    if not needs_fridge_close(env, fridge):
        print(colored("Fridge already closed; skipping pre-return close phase", "cyan"))
        return [], np.asarray(end_anchor)

    print(colored("Pre-return: navigate to fridge and close door", "cyan"))
    task_end = RobotSnapshot.from_env(env)
    fridge_approach = build_fridge_approach_snapshot(
        env, end_anchor, fridge, open_gripper_qpos
    )

    extra: list[np.ndarray] = []
    extra.extend(
        build_robot_transition_states(
            env,
            end_anchor,
            task_end,
            fridge_approach,
            open_gripper_qpos,
            n_base=PHASE_GOTO_FRIDGE_BASE,
            n_arm=0,
            avoid_obstacles=True,
        )
    )

    goto_anchor = extra[-1] if extra else np.asarray(end_anchor)
    reset_to(env, {"states": goto_anchor})
    robot_at_fridge = RobotSnapshot.from_env(env)

    extra.extend(
        build_door_joint_close_states(
            env,
            goto_anchor,
            fridge,
            open_gripper_qpos,
            PHASE_CLOSE_FRIDGE,
            robot=robot_at_fridge,
        )
    )

    close_anchor = extra[-1] if extra else goto_anchor
    extra.extend(
        build_hold_states(
            env,
            close_anchor,
            robot_at_fridge,
            open_gripper_qpos,
            PHASE_HOLD_AFTER_CLOSE,
        )
    )
    if extra:
        close_anchor = capture_anchor_state(env)

    return extra, np.asarray(close_anchor)
