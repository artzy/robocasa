"""Return-to-home timeline segment for live preview demos."""

from __future__ import annotations

from typing import Callable

import numpy as np

from robocasa.demos.live_preview.timeline_utils import lerp_vec, lerp_yaw
from robocasa.demos.live_preview.base_path_planning import (
    sample_base_poses_along_waypoints,
)

PHASE_RETURN_BASE = 30
PHASE_RETURN_ARM = 20
PHASE_HOLD_END = 30


def append_return_home_frames(
    add_frame: Callable[..., None],
    *,
    retreat_target: np.ndarray,
    home_eef_pos: np.ndarray,
    home_eef_quat: np.ndarray,
    home_base_pos: np.ndarray,
    home_base_ori: np.ndarray,
    place_base_pos: np.ndarray,
    place_base_ori: np.ndarray,
    end_pan: np.ndarray,
    end_quat: np.ndarray,
    gripper: float = 0.0,
    attach: bool = False,
    base_waypoints: list | None = None,
) -> None:
    """Append base return, arm return, and hold-at-home frames via ``add_frame``."""
    if base_waypoints is not None:
        base_poses = sample_base_poses_along_waypoints(
            base_waypoints,
            place_base_ori,
            home_base_ori,
            PHASE_RETURN_BASE,
        )
        for base_pos, base_ori in base_poses:
            add_frame(
                eef_pos=retreat_target.copy(),
                eef_quat=home_eef_quat.copy(),
                gripper=gripper,
                pan_pos=end_pan.copy(),
                pan_quat=end_quat.copy(),
                attach=attach,
                base_pos=base_pos,
                base_ori=base_ori,
            )
    else:
        for i in range(PHASE_RETURN_BASE):
            t = (i + 1) / PHASE_RETURN_BASE
            add_frame(
                eef_pos=retreat_target.copy(),
                eef_quat=home_eef_quat.copy(),
                gripper=gripper,
                pan_pos=end_pan.copy(),
                pan_quat=end_quat.copy(),
                attach=attach,
                base_pos=lerp_vec(place_base_pos, home_base_pos, t),
                base_ori=lerp_yaw(place_base_ori, home_base_ori, t),
            )

    for i in range(PHASE_RETURN_ARM):
        t = (i + 1) / PHASE_RETURN_ARM
        add_frame(
            eef_pos=lerp_vec(retreat_target, home_eef_pos, t),
            eef_quat=home_eef_quat.copy(),
            gripper=gripper,
            pan_pos=end_pan.copy(),
            pan_quat=end_quat.copy(),
            attach=attach,
            base_pos=home_base_pos.copy(),
            base_ori=home_base_ori.copy(),
        )

    for _ in range(PHASE_HOLD_END):
        add_frame(
            eef_pos=home_eef_pos.copy(),
            eef_quat=home_eef_quat.copy(),
            gripper=gripper,
            pan_pos=end_pan.copy(),
            pan_quat=end_quat.copy(),
            attach=attach,
            base_pos=home_base_pos.copy(),
            base_ori=home_base_ori.copy(),
        )
