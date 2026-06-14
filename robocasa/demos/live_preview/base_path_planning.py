"""Collision-aware mobile-base path planning for live preview Home transitions."""

from __future__ import annotations

import heapq
from typing import Callable

import numpy as np
from termcolor import colored

from robocasa.demos.live_preview.timeline_utils import lerp_yaw
from robocasa.demos.live_preview.robot_session import resolve_base_env
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.utils.env_utils import detect_robot_collision

DEFAULT_GRID_STEP = 0.15
DETOUR_OFFSETS = (0.35, 0.55, 0.75, 1.0, 1.25)
MAX_ASTAR_NODES = 1500
STRAIGHT_SAMPLES = 16
PATH_VALIDATE_FRAMES = 10


def is_base_pose_collision_free(
    env,
    anchor_state: np.ndarray,
    base_pos: np.ndarray,
    base_ori: np.ndarray,
    arm,
    open_gripper_qpos: np.ndarray,
    *,
    reset_anchor: bool = True,
) -> bool:
    from robocasa.demos.live_preview.robot_bridge import (
        RobotSnapshot,
        apply_robot_snapshot,
    )

    if reset_anchor:
        reset_to(env, {"states": np.asarray(anchor_state)})
    robot = RobotSnapshot(
        base_pos=np.asarray(base_pos, dtype=float),
        base_ori=np.asarray(base_ori, dtype=float),
        eef_pos=arm.eef_pos.copy(),
        eef_quat=arm.eef_quat.copy(),
        gripper=arm.gripper,
    )
    apply_robot_snapshot(env, robot, open_gripper_qpos)
    base_env = resolve_base_env(env)
    return not detect_robot_collision(base_env)


class _BasePathCollisionChecker:
    """Reuse one anchor reset while probing many base poses."""

    def __init__(
        self,
        env,
        anchor_state: np.ndarray,
        start_ori: np.ndarray,
        end_ori: np.ndarray,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        z: float,
        arm,
        open_gripper_qpos: np.ndarray,
    ):
        self.env = env
        self.anchor_state = np.asarray(anchor_state)
        self.start_ori = start_ori
        self.end_ori = end_ori
        self.start_xy = start_xy
        self.goal_xy = goal_xy
        self.z = z
        self.arm = arm
        self.open_gripper_qpos = open_gripper_qpos
        self._cache: dict[tuple[float, float, float], bool] = {}
        self._anchor_ready = False

    def _ensure_anchor(self) -> None:
        if not self._anchor_ready:
            reset_to(self.env, {"states": self.anchor_state})
            self._anchor_ready = True

    def ori_at(self, progress: float) -> np.ndarray:
        return lerp_yaw(self.start_ori, self.end_ori, progress)

    def progress_for_xy(self, x: float, y: float) -> float:
        span = max(float(np.linalg.norm(self.goal_xy - self.start_xy)), 1e-9)
        progress = float(np.linalg.norm(np.array([x, y]) - self.start_xy) / span)
        return min(max(progress, 0.0), 1.0)

    def is_free_xy(self, x: float, y: float) -> bool:
        progress = self.progress_for_xy(x, y)
        key = (round(x, 2), round(y, 2), round(progress, 2))
        if key in self._cache:
            return self._cache[key]
        self._ensure_anchor()
        base_pos = np.array([x, y, self.z], dtype=float)
        ok = is_base_pose_collision_free(
            self.env,
            self.anchor_state,
            base_pos,
            self.ori_at(progress),
            self.arm,
            self.open_gripper_qpos,
            reset_anchor=False,
        )
        self._cache[key] = ok
        return ok

    def validate_waypoints(self, waypoints: list[np.ndarray], n_frames: int) -> bool:
        poses = sample_base_poses_along_waypoints(
            waypoints, self.start_ori, self.end_ori, n_frames
        )
        for base_pos, base_ori in poses:
            self._ensure_anchor()
            if not is_base_pose_collision_free(
                self.env,
                self.anchor_state,
                base_pos,
                base_ori,
                self.arm,
                self.open_gripper_qpos,
                reset_anchor=False,
            ):
                return False
        return True


def _sample_polyline(
    points: list[np.ndarray],
    n: int,
) -> list[np.ndarray]:
    if n <= 0:
        return []
    if len(points) < 2:
        return [points[-1].copy() for _ in range(n)]

    seg_lengths = [
        float(np.linalg.norm(points[i + 1][:2] - points[i][:2]))
        for i in range(len(points) - 1)
    ]
    total = sum(seg_lengths)
    if total < 1e-9:
        return [points[-1].copy() for _ in range(n)]

    cum = np.cumsum([0.0] + seg_lengths)
    out: list[np.ndarray] = []
    for i in range(n):
        target = total * (i + 1) / n
        seg_idx = int(np.searchsorted(cum, target, side="right") - 1)
        seg_idx = min(max(seg_idx, 0), len(points) - 2)
        seg_start = cum[seg_idx]
        seg_len = max(seg_lengths[seg_idx], 1e-9)
        alpha = (target - seg_start) / seg_len
        pos = points[seg_idx] + alpha * (points[seg_idx + 1] - points[seg_idx])
        out.append(np.asarray(pos, dtype=float))
    return out


def _astar_xy(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    is_free: Callable[[float, float], bool],
    *,
    step: float = DEFAULT_GRID_STEP,
    max_nodes: int = MAX_ASTAR_NODES,
) -> list[np.ndarray] | None:
    start = tuple(np.round(start_xy / step).astype(int))
    goal = tuple(np.round(goal_xy / step).astype(int))

    def h(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    open_heap: list[tuple[float, tuple[int, int]]] = [(h(start, goal), start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score = {start: 0.0}
    neighbors = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )
    expanded = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return [np.array([ix * step, iy * step], dtype=float) for ix, iy in path]

        expanded += 1
        if expanded > max_nodes:
            return None

        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            world = np.array([nxt[0] * step, nxt[1] * step], dtype=float)
            if not is_free(float(world[0]), float(world[1])):
                continue
            move_cost = float(np.hypot(dx, dy))
            tentative = g_score[current] + move_cost
            if tentative >= g_score.get(nxt, float("inf")):
                continue
            came_from[nxt] = current
            g_score[nxt] = tentative
            heapq.heappush(open_heap, (tentative + h(nxt, goal), nxt))
    return None


def _try_detour_xy(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    is_free: Callable[[float, float], bool],
) -> list[np.ndarray] | None:
    mid = 0.5 * (start_xy + goal_xy)
    delta = goal_xy - start_xy
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return None
    perp = np.array([-delta[1], delta[0]], dtype=float) / norm

    for dist in DETOUR_OFFSETS:
        for sign in (1.0, -1.0):
            via = mid + sign * dist * perp
            if not is_free(float(via[0]), float(via[1])):
                continue
            samples = (
                start_xy,
                0.5 * (start_xy + via),
                via,
                0.5 * (via + goal_xy),
                goal_xy,
            )
            if not all(is_free(float(p[0]), float(p[1])) for p in samples):
                continue
            return [
                np.array([start_xy[0], start_xy[1]], dtype=float),
                np.array([via[0], via[1]], dtype=float),
                np.array([goal_xy[0], goal_xy[1]], dtype=float),
            ]
    return None


def _waypoints_from_xy(
    xy_points: list[np.ndarray],
    z: float,
) -> list[np.ndarray]:
    return [np.array([p[0], p[1], z], dtype=float) for p in xy_points]


def plan_base_waypoints(
    env,
    anchor_state: np.ndarray,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    start_ori: np.ndarray,
    end_ori: np.ndarray,
    arm,
    open_gripper_qpos: np.ndarray,
    *,
    grid_step: float = DEFAULT_GRID_STEP,
    validate_frames: int = PATH_VALIDATE_FRAMES,
) -> list[np.ndarray]:
    """Plan collision-free base XY waypoints (3D positions, z from start)."""
    start_pos = np.asarray(start_pos, dtype=float)
    end_pos = np.asarray(end_pos, dtype=float)
    z = float(start_pos[2])
    start_xy = start_pos[:2]
    goal_xy = end_pos[:2]

    checker = _BasePathCollisionChecker(
        env,
        anchor_state,
        start_ori,
        end_ori,
        start_xy,
        goal_xy,
        z,
        arm,
        open_gripper_qpos,
    )

    if np.allclose(start_xy, goal_xy, atol=1e-4):
        return [start_pos.copy(), end_pos.copy()]

    def accept(candidate: list[np.ndarray], label: str) -> list[np.ndarray]:
        if checker.validate_waypoints(candidate, validate_frames):
            print(colored(f"Planned {label} base path for Home transition", "cyan"))
            return candidate
        return []

    straight = [start_pos.copy(), end_pos.copy()]
    straight_ok = all(
        checker.is_free_xy(float(xy[0]), float(xy[1]))
        for xy in (
            start_xy + t * (goal_xy - start_xy)
            for t in np.linspace(0.0, 1.0, STRAIGHT_SAMPLES)
        )
    )
    if straight_ok:
        accepted = accept(straight, "straight")
        if accepted:
            return accepted

    detour_xy = _try_detour_xy(start_xy, goal_xy, checker.is_free_xy)
    if detour_xy is not None:
        accepted = accept(_waypoints_from_xy(detour_xy, z), "detour")
        if accepted:
            return accepted

    pad = max(float(np.linalg.norm(goal_xy - start_xy)) + 0.8, 1.0)
    min_xy = np.minimum(start_xy, goal_xy) - pad
    max_xy = np.maximum(start_xy, goal_xy) + pad

    def grid_free(x: float, y: float) -> bool:
        if x < min_xy[0] or x > max_xy[0] or y < min_xy[1] or y > max_xy[1]:
            return False
        return checker.is_free_xy(x, y)

    astar_xy = _astar_xy(start_xy, goal_xy, grid_free, step=grid_step)
    if astar_xy is not None:
        accepted = accept(_waypoints_from_xy(astar_xy, z), "A*")
        if accepted:
            return accepted

    for finer_step in (grid_step * 0.75, grid_step * 0.5):
        astar_xy = _astar_xy(
            start_xy,
            goal_xy,
            grid_free,
            step=finer_step,
            max_nodes=MAX_ASTAR_NODES * 2,
        )
        if astar_xy is None:
            continue
        accepted = accept(_waypoints_from_xy(astar_xy, z), "A*")
        if accepted:
            return accepted

    print(
        colored(
            "warning: could not find collision-free base path; falling back to straight line",
            "yellow",
        )
    )
    return straight


def sample_base_poses_along_waypoints(
    waypoints: list[np.ndarray],
    start_ori: np.ndarray,
    end_ori: np.ndarray,
    n_frames: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    positions = _sample_polyline(waypoints, n_frames)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i, pos in enumerate(positions):
        t = (i + 1) / max(n_frames, 1)
        out.append((pos, lerp_yaw(start_ori, end_ori, t)))
    return out
