"""
Table sandbox: move coppelia_frying_pan from a fixed start XY to a goal region.

Usage:
    python -m robocasa.demos.demo_pan_move_sandbox
    python -m robocasa.demos.demo_pan_move_sandbox --start-x 0.2 --goal-x -0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from termcolor import colored

import robocasa
from robocasa.models.objects.objects import MJCFObject
from robocasa.scripts.collect_demos import collect_human_trajectory
from robocasa.utils.model_zoo.object_play_env import ObjectPlayEnv
from robocasa.utils.playback_viewer import (
    PygamePlaybackViewer,
    robosuite_viewer_kwargs,
)


DEFAULT_PAN_MJCF = (
    Path(robocasa.models.assets_root)
    / "objects"
    / "coppelia_edu"
    / "kitchenware"
    / "coppelia_frying_pan"
    / "model.xml"
)


class PanMoveSandboxEnv(ObjectPlayEnv):
    """ObjectPlayEnv with fixed start position and XY goal region success check."""

    def __init__(
        self,
        start_x: float = 0.15,
        start_y: float = 0.0,
        goal_x: float = -0.15,
        goal_y: float = 0.0,
        goal_radius: float = 0.08,
        **kwargs,
    ):
        self._goal_xy = np.array([goal_x, goal_y], dtype=float)
        self._goal_radius = goal_radius
        self._pan_obj = None
        super().__init__(
            x_range=(start_x, start_x),
            y_range=(start_y, start_y),
            rotation=(0.0, 0.0),
            **kwargs,
        )

    def _load_model(self):
        super()._load_model()
        for obj in self.model.mujoco_objects:
            if isinstance(obj, MJCFObject):
                self._pan_obj = obj
                break

    def _check_success(self):
        if self._pan_obj is None:
            return False
        body_id = self.sim.model.body_name2id(self._pan_obj.root_body)
        pan_xy = self.sim.data.body_xpos[body_id][:2] - self.table_offset[:2]
        in_goal = np.linalg.norm(pan_xy - self._goal_xy) <= self._goal_radius
        gripper_site = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        pan_pos = self.sim.data.body_xpos[body_id]
        gripper_far = np.linalg.norm(gripper_site - pan_pos) > 0.25
        return in_goal and gripper_far


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pan move sandbox on table.")
    parser.add_argument(
        "--mjcf-path",
        type=str,
        default=str(DEFAULT_PAN_MJCF),
        help="Path to pan model.xml",
    )
    parser.add_argument("--start-x", type=float, default=0.15)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--goal-x", type=float, default=-0.15)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-radius", type=float, default=0.08)
    parser.add_argument(
        "--device",
        type=str,
        default="keyboard",
        choices=["keyboard", "spacemouse"],
    )
    args = parser.parse_args()

    mjcf_path = Path(args.mjcf_path)
    if not mjcf_path.exists():
        raise FileNotFoundError(
            f"Missing {mjcf_path}. Run:\n"
            "  python robocasa/scripts/asset_scripts/import_coppelia_batch.py"
        )

    controller_config = load_composite_controller_config(robot="Panda")
    onscreen_renderer, viewer_kwargs = robosuite_viewer_kwargs(render_camera="agentview")

    print(colored("Initializing pan move sandbox...", "yellow"))
    env = PanMoveSandboxEnv(
        robots="Panda",
        controller_configs=controller_config,
        obj_mjcf_path=str(mjcf_path),
        start_x=args.start_x,
        start_y=args.start_y,
        goal_x=args.goal_x,
        goal_y=args.goal_y,
        goal_radius=args.goal_radius,
        has_renderer=viewer_kwargs["has_renderer"],
        has_offscreen_renderer=viewer_kwargs["has_offscreen_renderer"],
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        render_camera="agentview",
        renderer=viewer_kwargs["renderer"],
    )
    env = VisualizationWrapper(env)

    pygame_viewer = None
    if onscreen_renderer == "pygame":
        pygame_viewer = PygamePlaybackViewer(
            camera_name="agentview",
            width=768,
            height=512,
            title="Pan Move Sandbox",
        )
        env._pygame_viewer = pygame_viewer

    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env=env, pos_sensitivity=4.0, rot_sensitivity=4.0)
    else:
        from robosuite.devices import SpaceMouse

        import robocasa.macros as macros

        device = SpaceMouse(
            env=env,
            pos_sensitivity=4.0,
            rot_sensitivity=4.0,
            vendor_id=macros.SPACEMOUSE_VENDOR_ID,
            product_id=macros.SPACEMOUSE_PRODUCT_ID,
        )

    print(
        colored(
            f"Start=({args.start_x}, {args.start_y})  "
            f"Goal=({args.goal_x}, {args.goal_y}) r={args.goal_radius}",
            "cyan",
        )
    )

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
