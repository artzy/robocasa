"""Shared MovePan live preview and teleop helpers for demo_tasks and demo_move_pan."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import imageio
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.parts.arm.ik import InverseKinematicsController
from robosuite.wrappers import VisualizationWrapper
from termcolor import colored

import robocasa  # noqa: F401 — register environments
import robocasa.macros as macros
from robocasa.models.fixtures.fixture import FixtureType
from robocasa.models.fixtures.fixture_utils import fixture_is_type
from robocasa.scripts.collect_demos import collect_human_trajectory
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.utils.playback_viewer import (
    PygamePlaybackViewer,
    apply_mjviewer_camera_config,
    get_layout_camera_config,
    onscreen_renderer_name,
    render_free_camera,
    robosuite_viewer_kwargs,
)
from robocasa.wrappers.enclosing_wall_render_wrapper import (
    EnclosingWallRenderWrapper,
    install_enclosing_wall_hotkeys,
)

CAMERA_NAME_TELEOP = "robot0_frontview"
CAMERA_WIDTH = 768
CAMERA_HEIGHT = 512
PREVIEW_FPS = 20

PREVIEW_DEFAULT_LAYOUT = 15
PREVIEW_DEFAULT_STYLE = 34
PREVIEW_DEFAULT_SEED = 0
PREVIEW_MAX_CAMERA_DISTANCE = 5.0

GRASP_Z_OFFSET = 0.03
APPROACH_Z = 0.12
LIFT_Z = 0.15
PLACE_Z = 0.05
RETREAT_Z = 0.12
# Pan center relative to gripper site in EEF frame (meters).
GRASP_LOCAL_OFFSET = np.array([0.0, 0.0, -0.08])
GRASP_ATTACH_GRIPPER = 0.85
MAX_PAN_EEF_ATTACH_DIST = 0.12

PHASE_HOLD_START = 15
PHASE_APPROACH = 20
PHASE_DESCEND = 15
PHASE_LIFT = 15
PHASE_TRANSPORT = 50
PHASE_PLACE = 15
PHASE_RETREAT = 10
PHASE_HOLD_END = 50

IK_POS_TOL = 0.015
IK_MAX_ITERS = 40


def parse_registries(value: str) -> tuple[str, ...]:
    parts = tuple(p.strip() for p in value.split(",") if p.strip())
    return parts or ("coppelia_edu",)


def _preview_env_defaults(
    teleop: bool,
    layout: int | None,
    style: int | None,
    seed: int,
) -> tuple[int | None, int | None, int]:
    if teleop:
        return layout, style, seed
    return (
        PREVIEW_DEFAULT_LAYOUT if layout is None else layout,
        PREVIEW_DEFAULT_STYLE if style is None else style,
        PREVIEW_DEFAULT_SEED if seed is None else seed,
    )


def _compute_preview_cam_config(env):
    """Layout overview camera centered on source→target pan motion."""
    config = get_layout_camera_config(env)
    try:
        start_pan, start_quat = _get_pan_pose(env)
        end_pan, _ = _compute_target_pan_pose(env, start_pan, start_quat)
        config["lookat"] = ((start_pan + end_pan) / 2.0).tolist()
    except Exception:
        pass
    # Offscreen free-camera rendering breaks above ~5.5m (blank gray frame on Windows/pygame).
    config["distance"] = min(
        float(config["distance"]), PREVIEW_MAX_CAMERA_DISTANCE
    )
    return config


def make_move_pan_env(
    source_fixture: str = "counter",
    target_fixture: str = "sink",
    obj_registries: tuple[str, ...] = ("coppelia_edu",),
    layout: int | None = None,
    style: int | None = None,
    seed: int = 0,
    *,
    teleop: bool = False,
    render_offscreen: bool = False,
):
    """Create MovePan env with optional pygame viewer."""
    layout, style, seed = _preview_env_defaults(teleop, layout, style, seed)
    config = {
        "env_name": "MovePan",
        "robots": "PandaOmron",
        "controller_configs": load_composite_controller_config(robot="PandaOmron"),
        "source_fixture": source_fixture,
        "target_fixture": target_fixture,
        "obj_registries": obj_registries,
        "layout_ids": layout,
        "style_ids": style,
        "translucent_robot": teleop,
        "seed": seed,
    }

    render_camera_kw = {}
    if render_offscreen:
        viewer_kwargs = {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "renderer": "mjviewer",
        }
        onscreen_renderer = "offscreen"
        pygame_viewer = None
        render_camera_kw = {"render_camera": None}
    elif teleop:
        onscreen_renderer, viewer_kwargs = robosuite_viewer_kwargs(
            render_camera=CAMERA_NAME_TELEOP
        )
        pygame_viewer = None
        if onscreen_renderer == "pygame":
            pygame_viewer = PygamePlaybackViewer(
                camera_name=CAMERA_NAME_TELEOP,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                title="RoboCasa Teleop",
            )
        if "render_camera" in viewer_kwargs:
            render_camera_kw = {"render_camera": viewer_kwargs["render_camera"]}
    else:
        # Preview uses a kitchen overview free camera so counter→sink motion stays in frame.
        onscreen_renderer = "mjviewer"
        viewer_kwargs = {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "renderer": "mjviewer",
        }
        render_camera_kw = {"render_camera": None}
        pygame_viewer = None
        if onscreen_renderer_name() == "pygame":
            onscreen_renderer = "pygame"
            pygame_viewer = PygamePlaybackViewer(
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                title="RoboCasa",
            )

    print(colored("Initializing MovePan environment...", "yellow"))
    env = robosuite.make(
        **config,
        has_renderer=viewer_kwargs["has_renderer"],
        has_offscreen_renderer=viewer_kwargs["has_offscreen_renderer"],
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer=viewer_kwargs["renderer"],
        **render_camera_kw,
    )

    if pygame_viewer is not None:
        print(colored("Opening viewer (pygame window)...", "yellow"))

    env = VisualizationWrapper(env) if teleop else env
    env = EnclosingWallRenderWrapper(env, alpha=0.1, enabled=False)
    install_enclosing_wall_hotkeys(env)
    if pygame_viewer is not None:
        env._pygame_viewer = pygame_viewer

    json.dumps(config)
    return env, pygame_viewer, onscreen_renderer


def make_input_device(env, device: str = "keyboard"):
    if device == "keyboard":
        from robosuite.devices import Keyboard

        return Keyboard(env=env, pos_sensitivity=4.0, rot_sensitivity=4.0)

    from robosuite.devices import SpaceMouse

    return SpaceMouse(
        env=env,
        pos_sensitivity=4.0,
        rot_sensitivity=4.0,
        vendor_id=macros.SPACEMOUSE_VENDOR_ID,
        product_id=macros.SPACEMOUSE_PRODUCT_ID,
    )


def _resolve_base_env(env):
    cur = env
    while cur is not None:
        if hasattr(cur, "robots") and hasattr(cur, "sim"):
            return cur
        cur = getattr(cur, "env", None)
    return env


def _resolve_robot(env):
    return _resolve_base_env(env).robots[0]


def _print_instruction(env):
    ep_meta = env.get_ep_meta()
    if isinstance(ep_meta, str):
        ep_meta = json.loads(ep_meta)
    lang = ep_meta.get("lang")
    if lang:
        print(colored(f"Instruction: {lang}", "green"))
    print(colored("Spawning environment...", "yellow"))


def _smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _lerp_vec(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return a + _smoothstep(alpha) * (b - a)


def _lerp_scalar(a: float, b: float, alpha: float) -> float:
    return float(a + _smoothstep(alpha) * (b - a))


def _get_pan_pose(env):
    body_id = env.obj_body_id["pan"]
    pos = np.array(env.sim.data.body_xpos[body_id], dtype=float)
    quat = np.array(env.sim.data.body_xquat[body_id], dtype=float)
    return pos, quat


def _set_pan_pose(env, pos, quat):
    pan = env.objects["pan"]
    env.sim.data.set_joint_qpos(pan.joints[0], np.concatenate([pos, quat]))
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()


def _get_eef_pose(env):
    robot = _resolve_robot(env)
    site_id = robot.eef_site_id["right"]
    pos = np.array(env.sim.data.site_xpos[site_id], dtype=float)
    mat = env.sim.data.site_xmat[site_id].reshape(3, 3)
    quat_xyzw = T.mat2quat(mat)
    quat = T.convert_quat(quat_xyzw, to="wxyz")
    return pos, quat, mat


def _eef_local_to_world(eef_pos: np.ndarray, eef_mat: np.ndarray, local_offset: np.ndarray):
    return eef_pos + eef_mat @ local_offset


def _snap_pan_to_gripper(env, pan_quat: np.ndarray, local_offset: np.ndarray | None = None):
    """Place pan at a fixed offset from the gripper site (grasp snap)."""
    if local_offset is None:
        local_offset = GRASP_LOCAL_OFFSET.copy()
    eef_pos, _, eef_mat = _get_eef_pose(env)
    pan_pos = _eef_local_to_world(eef_pos, eef_mat, local_offset)
    _set_pan_pose(env, pan_pos, pan_quat)
    return local_offset.copy()


def _arm_qpos_indices(env):
    robot = _resolve_robot(env)
    split = robot._joint_split_idx
    return np.array(robot._ref_arm_joint_pos_indexes[:split], dtype=int)


def _eef_site_name(env) -> str:
    robot = _resolve_robot(env)
    site_id = robot.eef_site_id["right"]
    return env.sim.model.site(site_id).name


def _set_gripper(env, closed: float, open_qpos: np.ndarray):
    robot = _resolve_robot(env)
    idx = robot._ref_gripper_joint_pos_indexes["right"]
    closed = float(np.clip(closed, 0.0, 1.0))
    env.sim.data.qpos[idx] = open_qpos * (1.0 - closed)


def _solve_eef_pose(
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
    site_id = _resolve_robot(env).eef_site_id["right"]

    tgt_mat = T.quat2mat(T.convert_quat(target_quat, to="xyzw"))

    for _ in range(iters):
        cur_pos = np.array(sim.data.site_xpos[site_id], dtype=float)
        dpos = target_pos - cur_pos
        if np.linalg.norm(dpos) < pos_tol:
            return True

        cur_quat_xyzw = T.mat2quat(
            sim.data.site_xmat[site_id].reshape(3, 3)
        )
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


def _compute_target_pan_pose(env, start_pos, start_quat):
    end_pos = start_pos.copy()
    target = env.target

    try:
        regions = target.get_reset_regions(env=env)
        if regions:
            region = next(iter(regions.values()))
            end_pos = np.array(region["pos"], dtype=float)
            end_pos[2] = float(region["pos"][2]) + 0.02
    except Exception:
        end_pos[:2] = np.array(target.pos[:2], dtype=float)

    if fixture_is_type(target, FixtureType.SINK):
        end_pos[2] = min(end_pos[2], start_pos[2] - 0.03)
    elif fixture_is_type(target, (FixtureType.COUNTER, FixtureType.STOVE)):
        end_pos[2] = start_pos[2]

    return end_pos, start_quat


@dataclass
class _PreviewFrame:
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    gripper: float
    pan_pos: np.ndarray
    pan_quat: np.ndarray
    attach: bool


def _build_move_pan_timeline(
    start_pan: np.ndarray,
    end_pan: np.ndarray,
    start_quat: np.ndarray,
    end_quat: np.ndarray,
    home_eef_pos: np.ndarray,
    home_eef_quat: np.ndarray,
) -> list[_PreviewFrame]:
    frames: list[_PreviewFrame] = []

    approach_target = start_pan + np.array([0.0, 0.0, APPROACH_Z])
    descend_target = start_pan + np.array([0.0, 0.0, GRASP_Z_OFFSET])
    lift_target = start_pan + np.array([0.0, 0.0, LIFT_Z])
    end_hover = end_pan + np.array([0.0, 0.0, LIFT_Z])
    place_down = end_pan + np.array([0.0, 0.0, PLACE_Z])
    retreat_target = end_pan + np.array([0.0, 0.0, RETREAT_Z])

    def add_frame(**kwargs):
        frames.append(_PreviewFrame(**kwargs))

    for _ in range(PHASE_HOLD_START):
        add_frame(
            eef_pos=home_eef_pos.copy(),
            eef_quat=home_eef_quat.copy(),
            gripper=0.0,
            pan_pos=start_pan.copy(),
            pan_quat=start_quat.copy(),
            attach=False,
        )

    for i in range(PHASE_APPROACH):
        t = (i + 1) / PHASE_APPROACH
        add_frame(
            eef_pos=_lerp_vec(home_eef_pos, approach_target, t),
            eef_quat=home_eef_quat.copy(),
            gripper=0.0,
            pan_pos=start_pan.copy(),
            pan_quat=start_quat.copy(),
            attach=False,
        )

    for i in range(PHASE_DESCEND):
        t = (i + 1) / PHASE_DESCEND
        gripper = _lerp_scalar(0.0, 1.0, t)
        add_frame(
            eef_pos=_lerp_vec(approach_target, descend_target, t),
            eef_quat=home_eef_quat.copy(),
            gripper=gripper,
            pan_pos=start_pan.copy(),
            pan_quat=start_quat.copy(),
            attach=gripper >= GRASP_ATTACH_GRIPPER,
        )

    for i in range(PHASE_LIFT):
        t = (i + 1) / PHASE_LIFT
        add_frame(
            eef_pos=_lerp_vec(descend_target, lift_target, t),
            eef_quat=home_eef_quat.copy(),
            gripper=1.0,
            pan_pos=start_pan.copy(),
            pan_quat=start_quat.copy(),
            attach=True,
        )

    for i in range(PHASE_TRANSPORT):
        t = (i + 1) / PHASE_TRANSPORT
        add_frame(
            eef_pos=_lerp_vec(lift_target, end_hover, t),
            eef_quat=home_eef_quat.copy(),
            gripper=1.0,
            pan_pos=end_pan.copy(),
            pan_quat=end_quat.copy(),
            attach=True,
        )

    for i in range(PHASE_PLACE):
        t = (i + 1) / PHASE_PLACE
        add_frame(
            eef_pos=_lerp_vec(end_hover, place_down, t),
            eef_quat=home_eef_quat.copy(),
            gripper=_lerp_scalar(1.0, 0.0, t),
            pan_pos=end_pan.copy(),
            pan_quat=end_quat.copy(),
            attach=t < 0.55,
        )

    for i in range(PHASE_RETREAT):
        t = (i + 1) / PHASE_RETREAT
        add_frame(
            eef_pos=_lerp_vec(place_down, retreat_target, t),
            eef_quat=home_eef_quat.copy(),
            gripper=0.0,
            pan_pos=end_pan.copy(),
            pan_quat=end_quat.copy(),
            attach=False,
        )

    for _ in range(PHASE_HOLD_END):
        add_frame(
            eef_pos=retreat_target.copy(),
            eef_quat=home_eef_quat.copy(),
            gripper=0.0,
            pan_pos=end_pan.copy(),
            pan_quat=end_quat.copy(),
            attach=False,
        )

    return frames


def _apply_preview_frame(
    env,
    frame: _PreviewFrame,
    *,
    open_gripper_qpos: np.ndarray,
    pan_local_offset: np.ndarray | None,
    ik_warned: list[bool],
    degraded: list[bool],
) -> np.ndarray | None:
    ik_ok = _solve_eef_pose(env, frame.eef_pos, frame.eef_quat)
    if not ik_ok:
        degraded[0] = True
        if not ik_warned[0]:
            print(
                colored(
                    "warning: IK did not fully converge; continuing with best-effort pose",
                    "yellow",
                )
            )
            ik_warned[0] = True

    _set_gripper(env, frame.gripper, open_gripper_qpos)

    if frame.attach:
        if pan_local_offset is None:
            pan_local_offset = _snap_pan_to_gripper(env, frame.pan_quat)
        else:
            eef_pos, _, eef_mat = _get_eef_pose(env)
            pan_pos = _eef_local_to_world(eef_pos, eef_mat, pan_local_offset)
            _set_pan_pose(env, pan_pos, frame.pan_quat)
    else:
        _set_pan_pose(env, frame.pan_pos, frame.pan_quat)
        if frame.gripper < 0.05:
            pan_local_offset = None

    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()
    return pan_local_offset


def _generate_preview_states(env):
    """Build scripted pick-place trajectory with arm IK for demo playback."""
    env.reset()
    _print_instruction(env)

    start_pan, start_quat = _get_pan_pose(env)
    end_pan, end_quat = _compute_target_pan_pose(env, start_pan, start_quat)
    home_eef_pos, home_eef_quat, _ = _get_eef_pose(env)
    open_gripper_qpos = _resolve_robot(env).get_gripper_joint_positions("right").copy()

    timeline = _build_move_pan_timeline(
        start_pan, end_pan, start_quat, end_quat, home_eef_pos, home_eef_quat
    )

    states = []
    pan_local_offset = None
    ik_warned = [False]
    degraded = [False]

    for frame in timeline:
        pan_local_offset = _apply_preview_frame(
            env,
            frame,
            open_gripper_qpos=open_gripper_qpos,
            pan_local_offset=pan_local_offset,
            ik_warned=ik_warned,
            degraded=degraded,
        )
        states.append(np.array(env.sim.get_state().flatten()))

    if degraded[0]:
        print(
            colored(
                "MovePan preview used partial IK convergence (arm motion may be limited).",
                "yellow",
            )
        )

    return np.array(states)


def _play_state_sequence(
    env,
    states,
    *,
    pygame_viewer=None,
    video_writer=None,
    cam_config=None,
):
    print(colored("Playing back episode: MovePan scripted demo", "yellow"))
    for state in states:
        start = time.time()
        if pygame_viewer is not None and not pygame_viewer.pump_events():
            break

        reset_to(env, {"states": state})

        if pygame_viewer is not None:
            if not pygame_viewer.update(env.sim):
                break
        elif video_writer is not None:
            frame = render_free_camera(
                env.sim, CAMERA_WIDTH, CAMERA_HEIGHT, cam_config
            )
            video_writer.append_data(frame)
        elif env.renderer == "mjviewer":
            if env.viewer is None:
                env.initialize_renderer()
            if cam_config is not None:
                apply_mjviewer_camera_config(env, cam_config)
            env.viewer.update()
        else:
            env.render()

        elapsed = time.time() - start
        diff = 1.0 / PREVIEW_FPS - elapsed
        if diff > 0:
            time.sleep(diff)

    print(colored("Playback finished.", "green"))


def _record_preview_video(env, video_path: str):
    env.reset()
    cam_config = _compute_preview_cam_config(env)
    states = _generate_preview_states(env)
    writer = imageio.get_writer(video_path, fps=PREVIEW_FPS)
    _play_state_sequence(env, states, video_writer=writer, cam_config=cam_config)
    writer.close()
    print(colored(f"Saved preview video to {video_path}", "green"))


def play_move_pan_live(
    source_fixture: str = "counter",
    target_fixture: str = "sink",
    obj_registries: str | tuple[str, ...] = "coppelia_edu",
    teleop: bool = False,
    render_offscreen: bool = False,
    video_path: str | bool = False,
    layout: int | None = None,
    style: int | None = None,
    seed: int = 0,
    device: str = "keyboard",
):
    """Run one MovePan live session: preview (default) or teleop."""
    if isinstance(obj_registries, str):
        registries = parse_registries(obj_registries)
    else:
        registries = obj_registries

    layout, style, seed = _preview_env_defaults(teleop, layout, style, seed)

    env, pygame_viewer, onscreen_renderer = make_move_pan_env(
        source_fixture=source_fixture,
        target_fixture=target_fixture,
        obj_registries=registries,
        layout=layout,
        style=style,
        seed=seed,
        teleop=teleop,
        render_offscreen=render_offscreen,
    )

    if teleop:
        input_device = make_input_device(env, device)
        collect_human_trajectory(
            env,
            input_device,
            "right",
            "single-arm-opposed",
            mirror_actions=True,
            render=(onscreen_renderer not in ("mjviewer", "offscreen")),
            max_fr=30,
            pygame_viewer=pygame_viewer,
        )
        return

    if render_offscreen and video_path:
        _record_preview_video(env, video_path)
        return

    env.reset()
    cam_config = _compute_preview_cam_config(env)
    states = _generate_preview_states(env)
    reset_to(env, {"states": states[0]})

    if pygame_viewer is not None:
        pygame_viewer.free_cam_config = cam_config
        print(colored("Opening viewer (pygame window)...", "yellow"))
        _play_state_sequence(env, states, pygame_viewer=pygame_viewer)
        print(
            colored(
                "Close the window, press Esc/Enter/q in the viewer, "
                "or press Enter/q in the terminal.",
                "green",
            )
        )
        pygame_viewer.wait_until_closed(env.sim)
        pygame_viewer.close()
    elif onscreen_renderer == "mjviewer":
        _play_state_sequence(env, states, cam_config=cam_config)
        try:
            input("Press Enter to close the viewer...")
        except EOFError:
            pass
        if env.viewer is not None:
            env.viewer.close()
            env.viewer = None
    elif onscreen_renderer == "mujoco":
        _play_state_sequence(env, states, cam_config=cam_config)
        try:
            input("Press Enter to close the viewer...")
        except EOFError:
            pass
        if env.viewer is not None:
            env.viewer.close()
            env.viewer = None
