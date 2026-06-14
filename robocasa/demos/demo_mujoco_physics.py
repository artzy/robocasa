"""
Play upstream MuJoCo official MJCF scenes outside the RoboCasa Kitchen env.

Use this for articulated / multi-body demos (e.g. Rubik's cube) that should not
be merged into Kitchen tasks.

Usage:
    python -m robocasa.demos.demo_mujoco_physics --model cube
    python -m robocasa.demos.demo_mujoco_physics --model dobot_magician
    python -m robocasa.demos.demo_mujoco_physics --model mug --mjcf-path path/to/custom.xml
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import robocasa
from termcolor import colored

DEFAULT_MUJOCO_MODELS_ROOT = Path(__file__).resolve().parents[3].parent / "mujoco-models"
DEFAULT_COPPELIA_ROBOTS_ROOT = (
    Path(robocasa.models.assets_root) / "robots" / "coppelia_edu"
)

MODEL_PRESETS = {
    "mug": ("mujoco", "model/mug/mug.xml"),
    "cube": ("mujoco", "model/cube/cube_3x3x3.xml"),
    "cards": ("mujoco", "model/cards/cards.xml"),
    "dobot_magician": ("coppelia", "dobot_magician/model.urdf"),
}


def resolve_model_path(
    model: str,
    models_root: Path,
    coppelia_root: Path,
    mjcf_path: str | None,
) -> Path:
    if mjcf_path is not None:
        path = Path(mjcf_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return path
    if model not in MODEL_PRESETS:
        raise ValueError(f"Unknown model '{model}'. Choices: {list(MODEL_PRESETS)}")
    source, rel = MODEL_PRESETS[model]
    if source == "mujoco":
        path = models_root / rel
    else:
        path = coppelia_root / rel
    if not path.exists():
        if source == "coppelia":
            raise FileNotFoundError(
                f"Missing {path}. Run:\n"
                "  python robocasa/scripts/asset_scripts/import_coppelia_robot_urdf.py"
            )
        raise FileNotFoundError(
            f"Missing {path}. Clone upstream models:\n"
            "  git clone --depth 1 --filter=blob:none --sparse "
            "https://github.com/google-deepmind/mujoco.git ../mujoco-models\n"
            "  cd ../mujoco-models && git sparse-checkout set model/mug model/cube model/cards"
        )
    return path


def run_viewer(model_path: Path) -> None:
    print(colored(f"Loading {model_path}", "yellow"))
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    print(
        colored(
            f"Model loaded: nbody={model.nbody}, ngeom={model.ngeom}, njnt={model.njnt}",
            "green",
        )
    )
    print(colored("Close the MuJoCo viewer window to exit.", "green"))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.time() - step_start
            if model.opt.timestep - elapsed > 0:
                time.sleep(model.opt.timestep - elapsed)


def main():
    parser = argparse.ArgumentParser(
        description="View upstream MuJoCo official MJCF physics demos."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mug",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Preset model under mujoco-models/",
    )
    parser.add_argument(
        "--mjcf-path",
        type=str,
        default=None,
        help="Optional explicit MJCF path (overrides --model).",
    )
    parser.add_argument(
        "--mujoco-models-root",
        type=str,
        default=os.environ.get("MUJOCO_MODELS_ROOT", str(DEFAULT_MUJOCO_MODELS_ROOT)),
        help="Path to sparse-cloned google-deepmind/mujoco repo.",
    )
    parser.add_argument(
        "--coppelia-robots-root",
        type=str,
        default=str(DEFAULT_COPPELIA_ROBOTS_ROOT),
        help="Path to converted CoppeliaSim URDF robots.",
    )
    args = parser.parse_args()

    model_path = resolve_model_path(
        args.model,
        Path(args.mujoco_models_root),
        Path(args.coppelia_robots_root),
        args.mjcf_path,
    )
    run_viewer(model_path)


if __name__ == "__main__":
    main()
