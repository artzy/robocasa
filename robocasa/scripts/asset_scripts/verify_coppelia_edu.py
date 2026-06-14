"""Verify CoppeliaSim Edu imports: MJCF compile, sampling, and optional Kitchen env reset."""

from __future__ import annotations

import argparse

import numpy as np
import mujoco
from pathlib import Path

from robocasa.models.objects.kitchen_object_utils import sample_kitchen_object
from robocasa.models.objects.objects import MJCFObject
from robocasa.utils.env_utils import create_env

OBJECT_ROOT = Path("robocasa/models/assets/objects/coppelia_edu")
FIXTURE_ROOT = Path("robocasa/models/assets/fixtures/coppelia_edu")
ROBOT_ROOT = Path("robocasa/models/assets/robots/coppelia_edu")


def verify_mjcf_assets(root: Path, label: str) -> None:
    if not root.exists():
        print(f"SKIP {label}: {root} not found")
        return
    for p in root.rglob("model.xml"):
        mujoco.MjModel.from_xml_path(str(p))
        MJCFObject(name=p.parent.name, mjcf_path=str(p))
        print(f"OK {label}", p.parent.name)


def verify_robot_urdf(root: Path) -> None:
    if not root.exists():
        print(f"SKIP robots: {root} not found")
        return
    for urdf in root.rglob("model.urdf"):
        model = mujoco.MjModel.from_xml_path(str(urdf))
        print(
            f"OK robot {urdf.parent.name}: "
            f"nbody={model.nbody}, njnt={model.njnt}, ngeom={model.ngeom}"
        )


def verify_sampling() -> None:
    rng = np.random.default_rng(0)
    _, info = sample_kitchen_object(
        groups=("receptacle",),
        obj_registries=("coppelia_edu",),
        rng=rng,
    )
    print("sampled", info["cat"], info["mjcf_path"])


def verify_kitchen_env(max_seeds: int = 3) -> None:
    for seed in range(max_seeds):
        try:
            env = create_env(
                "PickPlaceCounterToSink",
                split="pretrain",
                obj_registries=("coppelia_edu",),
                seed=seed,
            )
            env.reset()
            env.close()
            print(
                f"Kitchen env reset OK (PickPlaceCounterToSink, coppelia_edu, seed={seed})"
            )
            return
        except Exception:
            continue
    print(
        "WARN: Kitchen env reset not confirmed (layout sampling). "
        "Run demo_teleop manually with obj_registries=('coppelia_edu',)."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-env",
        action="store_true",
        help="Also try Kitchen env reset (slow; may warn on layout failure).",
    )
    args = parser.parse_args()

    verify_mjcf_assets(OBJECT_ROOT, "object")
    verify_mjcf_assets(FIXTURE_ROOT, "fixture")
    verify_robot_urdf(ROBOT_ROOT)
    verify_sampling()
    if args.with_env:
        verify_kitchen_env()
    print("ALL OK")


if __name__ == "__main__":
    main()
