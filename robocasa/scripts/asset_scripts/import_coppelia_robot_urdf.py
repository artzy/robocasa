"""Import CoppeliaSim URDF robots: DAE->OBJ conversion and MuJoCo compile verification."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco

from robocasa.scripts.asset_scripts.convert_coppelia_urdf_meshes import (
    DEFAULT_EXPORT_ROOT,
    DEFAULT_OUT_ROOT,
    ROBOT_IMPORTS,
    prepare_robot_for_mujoco,
)


def main():
    parser = argparse.ArgumentParser(
        description="Convert CoppeliaSim URDF exports for MuJoCo (DAE to OBJ)."
    )
    parser.add_argument(
        "--export-root",
        type=str,
        default=str(DEFAULT_EXPORT_ROOT),
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(DEFAULT_OUT_ROOT),
    )
    args = parser.parse_args()

    export_root = Path(args.export_root)
    out_root = Path(args.out_root)

    verified = 0
    for rel_path, stem, out_name in ROBOT_IMPORTS:
        export_dir = export_root / rel_path / stem
        urdf_src = export_dir / "model.urdf"
        if not urdf_src.exists():
            print(f"SKIP (missing URDF): {urdf_src}")
            continue

        out_dir = out_root / out_name
        urdf_out = prepare_robot_for_mujoco(export_dir, out_dir)
        print(f"Prepared {out_name} -> {urdf_out}")

        try:
            model = mujoco.MjModel.from_xml_path(str(urdf_out))
            print(
                f"OK {out_name}: nbody={model.nbody}, njnt={model.njnt}, ngeom={model.ngeom}"
            )
            verified += 1
        except ValueError as exc:
            print(f"FAIL {out_name}: {exc}")

    if verified == 0:
        raise RuntimeError("No CoppeliaSim URDF robots compiled in MuJoCo.")


if __name__ == "__main__":
    main()
