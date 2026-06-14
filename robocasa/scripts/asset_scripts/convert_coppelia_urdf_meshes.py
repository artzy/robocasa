"""Convert CoppeliaSim URDF mesh references from DAE to OBJ for MuJoCo."""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse

import trimesh

DEFAULT_EXPORT_ROOT = Path(__file__).resolve().parents[3] / "exports" / "coppelia_edu"
DEFAULT_OUT_ROOT = (
    Path(__file__).resolve().parents[2] / "models" / "assets" / "robots" / "coppelia_edu"
)

ROBOT_IMPORTS = (
    ("robots/non-mobile", "Dobot_Magician", "dobot_magician"),
)


def _mesh_path_from_urdf_ref(filename: str) -> Path:
    """Resolve URDF mesh filename (file:// or relative) to a local path."""
    filename = filename.strip()
    if filename.startswith("file://"):
        parsed = urlparse(filename)
        path = unquote(parsed.path)
        if len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return Path(path)
    return Path(filename)


def convert_dae_to_obj(dae_path: Path, obj_path: Path) -> None:
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    if obj_path.exists() and obj_path.stat().st_mtime >= dae_path.stat().st_mtime:
        return
    mesh = trimesh.load(dae_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.export(obj_path)


def convert_robot_meshes(
    export_robot_dir: Path,
    mesh_out_dir: Path,
) -> int:
    """Convert all .dae in export_robot_dir to OBJ in mesh_out_dir. Returns count converted."""
    mesh_out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for dae in sorted(export_robot_dir.glob("*.dae")):
        convert_dae_to_obj(dae, mesh_out_dir / f"{dae.stem}.obj")
        count += 1
    return count


def rewrite_urdf_mesh_paths(
    urdf_in: Path,
    urdf_out: Path,
) -> None:
    """Rewrite .dae file:// references to basename .obj paths (MuJoCo URDF uses filename only)."""
    tree = ET.parse(urdf_in)
    root = tree.getroot()

    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        src = _mesh_path_from_urdf_ref(filename)
        if src.suffix.lower() != ".dae":
            continue
        mesh.set("filename", f"{src.stem}.obj")

    urdf_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(urdf_out, encoding="utf-8", xml_declaration=True)


def prepare_robot_for_mujoco(
    export_robot_dir: Path,
    out_robot_dir: Path,
    urdf_name: str = "model.urdf",
) -> Path:
    """
    Convert DAE meshes, write MuJoCo-ready URDF to out_robot_dir.
    OBJ files are placed next to the URDF (MuJoCo resolves mesh paths by basename).
    Returns path to the output URDF.
    """
    urdf_src = export_robot_dir / urdf_name
    if not urdf_src.exists():
        raise FileNotFoundError(f"Missing URDF: {urdf_src}")

    if out_robot_dir.exists():
        shutil.rmtree(out_robot_dir)
    out_robot_dir.mkdir(parents=True)

    n_meshes = convert_robot_meshes(export_robot_dir, out_robot_dir)
    if n_meshes == 0:
        raise FileNotFoundError(f"No .dae meshes found in {export_robot_dir}")

    urdf_out = out_robot_dir / urdf_name
    rewrite_urdf_mesh_paths(urdf_src, urdf_out)
    return urdf_out


def main():
    parser = argparse.ArgumentParser(
        description="Convert CoppeliaSim URDF DAE meshes to OBJ for MuJoCo."
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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run mujoco.MjModel.from_xml_path after conversion.",
    )
    args = parser.parse_args()

    export_root = Path(args.export_root)
    out_root = Path(args.out_root)

    for rel_path, stem, out_name in ROBOT_IMPORTS:
        export_dir = export_root / rel_path / stem
        if not (export_dir / "model.urdf").exists():
            print(f"SKIP (no URDF): {export_dir}")
            continue
        out_dir = out_root / out_name
        urdf_out = prepare_robot_for_mujoco(export_dir, out_dir)
        print(f"Converted {stem} -> {urdf_out}")

        if args.verify:
            import mujoco

            model = mujoco.MjModel.from_xml_path(str(urdf_out))
            print(
                f"  MuJoCo OK: nbody={model.nbody}, njnt={model.njnt}, ngeom={model.ngeom}"
            )


if __name__ == "__main__":
    main()
