"""Import CoppeliaSim-exported OBJ meshes into RoboCasa MJCFObject assets.

Uses coacd for convex collision decomposition on Windows (VHACD/TestVHACD optional).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import coacd
import numpy as np
import trimesh
from termcolor import colored

import robocasa.utils.model_zoo.mjcf_gen_utils as MJCFGenUtils
import robocasa.utils.model_zoo.parser_utils as ParserUtils


def _decompose_coacd(mesh: trimesh.Trimesh, coll_dir: Path, max_hulls: int = 16) -> None:
    coll_dir.mkdir(parents=True, exist_ok=True)
    mesh = mesh.copy()
    if not mesh.is_watertight:
        mesh = mesh.convex_hull
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    parts = coacd.run_coacd(
        coacd.Mesh(vertices, faces),
        threshold=0.05,
        max_convex_hull=max_hulls,
    )
    for i, part in enumerate(parts):
        if isinstance(part, (list, tuple)) and len(part) >= 2:
            verts, faces = part[0], part[1]
        else:
            verts, faces = part.vertices, part.indices
        part_mesh = trimesh.Trimesh(verts, faces)
        part_mesh.export(coll_dir / f"model_normalized_collision_{i}.obj")


def import_coppelia_obj(
    src_obj: Path,
    output_path: Path,
    model_name: str,
    rot: str = "none",
    scale: float = 1.0,
    prescale: bool = False,
    center: bool = True,
    verbose: bool = False,
) -> Path:
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    raw_dir = output_path / "raw"
    raw_dir.mkdir()
    mesh = trimesh.load(src_obj, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    raw_obj = raw_dir / "model_normalized.obj"
    mesh.export(raw_obj)

    coll_path = output_path / "collision"
    _decompose_coacd(mesh, coll_path)

    model_info = MJCFGenUtils.parse_model_info(
        model_path=str(raw_dir),
        model_name=model_name,
        coll_model_path=str(coll_path),
        asset_path=str(output_path),
        rot=[] if rot == "none" else rot,
        verbose=verbose,
        prescale=prescale,
        center=center,
    )

    MJCFGenUtils.generate_mjcf(
        asset_path=str(output_path),
        model_name=model_name,
        model_info=model_info,
        sc=scale,
        verbose=verbose,
    )
    return output_path / "model.xml"


def main():
    parser = ParserUtils.get_base_parser()
    parser.add_argument("--path", type=str, required=True, help="Path to exported .obj")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--rot",
        type=str,
        default="none",
        choices=[
            "none",
            "x",
            "y",
            "z",
            "x90",
            "x180",
            "x270",
            "y90",
            "y180",
            "y270",
            "z60",
            "z90",
            "z180",
            "z270",
        ],
        help="Axis rotation; CoppeliaSim exports are Z-up — use none.",
    )
    parser.add_argument("--center", type=str, nargs="?", const="False", default="True")
    parser.add_argument("--prescale", type=str, nargs="?", const="False", default="False")
    args = parser.parse_args()

    model_name = args.model_name or Path(args.path).stem
    out = import_coppelia_obj(
        src_obj=Path(args.path),
        output_path=Path(args.output_path),
        model_name=model_name,
        rot=args.rot,
        scale=args.scale,
        prescale=args.prescale not in ("False", "false", False),
        center=args.center not in ("False", "false", False),
        verbose=args.verbose,
    )
    print(colored(f"Model output to: {out}", color="green"))


if __name__ == "__main__":
    main()
