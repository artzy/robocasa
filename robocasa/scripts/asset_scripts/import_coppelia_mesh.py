"""Import CoppeliaSim-exported OBJ meshes into RoboCasa MJCFObject assets.

Uses coacd for convex collision decomposition on Windows (VHACD/TestVHACD optional).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import coacd
import numpy as np
import trimesh
from termcolor import colored

import robocasa.utils.model_zoo.mjcf_gen_utils as MJCFGenUtils
import robocasa.utils.model_zoo.parser_utils as ParserUtils


def _load_mesh_from_obj(src_obj: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(src_obj, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _mesh_bbox_center(mesh: trimesh.Trimesh) -> np.ndarray:
    return (mesh.bounds[0] + mesh.bounds[1]) / 2.0


def _load_grasp_sidecar(sidecar_path: Path) -> dict | None:
    if not sidecar_path.exists():
        return None
    with sidecar_path.open(encoding="utf-8") as f:
        return json.load(f)


def _grasp_pos_in_mjcf(
    sidecar: dict,
    center: np.ndarray,
) -> np.ndarray:
    """Map exported world-frame graspPoint to centered MJCF body coords."""
    return np.asarray(sidecar["pos"], dtype=float) - center


def _inject_grasp_site(model_xml: Path, grasp_pos_mjcf: np.ndarray) -> None:
    tree = ET.parse(model_xml)
    root = tree.getroot()
    object_body = root.find("./worldbody/body/body[@name='object']")
    if object_body is None:
        raise ValueError(f"object body not found in {model_xml}")

    for site in list(object_body.findall("site")):
        if site.get("name") == "grasp_site":
            object_body.remove(site)

    pos_str = " ".join(f"{v:.9f}" for v in grasp_pos_mjcf)
    ET.SubElement(
        object_body,
        "site",
        {
            "name": "grasp_site",
            "pos": pos_str,
            "rgba": "0 0 0 0",
            "size": "0.004",
        },
    )
    tree.write(model_xml, encoding="utf-8")


def _maybe_inject_grasp_site(
    model_xml: Path,
    src_obj: Path,
    *,
    center: bool,
    grasp_sidecar: Path | None = None,
) -> bool:
    sidecar_path = grasp_sidecar or src_obj.parent / "grasp_point.json"
    sidecar = _load_grasp_sidecar(sidecar_path)
    if sidecar is None:
        return False

    mesh = _load_mesh_from_obj(src_obj)
    center_vec = _mesh_bbox_center(mesh) if center else np.zeros(3, dtype=float)
    grasp_pos = _grasp_pos_in_mjcf(sidecar, center_vec)
    _inject_grasp_site(model_xml, grasp_pos)
    return True


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
    grasp_sidecar: Path | None = None,
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
    model_xml = output_path / "model.xml"
    if _maybe_inject_grasp_site(
        model_xml,
        src_obj,
        center=center,
        grasp_sidecar=grasp_sidecar,
    ):
        print(colored(f"Injected grasp_site into {model_xml}", color="cyan"))
    return model_xml


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
    parser.add_argument(
        "--grasp_sidecar",
        type=str,
        default=None,
        help="Optional grasp_point.json from CoppeliaSim export.",
    )
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
        grasp_sidecar=Path(args.grasp_sidecar) if args.grasp_sidecar else None,
    )
    print(colored(f"Model output to: {out}", color="green"))


if __name__ == "__main__":
    main()
