"""
Convert upstream MuJoCo official MJCF models (github.com/google-deepmind/mujoco model/)
into RoboCasa MJCFObject-compatible model.xml files.

Usage:
    python robocasa/scripts/asset_scripts/import_mujoco_mjcf.py \\
        --src D:/path/to/mujoco/model/mug/mug.xml \\
        --name mujoco_mug \\
        --update-bb
"""

from __future__ import annotations

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import robocasa

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "utils" / "model_zoo" / "object_template.xml"
DEFAULT_OUT_ROOT = Path(robocasa.models.assets_root) / "objects" / "mujoco_official"


def _resolve_asset_path(src_dir: Path, src_root: ET.Element, filename: str) -> Path:
    compiler = src_root.find("compiler")
    asset_subdir = "."
    if compiler is not None and compiler.get("assetdir"):
        asset_subdir = compiler.get("assetdir")
    for candidate in (src_dir / filename, src_dir / asset_subdir / filename):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing asset file: {filename} under {src_dir}")


def _copy_asset_file(
    src_dir: Path, src_root: ET.Element, filename: str, dest_dir: Path
) -> str:
    src = _resolve_asset_path(src_dir, src_root, filename)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / src.name)
    return f"visual/{src.name}"


def _find_object_body(root: ET.Element, body_index: int = 0) -> ET.Element | None:
    bodies = [body for body in root.findall(".//body") if body.find("freejoint") is not None]
    if bodies:
        if body_index < 0 or body_index >= len(bodies):
            raise IndexError(
                f"body_index {body_index} out of range for {len(bodies)} freejoint bodies"
            )
        return bodies[body_index]
    for body in root.findall(".//worldbody/body/body"):
        return body
    return None


def _materials_for_body(body: ET.Element) -> set[str]:
    names = set()
    for geom in body.findall(".//geom"):
        material = geom.get("material")
        if material:
            names.add(material)
    return names


def _filter_assets(out_asset: ET.Element, keep_materials: set[str] | None) -> None:
    if not keep_materials:
        return
    keep_textures = set()
    for material in out_asset.findall("material"):
        if material.get("name") in keep_materials:
            tex_ref = material.get("texture")
            if tex_ref:
                keep_textures.add(tex_ref)
    for texture in list(out_asset.findall("texture")):
        tex_name = texture.get("name")
        if tex_name and tex_name not in keep_textures:
            out_asset.remove(texture)
    for material in list(out_asset.findall("material")):
        mat_name = material.get("name")
        if mat_name and mat_name not in keep_materials:
            out_asset.remove(material)


def _normalize_object_body(body: ET.Element) -> ET.Element:
    body = deepcopy(body)
    for attr in ("pos", "euler", "quat", "axisangle"):
        body.attrib.pop(attr, None)
    for joint in list(body.findall("freejoint")):
        body.remove(joint)
    return body


def _normalize_collision_defaults(root: ET.Element) -> None:
    for geom in root.findall(".//default//geom"):
        if geom.get("group") in ("3", "2"):
            geom.set("group", "0")


def _strip_freejoint(body: ET.Element) -> None:
    for joint in list(body.findall("freejoint")):
        body.remove(joint)


def _rewrite_mesh_assets(
    root: ET.Element,
    asset: ET.Element,
    src_dir: Path,
    src_root: ET.Element,
    visual_dir: Path,
) -> None:
    for mesh in asset.findall("mesh"):
        file_attr = mesh.get("file")
        if not file_attr:
            continue
        rel = _copy_asset_file(src_dir, src_root, Path(file_attr).name, visual_dir)
        mesh.set("file", rel)
        if mesh.get("name") is None:
            stem = Path(file_attr).stem
            mesh.set("name", stem)
            for geom in root.findall(".//geom"):
                if geom.get("type") == "mesh" and geom.get("mesh") in (None, ""):
                    geom.set("mesh", stem)


def _rewrite_texture_assets(
    asset: ET.Element, src_dir: Path, src_root: ET.Element, visual_dir: Path
) -> None:
    for texture in asset.findall("texture"):
        file_attr = texture.get("file")
        if not file_attr or texture.get("builtin"):
            continue
        rel = _copy_asset_file(src_dir, src_root, Path(file_attr).name, visual_dir)
        texture.set("file", rel)
        if texture.get("name") is None:
            texture.set("name", Path(file_attr).stem)


def _collect_default_classes(defaults_elem: ET.Element) -> dict[str, dict[str, str]]:
    """Map default class name -> merged geom attributes (parent chain included)."""
    result: dict[str, dict[str, str]] = {}

    def walk(elem: ET.Element, inherited: dict[str, str]) -> None:
        for child in elem:
            if child.tag != "default":
                continue
            cls_name = child.get("class")
            if not cls_name:
                continue
            attrs = dict(inherited)
            geom = child.find("geom")
            if geom is not None:
                attrs.update(geom.attrib)
            result[cls_name] = attrs
            walk(child, attrs)

    walk(defaults_elem, {})
    return result


def _flatten_geom_classes(root: ET.Element) -> None:
    defaults_elem = root.find("default")
    if defaults_elem is None:
        return
    class_map = _collect_default_classes(defaults_elem)
    for geom in root.findall(".//geom"):
        cls_name = geom.get("class")
        if not cls_name:
            continue
        if cls_name not in class_map:
            continue
        base = class_map[cls_name]
        merged = dict(base)
        merged.update(geom.attrib)
        merged.pop("class", None)
        geom.attrib.clear()
        geom.attrib.update(merged)
    root.remove(defaults_elem)


def _insert_reg_bbox_from_geoms(root: ET.Element) -> None:
    import numpy as np
    from robosuite.utils.mjcf_utils import array_to_string, string_to_array

    object_body = root.find("worldbody/body/body[@name='object']")
    points = []
    for geom in object_body.findall(".//geom"):
        if geom.get("name") == "reg_bbox":
            continue
        gtype = geom.get("type", "box")
        if gtype not in ("box", "cylinder", "sphere"):
            continue
        pos = string_to_array(geom.get("pos", "0 0 0"))
        size = string_to_array(geom.get("size", "0 0 0"))
        if gtype == "sphere":
            half = np.array([size[0]] * 3)
        elif gtype == "cylinder":
            half = np.array([size[0], size[0], size[1]])
        else:
            half = size
        points.append(pos - half)
        points.append(pos + half)
    if not points:
        return
    points = np.array(points)
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    half_size = (points.max(axis=0) - points.min(axis=0)) / 2.0
    ET.SubElement(
        object_body,
        "geom",
        {
            "group": "1",
            "conaffinity": "0",
            "contype": "0",
            "rgba": "0 1 0 0",
            "name": "reg_bbox",
            "type": "box",
            "pos": array_to_string(center),
            "size": array_to_string(half_size),
        },
    )


def _insert_reg_bbox_from_visual_mesh(
    root: ET.Element, visual_obj: Path, mesh_scale: float = 1.0
) -> None:
    import numpy as np
    import trimesh
    from robosuite.utils.mjcf_utils import array_to_string

    mesh = trimesh.load(visual_obj, force="mesh")
    if mesh_scale != 1.0:
        mesh.apply_scale(mesh_scale)
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    half_size = (bounds[1] - bounds[0]) / 2.0

    object_body = root.find("worldbody/body/body[@name='object']")
    ET.SubElement(
        object_body,
        "geom",
        {
            "class": "region",
            "name": "reg_bbox",
            "type": "box",
            "pos": array_to_string(center),
            "size": array_to_string(half_size),
        },
    )


def _ensure_geom_groups(root: ET.Element) -> None:
    object_body = root.find("worldbody/body/body[@name='object']")
    if object_body is None:
        return
    for geom in object_body.findall(".//geom"):
        if geom.get("name") == "reg_bbox":
            geom.set("group", "1")
            continue
        if geom.get("group") is not None:
            continue
        if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
            geom.set("group", "1")
        else:
            geom.set("group", "0")


def _sanitize_mass_attributes(root: ET.Element) -> None:
    """Remove zero/tiny explicit masses that break MuJoCo when merged into Kitchen."""
    for geom in root.findall(".//geom"):
        mass = geom.get("mass")
        if mass is None:
            continue
        try:
            mass_val = float(mass)
        except ValueError:
            continue
        if mass_val <= 0 or mass_val < 1e-4:
            geom.attrib.pop("mass", None)


def convert_mujoco_object_model(
    src_xml: Path,
    out_dir: Path,
    model_name: str = "mujoco_object",
    category: str | None = None,
    body_index: int = 0,
) -> Path:
    src_xml = Path(src_xml)
    src_dir = src_xml.parent
    category = category or src_dir.name
    instance_dir = out_dir / category / model_name
    visual_dir = instance_dir / "visual"
    instance_dir.mkdir(parents=True, exist_ok=True)

    src_tree = ET.parse(src_xml)
    src_root = src_tree.getroot()
    obj_body = _find_object_body(src_root, body_index=body_index)
    if obj_body is None:
        raise ValueError(f"No freejoint body found in {src_xml}")
    keep_materials = _materials_for_body(obj_body)

    template_tree = ET.parse(TEMPLATE_PATH)
    out_root = template_tree.getroot()
    out_root.set("model", model_name)

    src_asset = src_root.find("asset")
    out_asset = out_root.find("asset")
    for child in list(out_asset):
        out_asset.remove(child)
    if src_asset is not None:
        for child in list(src_asset):
            if child.tag in ("texture", "material") and child.get("builtin"):
                continue
            out_asset.append(deepcopy(child))

    _filter_assets(out_asset, keep_materials if keep_materials else None)
    _rewrite_mesh_assets(out_root, out_asset, src_dir, src_root, visual_dir)
    _rewrite_texture_assets(out_asset, src_dir, src_root, visual_dir)

    # Remove unused template-only assets
    for mat in list(out_asset.findall("material")):
        if mat.get("name") == "floor":
            out_asset.remove(mat)

    mesh_scale = 1.0
    mesh_elem = out_asset.find("mesh")
    if mesh_elem is not None and mesh_elem.get("scale"):
        mesh_scale = float(mesh_elem.get("scale").split()[0])

    # Defaults: upstream hierarchy + RoboCasa region class
    for default in list(out_root.findall("default")):
        out_root.remove(default)
    defaults = ET.Element("default")
    src_defaults = src_root.find("default")
    if src_defaults is not None:
        for child in list(src_defaults):
            defaults.append(deepcopy(child))
    _normalize_collision_defaults(defaults)
    region_default = ET.SubElement(defaults, "default", {"class": "region"})
    ET.SubElement(
        region_default,
        "geom",
        {
            "group": "1",
            "conaffinity": "0",
            "contype": "0",
            "rgba": "0 1 0 0",
        },
    )
    worldbody = out_root.find("worldbody")
    out_root.insert(list(out_root).index(worldbody), defaults)

    object_body = out_root.find("worldbody/body/body[@name='object']")
    if object_body is None:
        raise RuntimeError("object_template.xml missing body name='object'")

    for child in list(_normalize_object_body(obj_body)):
        object_body.append(child)

    model_path = instance_dir / "model.xml"
    template_tree.write(model_path, encoding="utf-8", xml_declaration=True)

    tree = ET.parse(model_path)
    root = tree.getroot()
    visual_obj = visual_dir / "mug.obj"
    if not visual_obj.exists():
        objs = list(visual_dir.glob("*.obj"))
        visual_obj = objs[0] if objs else None
    if visual_obj is not None:
        _insert_reg_bbox_from_visual_mesh(root, visual_obj, mesh_scale=mesh_scale)
    else:
        _insert_reg_bbox_from_geoms(root)
    _flatten_geom_classes(root)
    _ensure_geom_groups(root)
    _sanitize_mass_attributes(root)
    tree.write(model_path, encoding="utf-8", xml_declaration=True)

    return model_path


def main():
    parser = argparse.ArgumentParser(
        description="Import MuJoCo official MJCF into RoboCasa object assets."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to upstream MuJoCo MJCF (e.g. model/mug/mug.xml)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Instance folder name (default: mujoco_<category>)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Category subfolder under mujoco_official/ (default: parent dir name)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT_ROOT),
        help="Output root (default: assets/objects/mujoco_official)",
    )
    parser.add_argument(
        "--body-index",
        type=int,
        default=0,
        help="When the MJCF has multiple freejoint bodies (e.g. cards), pick one.",
    )
    parser.add_argument(
        "--update-bb",
        action="store_true",
        help="Run calc_object_bb_reg after conversion",
    )
    args = parser.parse_args()

    src_xml = Path(args.src)
    category = args.category or src_xml.parent.name
    model_name = args.name or f"mujoco_{category}"

    model_path = convert_mujoco_object_model(
        src_xml=src_xml,
        out_dir=Path(args.out),
        model_name=model_name,
        category=category,
        body_index=args.body_index,
    )
    print(f"Wrote {model_path}")

    if args.update_bb:
        from robocasa.scripts.asset_scripts.calc_object_bb_reg import update_bb_geom

        update_bb_geom(str(model_path.parent))
        print(f"Updated reg_bbox in {model_path}")

    import mujoco

    mujoco.MjModel.from_xml_path(str(model_path))
    print("MuJoCo compile OK")


if __name__ == "__main__":
    main()
