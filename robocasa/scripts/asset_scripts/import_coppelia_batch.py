"""Batch-import CoppeliaSim-exported OBJ meshes into RoboCasa MJCFObject assets."""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import robocasa

DEFAULT_EXPORT_ROOT = Path(__file__).resolve().parents[3] / "exports" / "coppelia_edu"
DEFAULT_OUT_ROOT = Path(robocasa.models.assets_root) / "objects" / "coppelia_edu"

# (export category, export stem, output model_name, kitchen category folder)
IMPORTS = (
    ("household", "cup", "coppelia_cup", "household"),
    ("household", "bowl", "coppelia_bowl", "household"),
    ("household", "largeBasket", "coppelia_basket", "household"),
    ("kitchenware", "frying_pan_01", "coppelia_frying_pan", "kitchenware"),
)

FIXTURE_IMPORTS = (
    ("furniture/tables", "diningTable", "coppelia_dining_table", "tables"),
)


def _ensure_region_default(model_xml: Path) -> None:
    """Add RoboCasa region default class required by calc_object_bb_reg / MJCFObject."""
    tree = ET.parse(model_xml)
    root = tree.getroot()
    if root.find("default") is None:
        defaults = ET.Element("default")
        region_default = ET.SubElement(defaults, "default", {"class": "region"})
        ET.SubElement(
            region_default,
            "geom",
            {"group": "1", "conaffinity": "0", "contype": "0", "rgba": "0 1 0 0"},
        )
        worldbody = root.find("worldbody")
        if worldbody is not None:
            root.insert(list(root).index(worldbody), defaults)
    tree.write(model_xml, encoding="utf-8")


def _relativize_asset_paths(model_xml: Path) -> None:
    """Convert absolute mesh paths to paths relative to model.xml."""
    model_dir = model_xml.parent
    tree = ET.parse(model_xml)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        return
    for mesh in asset.findall("mesh"):
        file_attr = mesh.get("file")
        if not file_attr:
            continue
        p = Path(file_attr)
        if p.is_absolute():
            try:
                mesh.set("file", str(p.relative_to(model_dir)).replace("\\", "/"))
            except ValueError:
                pass
    tree.write(model_xml, encoding="utf-8")


def _postprocess_model_dir(model_dir: Path) -> None:
    model_xml = model_dir / "model.xml"
    if model_xml.exists():
        _relativize_asset_paths(model_xml)
        _ensure_region_default(model_xml)


def import_mesh_obj(
    src_obj: Path,
    out_root: Path,
    category: str,
    model_name: str,
    scale: float = 1.0,
) -> Path:
    """Run CoppeliaSim OBJ import (coacd collision on Windows)."""
    import_mesh = Path(__file__).resolve().parent / "import_coppelia_mesh.py"
    out_dir = out_root / category / model_name
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)

    cmd = [
        sys.executable,
        str(import_mesh),
        "--path",
        str(src_obj),
        "--model_name",
        model_name,
        "--output_path",
        str(out_dir),
        "--prescale",
        "False",
        "--center",
        "True",
        "--rot",
        "none",
        "--scale",
        str(scale),
    ]
    subprocess.run(cmd, check=True)
    return out_dir


def update_bbox(out_root: Path) -> None:
    calc_bb = Path(__file__).resolve().parent / "calc_object_bb_reg.py"
    subprocess.run(
        [sys.executable, str(calc_bb), "--folder", str(out_root)],
        check=True,
    )


def import_fixtures(export_root: Path, fixture_root: Path) -> list[Path]:
    """Import exported furniture OBJs as static fixtures."""
    imported = []
    for export_cat, stem, model_name, asset_cat in FIXTURE_IMPORTS:
        src_obj = export_root / export_cat / stem / "model.obj"
        if not src_obj.exists():
            print(f"SKIP fixture (missing export): {src_obj}")
            continue
        out_dir = import_mesh_obj(src_obj, fixture_root, asset_cat, model_name)
        _postprocess_model_dir(out_dir)
        print(f"Imported fixture {export_cat}/{stem} -> {out_dir}")
        imported.append(out_dir)
    return imported


def main():
    parser = argparse.ArgumentParser(description="Import CoppeliaSim OBJ exports into RoboCasa.")
    parser.add_argument(
        "--export-root",
        type=str,
        default=str(DEFAULT_EXPORT_ROOT),
        help="Path to exports/coppelia_edu (OBJ output from CoppeliaSim).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(DEFAULT_OUT_ROOT),
        help="RoboCasa assets output root (default: assets/objects/coppelia_edu).",
    )
    parser.add_argument(
        "--skip-bbox",
        action="store_true",
        help="Skip calc_object_bb_reg after import.",
    )
    args = parser.parse_args()

    export_root = Path(args.export_root)
    out_root = Path(args.out_root)

    imported = []
    for export_cat, stem, model_name, asset_cat in IMPORTS:
        src_obj = export_root / export_cat / stem / "model.obj"
        if not src_obj.exists():
            print(f"SKIP (missing export): {src_obj}")
            continue
        out_dir = import_mesh_obj(src_obj, out_root, asset_cat, model_name)
        _postprocess_model_dir(out_dir)
        print(f"Imported {export_cat}/{stem} -> {out_dir}")
        imported.append(out_dir)

    if not imported:
        print("WARNING: no kitchen object OBJ exports found; continuing for fixtures/robots.")

    if imported and not args.skip_bbox:
        update_bbox(out_root)
        print(f"Updated bounding boxes under {out_root}")

    fixture_root = Path(robocasa.models.assets_root) / "fixtures" / "coppelia_edu"
    fixture_imported = import_fixtures(export_root, fixture_root)
    if fixture_imported and not args.skip_bbox:
        update_bbox(fixture_root)

    robot_script = Path(__file__).resolve().parent / "import_coppelia_robot_urdf.py"
    result = subprocess.run(
        [sys.executable, str(robot_script), "--export-root", str(export_root)],
        check=False,
    )
    if result.returncode != 0:
        print("WARNING: CoppeliaSim URDF robot import failed (missing export or compile error).")


if __name__ == "__main__":
    main()
