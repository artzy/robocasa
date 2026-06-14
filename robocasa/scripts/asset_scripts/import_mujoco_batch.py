"""Batch-import common MuJoCo official models into RoboCasa assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from robocasa.scripts.asset_scripts.import_mujoco_mjcf import (
    DEFAULT_OUT_ROOT,
    convert_mujoco_object_model,
)

DEFAULT_UPSTREAM = Path(__file__).resolve().parents[3].parent / "mujoco-models"

IMPORTS = (
    ("model/mug/mug.xml", "mug", "mujoco_mug", 0),
    ("model/cards/cards.xml", "cards", "mujoco_card_2_clubs", 0),
)


def main():
    parser = argparse.ArgumentParser(description="Batch import MuJoCo official MJCF objects.")
    parser.add_argument(
        "--upstream-root",
        type=str,
        default=str(DEFAULT_UPSTREAM),
        help="Path to sparse-cloned mujoco repo.",
    )
    args = parser.parse_args()
    upstream = Path(args.upstream_root)

    for rel_path, category, name, body_index in IMPORTS:
        src = upstream / rel_path
        if not src.exists():
            raise FileNotFoundError(f"Missing upstream model: {src}")
        out = convert_mujoco_object_model(
            src_xml=src,
            out_dir=DEFAULT_OUT_ROOT,
            model_name=name,
            category=category,
            body_index=body_index,
        )
        print(f"Imported {category}/{name} -> {out}")


if __name__ == "__main__":
    main()
