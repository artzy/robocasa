"""Capture MovePan preview home pose (wrapper around capture_robot_home)."""

from __future__ import annotations

import sys

from robocasa.scripts.asset_scripts import capture_robot_home


def main() -> None:
    if "--task" not in sys.argv:
        sys.argv = [sys.argv[0], "--task", "MovePan", *sys.argv[1:]]
    capture_robot_home.main()


if __name__ == "__main__":
    main()
