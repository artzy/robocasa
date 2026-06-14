"""Robot Home pose JSON presets for live preview demos."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from termcolor import colored

_DEMO_DIR = Path(__file__).resolve().parents[1]


def _demos_relative(path: str) -> Path:
    """Resolve a path under ``robocasa/demos/`` from the installed demos package."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    prefix = "robocasa/demos/"
    rel = path[len(prefix) :] if path.startswith(prefix) else path
    return _DEMO_DIR / rel


def _repo_relative(path: str) -> Path:
    return _demos_relative(path)


@dataclass
class RobotHomePose:
    base_pos: np.ndarray
    base_ori: np.ndarray
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    gripper: float = 0.0
    task_name: str = ""
    name: str = ""
    layout: int | None = None
    style: int | None = None
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def source_fixture(self) -> str | None:
        return self.extra.get("source_fixture")

    @property
    def target_fixture(self) -> str | None:
        return self.extra.get("target_fixture")

    def to_dict(self) -> dict:
        data = {
            "task_name": self.task_name,
            "name": self.name,
            "layout": self.layout,
            "style": self.style,
            "seed": self.seed,
            "base_pos": self.base_pos.tolist(),
            "base_yaw": float(self.base_ori[2]),
            "eef_pos": self.eef_pos.tolist(),
            "eef_quat_wxyz": self.eef_quat.tolist(),
            "gripper": float(self.gripper),
        }
        data.update(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> RobotHomePose:
        base_yaw = float(data.get("base_yaw", 0.0))
        known = {
            "task_name",
            "name",
            "layout",
            "style",
            "seed",
            "base_pos",
            "base_yaw",
            "eef_pos",
            "eef_quat_wxyz",
            "gripper",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            task_name=str(data.get("task_name", "")),
            name=str(data.get("name", "")),
            layout=data.get("layout"),
            style=data.get("style"),
            seed=data.get("seed"),
            base_pos=np.asarray(data["base_pos"], dtype=float),
            base_ori=np.array([0.0, 0.0, base_yaw], dtype=float),
            eef_pos=np.asarray(data["eef_pos"], dtype=float),
            eef_quat=np.asarray(data["eef_quat_wxyz"], dtype=float),
            gripper=float(data.get("gripper", 0.0)),
            extra=extra,
        )


HOME_BASE_TOLERANCE = 0.02
HOME_EEF_TOLERANCE = 0.05

DEFAULT_HOME_PRESETS: dict[str, str] = {
    "MovePan": "home_presets/MovePan/default_layout15_seed0.json",
    "DeliverStraw": "home_presets/DeliverStraw/default_from_demo_ep0.json",
    "HotDogSetup": "home_presets/HotDogSetup/default_from_demo_ep0.json",
}

LEGACY_HOME_PRESETS: dict[str, tuple[str, ...]] = {
    "MovePan": (
        "move_pan_home_presets/default_layout15_seed0.json",
    ),
}


def default_home_preset_path(task_name: str) -> Path | None:
    rel = DEFAULT_HOME_PRESETS.get(task_name)
    if rel is None:
        return None
    return _demos_relative(rel)


def resolve_home_preset_path(
    task_name: str,
    preset_path: Path | str | None,
) -> Path | None:
    if preset_path is not None:
        path = Path(preset_path)
        if not path.is_absolute():
            path = _demos_relative(str(preset_path))
        return path

    default = default_home_preset_path(task_name)
    if default is not None and default.exists():
        return default

    for rel in LEGACY_HOME_PRESETS.get(task_name, ()):
        legacy = _demos_relative(rel)
        if legacy.exists():
            return legacy
    return default


def load_home_preset(path: Path | str) -> RobotHomePose:
    preset_path = Path(path)
    with preset_path.open(encoding="utf-8") as f:
        return RobotHomePose.from_dict(json.load(f))


def save_home_preset(home: RobotHomePose, path: Path | str) -> Path:
    preset_path = Path(path)
    if not preset_path.is_absolute():
        preset_path = _demos_relative(str(path))
    preset_path.parent.mkdir(parents=True, exist_ok=True)
    with preset_path.open("w", encoding="utf-8") as f:
        json.dump(home.to_dict(), f, indent=2)
        f.write("\n")
    return preset_path


def warn_home_mismatch(
    home: RobotHomePose,
    *,
    task_name: str | None = None,
    layout: int | None = None,
    style: int | None = None,
    seed: int | None = None,
    extra_checks: dict[str, Any] | None = None,
) -> None:
    checks: list[tuple[str, Any, Any]] = []
    if task_name is not None:
        checks.append(("task_name", home.task_name, task_name))
    checks.extend(
        (
            ("layout", home.layout, layout),
            ("style", home.style, style),
            ("seed", home.seed, seed),
        )
    )
    if extra_checks:
        for key, runtime in extra_checks.items():
            preset_val = home.extra.get(key, getattr(home, key, None))
            if preset_val is None and key in ("source_fixture", "target_fixture"):
                preset_val = home.extra.get(key)
            checks.append((key, preset_val, runtime))

    mismatches = [
        f"{name}: preset={preset!r} runtime={runtime!r}"
        for name, preset, runtime in checks
        if preset is not None and runtime is not None and preset != runtime
    ]
    if mismatches:
        print(
            colored(
                "warning: home preset metadata mismatch - "
                + "; ".join(mismatches),
                "yellow",
            )
        )
