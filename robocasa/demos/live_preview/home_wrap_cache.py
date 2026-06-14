"""Disk cache for home-wrapped demo state sequences."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

_DEMO_DIR = Path(__file__).resolve().parents[1]
CACHE_ROOT = _DEMO_DIR / ".cache" / "home_wrap"


def _code_fingerprint() -> str:
    """Hash mtimes of wrap-related modules so cache invalidates on code changes."""
    modules = (
        "live_preview/robot_bridge.py",
        "live_preview/base_path_planning.py",
        "live_preview/fixture_transition.py",
        "live_preview/pre_return/hot_dog_setup.py",
        "live_preview/home_pose.py",
    )
    parts: list[str] = []
    for rel in modules:
        path = _DEMO_DIR / rel
        if path.exists():
            parts.append(f"{rel}:{int(path.stat().st_mtime)}")
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]
    return digest


def cache_path(
    task_name: str,
    episode_index: int,
    *,
    home_preset: Path | str | None,
    demo_stride: int,
    extend_demo_tail: int,
) -> Path:
    preset_part = "none"
    if home_preset is not None:
        preset_path = Path(home_preset)
        if preset_path.exists():
            preset_part = f"{preset_path.name}_{int(preset_path.stat().st_mtime)}"
        else:
            preset_part = preset_path.name
    key = (
        f"{task_name}_ep{episode_index}_"
        f"{preset_part}_s{demo_stride}_t{extend_demo_tail}_"
        f"{_code_fingerprint()}"
    )
    safe = key.replace(" ", "_").replace("\\", "_").replace("/", "_")
    return CACHE_ROOT / f"{safe}.npz"


def load_cached_states(path: Path) -> list[np.ndarray] | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    states = data["states"]
    return [np.asarray(s) for s in states]


def save_cached_states(path: Path, states: list[np.ndarray], meta: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=np.array(states, dtype=object),
        meta=json.dumps(meta),
    )
    return path
