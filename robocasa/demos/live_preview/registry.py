"""Registry of demo_tasks live preview and home-wrapped human demo tasks."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

from robocasa.demos.live_preview.home_pose import DEFAULT_HOME_PRESETS
from robocasa.demos.live_preview.robot_bridge import PreReturnFn


@dataclass(frozen=True)
class LiveDemoSpec:
    task_name: str
    module: str
    play_fn_name: str
    default_home_preset: str | None = None

    def play(self, **kwargs: Any):
        mod = importlib.import_module(self.module)
        fn: Callable = getattr(mod, self.play_fn_name)
        return fn(**kwargs)


@dataclass(frozen=True)
class HomeDemoSpec:
    task_name: str
    default_home_preset: str | None = None
    pre_return_module: str | None = None
    pre_return_fn: str | None = None
    demo_stride: int = 1
    demo_tail_extend: int = 50

    def resolve_pre_return(self) -> PreReturnFn | None:
        if self.pre_return_module is None or self.pre_return_fn is None:
            return None
        mod = importlib.import_module(self.pre_return_module)
        return getattr(mod, self.pre_return_fn)


LIVE_DEMO_REGISTRY: dict[str, LiveDemoSpec] = {
    "MovePan": LiveDemoSpec(
        task_name="MovePan",
        module="robocasa.demos.move_pan_live",
        play_fn_name="play_move_pan_live",
        default_home_preset=DEFAULT_HOME_PRESETS["MovePan"],
    ),
}

HOME_DEMO_REGISTRY: dict[str, HomeDemoSpec] = {
    "DeliverStraw": HomeDemoSpec(
        task_name="DeliverStraw",
        default_home_preset=DEFAULT_HOME_PRESETS["DeliverStraw"],
    ),
    "HotDogSetup": HomeDemoSpec(
        task_name="HotDogSetup",
        default_home_preset=DEFAULT_HOME_PRESETS["HotDogSetup"],
        pre_return_module="robocasa.demos.live_preview.pre_return.hot_dog_setup",
        pre_return_fn="append_hot_dog_pre_return_states",
        demo_stride=2,
        demo_tail_extend=20,
    ),
}

LIVE_DEMO_TASKS = frozenset(LIVE_DEMO_REGISTRY.keys())
HOME_DEMO_TASKS = frozenset(HOME_DEMO_REGISTRY.keys())


def default_home_preset_for_task(task_name: str) -> str | None:
    spec = LIVE_DEMO_REGISTRY.get(task_name) or HOME_DEMO_REGISTRY.get(task_name)
    if spec is None:
        return None
    return spec.default_home_preset


def home_demo_spec_for_task(task_name: str) -> HomeDemoSpec | None:
    return HOME_DEMO_REGISTRY.get(task_name)


def pre_return_fn_for_task(task_name: str) -> PreReturnFn | None:
    spec = HOME_DEMO_REGISTRY.get(task_name)
    if spec is None:
        return None
    return spec.resolve_pre_return()
