"""Verify HotDogSetup home wrap with pre-return fridge close."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import robocasa  # noqa: F401
from robocasa.demos.live_preview.fixture_transition import (
    needs_fridge_close,
    resolve_env_fixture,
)
from robocasa.demos.live_preview.home_demo_playback import (
    _load_episode,
    _make_env_from_dataset,
    build_home_wrapped_states_for_task,
)
from robocasa.demos.live_preview.registry import home_demo_spec_for_task
from robocasa.demos.live_preview.home_pose import (
    HOME_BASE_TOLERANCE,
    HOME_EEF_TOLERANCE,
    load_home_preset,
    resolve_home_preset_path,
)
from robocasa.demos.live_preview.pre_return.hot_dog_setup import (
    PHASE_CLOSE_FRIDGE,
    PHASE_GOTO_FRIDGE_BASE,
    PHASE_HOLD_AFTER_CLOSE,
)
from robocasa.demos.live_preview.return_home import (
    PHASE_HOLD_END,
    PHASE_RETURN_ARM,
    PHASE_RETURN_BASE,
)
from robocasa.demos.move_pan_live import _get_eef_pose, _get_robot_base_pose
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.utils.dataset_registry_utils import get_ds_path


def verify_hot_dog_setup_home() -> None:
    dataset = Path(get_ds_path("HotDogSetup", source="human", split="pretrain"))
    preset = resolve_home_preset_path("HotDogSetup", None)
    home = load_home_preset(preset)

    env, _ = _make_env_from_dataset(dataset, render=False)
    demo_states, initial_state = _load_episode(dataset, 0)
    ep_meta = json.loads(initial_state["ep_meta"])

    reset_to(env, initial_state)
    demo_end = demo_states[-1]
    reset_to(env, {"states": demo_end})
    fridge = resolve_env_fixture(env, "fridge")
    assert needs_fridge_close(env, fridge), "demo end should have fridge open"

    spec = home_demo_spec_for_task("HotDogSetup")
    verify_stride = 1
    verify_tail = spec.demo_tail_extend if spec is not None else 20

    wrapped, _ = build_home_wrapped_states_for_task(
        env,
        task_name="HotDogSetup",
        demo_states=demo_states,
        initial_state=initial_state,
        home_preset=preset,
        layout=ep_meta.get("layout_id"),
        style=ep_meta.get("style_id"),
        extend_demo_tail=verify_tail,
        demo_stride=verify_stride,
        rebuild_wrap=True,
    )

    pre_return_len = (
        PHASE_GOTO_FRIDGE_BASE
        + PHASE_CLOSE_FRIDGE
        + PHASE_HOLD_AFTER_CLOSE
    )
    return_len = PHASE_RETURN_BASE + PHASE_RETURN_ARM + PHASE_HOLD_END
    demo_len = len(demo_states[::verify_stride]) + verify_tail
    hold_start = 30
    leave_len = 30 + 20

    assert len(wrapped) == hold_start + leave_len + demo_len + pre_return_len + return_len

    pre_return_start = hold_start + leave_len + demo_len
    pre_return_end = pre_return_start + pre_return_len
    reset_to(env, {"states": wrapped[pre_return_end - 1]})
    assert not needs_fridge_close(env, fridge), "fridge should be closed after pre-return"

    for label, idx in (("start", 0), ("end", -1)):
        reset_to(env, {"states": wrapped[idx]})
        base_pos, _ = _get_robot_base_pose(env)
        eef_pos, _, _ = _get_eef_pose(env)
        assert float(np.linalg.norm(base_pos - home.base_pos)) < HOME_BASE_TOLERANCE
        assert float(np.linalg.norm(eef_pos - home.eef_pos)) < HOME_EEF_TOLERANCE

    env.close()
    print(
        f"OK HotDogSetup home wrap ({len(wrapped)} frames, fridge closed before return)"
    )


def main() -> None:
    verify_hot_dog_setup_home()
    print("ALL OK")


if __name__ == "__main__":
    main()
