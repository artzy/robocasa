import argparse
import json
import sys
import time
from collections import OrderedDict

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from termcolor import colored

import robocasa.macros as macros
from robocasa.scripts.collect_demos import collect_human_trajectory
from robocasa.wrappers.enclosing_wall_render_wrapper import (
    EnclosingWallRenderWrapper,
    install_enclosing_wall_hotkeys,
)
from robocasa.utils.playback_viewer import (
    PygamePlaybackViewer,
    robosuite_viewer_kwargs,
)


def choose_option(
    options, option_name, show_keys=False, default=None, default_message=None
):
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all tasks

    if default is None:
        default = options[0]

    if default_message is None:
        default_message = default

    # Select environment to run
    print("Here is a list of {}s:\n".format(option_name))

    for i, (k, v) in enumerate(options.items()):
        if show_keys:
            print("[{}] {}: {}".format(i, k, v))
        else:
            print("[{}] {}".format(i, v))
    print()
    try:
        s = input(
            "Choose an option 0 to {}, or any other key for default ({}): ".format(
                len(options) - 1,
                default_message,
            )
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(options) - 1)
        choice = list(options.keys())[k]
    except:
        if default is None:
            choice = options[0]
        else:
            choice = default
        print("Use {} by default.\n".format(choice))

    # Return the chosen environment name
    return choice


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="task (choose among 365 tasks)")
    parser.add_argument(
        "--layout", type=int, help="kitchen layout (choose number 1-60)"
    )
    parser.add_argument("--style", type=int, help="kitchen style (choose number 1-60)")
    parser.add_argument(
        "--device",
        type=str,
        default="keyboard",
        choices=["keyboard", "spacemouse"],
        help="Teleop device (default: keyboard)",
    )
    args = parser.parse_args()

    tasks = OrderedDict(
        [
            ("PickPlaceCounterToCabinet", "pick and place from counter to cabinet"),
            ("PickPlaceCounterToSink", "pick and place from counter to sink"),
            ("PickPlaceMicrowaveToCounter", "pick and place from microwave to counter"),
            ("PickPlaceStoveToCounter", "pick and place from stove to counter"),
            ("OpenSingleDoor", "open cabinet or microwave door"),
            ("CloseDrawer", "close drawer"),
            ("TurnOnMicrowave", "turn on microwave"),
            ("TurnOnSinkFaucet", "turn on sink faucet"),
            ("TurnOnStove", "turn on stove"),
            ("ArrangeVegetables", "arrange vegetables on a cutting board"),
            ("MicrowaveThawing", "place frozen food in microwave for thawing"),
            ("RestockPantry", "restock cans in pantry"),
            ("PreSoakPan", "prepare pan for washing"),
            ("PrepareCoffee", "make coffee"),
        ]
    )

    if args.task is None:
        args.task = choose_option(
            tasks, "task", default="PickPlaceCounterToCabinet", show_keys=True
        )

    # Create argument configuration
    config = {
        "env_name": args.task,
        "robots": "PandaOmron",
        "controller_configs": load_composite_controller_config(robot="PandaOmron"),
        "layout_ids": args.layout,
        "style_ids": args.style,
        "translucent_robot": True,
    }

    onscreen_renderer, viewer_kwargs = robosuite_viewer_kwargs(
        render_camera="robot0_frontview"
    )

    print(colored(f"Initializing environment...", "yellow"))
    env = robosuite.make(
        **config,
        has_renderer=viewer_kwargs["has_renderer"],
        has_offscreen_renderer=viewer_kwargs["has_offscreen_renderer"],
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer=viewer_kwargs["renderer"],
        **(
            {"render_camera": viewer_kwargs["render_camera"]}
            if "render_camera" in viewer_kwargs
            else {}
        ),
    )

    pygame_viewer = None
    if onscreen_renderer == "pygame":
        pygame_viewer = PygamePlaybackViewer(
            camera_name="robot0_frontview",
            width=768,
            height=512,
            title="RoboCasa Teleop",
        )
        print(colored("Opening viewer (pygame window)...", "yellow"))

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)
    env = EnclosingWallRenderWrapper(env, alpha=0.1, enabled=False)
    install_enclosing_wall_hotkeys(env)
    if pygame_viewer is not None:
        env._pygame_viewer = pygame_viewer

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # initialize device
    device = args.device
    if device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env=env, pos_sensitivity=4.0, rot_sensitivity=4.0)
    elif device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=4.0,
            rot_sensitivity=4.0,
            vendor_id=macros.SPACEMOUSE_VENDOR_ID,
            product_id=macros.SPACEMOUSE_PRODUCT_ID,
        )
    else:
        raise ValueError

    # collect demonstrations
    while True:
        ep_directory, discard_traj = collect_human_trajectory(
            env,
            device,
            "right",
            "single-arm-opposed",
            mirror_actions=True,
            render=(onscreen_renderer != "mjviewer"),
            max_fr=30,
            pygame_viewer=pygame_viewer,
        )
        print()
