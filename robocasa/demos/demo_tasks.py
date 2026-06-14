import argparse
import os
from collections import OrderedDict

from termcolor import colored

import robocasa
from robocasa.demos.live_preview.home_pose import resolve_home_preset_path
from robocasa.demos.live_preview.registry import HOME_DEMO_TASKS, LIVE_DEMO_REGISTRY, LIVE_DEMO_TASKS
from robocasa.demos.live_preview.home_demo_playback import play_human_demo_with_home
from robocasa.scripts.download_datasets import download_datasets
from robocasa.scripts.dataset_scripts.playback_dataset import playback_dataset
from robocasa.utils.dataset_registry_utils import get_ds_path
from robocasa.utils.playback_viewer import (
    DEFAULT_VIEWER_HEIGHT,
    DEFAULT_VIEWER_WIDTH,
)


def get_ds_path_any_split(task, source="human"):
    """Return dataset path trying pretrain first, then target (for tasks that only have target human demos, e.g. composite-unseen)."""
    path = get_ds_path(task, source=source, split="pretrain")
    if path is not None:
        return path
    return get_ds_path(task, source=source, split="target")


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
    print("{}s:".format(option_name.capitalize()))

    for i, (k, v) in enumerate(options.items()):
        if show_keys:
            print("[{}] {}: {}".format(i, k, v))
        else:
            print("[{}] {}".format(i, v))
    print()
    try:
        s = input(
            "Choose an option 0 to {}, [q] to quit, or any other key for default ({}): ".format(
                len(options) - 1,
                default_message,
            )
        )
        if s.strip().lower() == "q":
            return None
        # parse input into a number within range
        k = min(max(int(s), 0), len(options) - 1)
        choice = list(options.keys())[k]
    except ValueError:
        if default is None:
            choice = options[0]
        else:
            choice = default
        print("Use {} by default.\n".format(choice))
    except (EOFError, KeyboardInterrupt):
        return None

    # Return the chosen environment name
    return choice


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, help="task (must be task with demos collected already)"
    )
    parser.add_argument(
        "--render_offscreen",
        action="store_true",
        help="off-screen rendering",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/tmp/robocasa_demo_tasks",
        help="path to video folder for offscreen rendering.",
    )
    parser.add_argument(
        "--source-fixture",
        type=str,
        default="counter",
        help="MovePan source fixture (counter, sink, stove, etc.)",
    )
    parser.add_argument(
        "--target-fixture",
        type=str,
        default="stove",
        help="MovePan target fixture (default: stove)",
    )
    parser.add_argument(
        "--obj-registries",
        type=str,
        default="coppelia_edu",
        help="MovePan object registries (comma-separated)",
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="MovePan: use keyboard teleop instead of live preview",
    )
    parser.add_argument("--layout", type=int, default=None, help="MovePan kitchen layout id")
    parser.add_argument("--style", type=int, default=None, help="MovePan kitchen style id")
    parser.add_argument(
        "--home-preset",
        type=str,
        default=None,
        help="Live preview: JSON home pose preset (default: task-specific preset)",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default=None,
        choices=("mjviewer", "mujoco", "pygame"),
        help="Home demo on-screen renderer (default: mjviewer for smooth GPU playback)",
    )
    parser.add_argument(
        "--playback-fps",
        type=float,
        default=60,
        help="Home demo playback frame rate (default: 60)",
    )
    parser.add_argument(
        "--demo-stride",
        type=int,
        default=None,
        help="Subsample human demo frames during home wrap (default: task registry)",
    )
    parser.add_argument(
        "--viewer-width",
        type=int,
        default=1280,
        help="Pygame viewer width (default: 1280)",
    )
    parser.add_argument(
        "--viewer-height",
        type=int,
        default=720,
        help="Pygame viewer height (default: 720)",
    )
    parser.add_argument(
        "--no-dwell",
        action="store_true",
        help="Skip Home start/end dwell pause",
    )
    parser.add_argument(
        "--rebuild-wrap",
        action="store_true",
        help="Ignore cached home-wrapped states and rebuild",
    )
    args = parser.parse_args()

    all_tasks = OrderedDict(
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
            (
                "HotDogSetup",
                "gather ingredients for a hot dog and place them on the dining table [Human demo + Home start/end]",
            ),
            (
                "DeliverStraw",
                "deliver straw to glass cup [Human demo + Home start/end]",
            ),
            (
                "MovePan",
                "move coppelia pan [Live preview: Home start/end - NOT human demo]",
            ),
            (
                "GatherTableware",
                "gather tableware from around the kitchen [Navigation]",
            ),
        ]
    )
    tasks = OrderedDict(
        (k, v)
        for k, v in all_tasks.items()
        if k in LIVE_DEMO_TASKS
        or k in HOME_DEMO_TASKS
        or get_ds_path_any_split(k, source="human") is not None
    )
    if not tasks:
        raise RuntimeError(
            "No tasks with registered human demo paths or live demo tasks. "
            "Check dataset registry and DATASET_BASE_PATH."
        )

    video_num = -1
    while True:
        if args.task is None:
            task = choose_option(
                tasks, "task", default=list(tasks.keys())[0], show_keys=True
            )
            if task is None:
                break
        else:
            task = args.task
        video_num += 1

        if args.render_offscreen:
            if not os.path.exists(args.video_path):
                os.makedirs(args.video_path)
            video_path = os.path.join(args.video_path, f"video_{video_num}.mp4")
        else:
            video_path = False

        if task in LIVE_DEMO_TASKS:
            spec = LIVE_DEMO_REGISTRY[task]
            home_preset = resolve_home_preset_path(task, args.home_preset)
            if home_preset is not None and not home_preset.exists():
                print(
                    colored(
                        f"warning: home preset not found ({home_preset}); "
                        "MovePan will use reset pose as Home",
                        "yellow",
                    )
                )
            print(
                colored(
                    f"Live preview '{task}' - Home preset: "
                    f"{home_preset if home_preset else 'default'}",
                    "cyan",
                )
            )
            spec.play(
                source_fixture=args.source_fixture,
                target_fixture=args.target_fixture,
                obj_registries=args.obj_registries,
                teleop=args.teleop,
                render_offscreen=args.render_offscreen,
                video_path=video_path,
                layout=args.layout,
                style=args.style,
                seed=0,
                home_preset=str(home_preset) if home_preset else None,
            )
            if args.task is not None:
                break
            print()
            continue

        if task in HOME_DEMO_TASKS:
            home_preset = resolve_home_preset_path(task, args.home_preset)
            if home_preset is not None and not home_preset.exists():
                print(
                    colored(
                        f"warning: home preset not found ({home_preset}); "
                        f"{task} will use reset pose as Home",
                        "yellow",
                    )
                )
            print(
                colored(
                    f"Home-wrapped demo '{task}' - preset: "
                    f"{home_preset if home_preset else 'default'}",
                    "cyan",
                )
            )
            dataset = get_ds_path_any_split(task, source="human")
            if dataset is None:
                raise ValueError(f"No registered dataset path for task={task} source=human")
            if not os.path.exists(dataset):
                print(
                    colored(
                        "Unable to find dataset locally. Downloading...", color="yellow"
                    )
                )
                download_datasets(
                    tasks=[task], split=["pretrain", "target"], source=["human"]
                )
            play_human_demo_with_home(
                task,
                dataset=dataset,
                render_offscreen=args.render_offscreen,
                video_path=video_path,
                home_preset=str(home_preset) if home_preset else None,
                layout=args.layout,
                style=args.style,
                demo_stride=args.demo_stride,
                playback_fps=args.playback_fps,
                renderer=args.renderer,
                viewer_width=args.viewer_width,
                viewer_height=args.viewer_height,
                no_dwell=args.no_dwell,
                rebuild_wrap=args.rebuild_wrap,
            )
            if args.task is not None:
                break
            print()
            continue

        dataset = get_ds_path_any_split(task, source="human")
        if dataset is None:
            raise ValueError(f"No registered dataset path for task={task} source=human")

        if os.path.exists(dataset) is False:
            # download dataset files (try both splits so target-only tasks e.g. GatherTableware work)
            print(
                colored(
                    "Unable to find dataset locally. Downloading...", color="yellow"
                )
            )
            download_datasets(
                tasks=[task], split=["pretrain", "target"], source=["human"]
            )

        render = not args.render_offscreen
        use_actions = False
        use_abs_actions = False
        render_image_names = ["robot0_agentview_center"]
        use_obs = False
        n = 1 if args.task is None else None
        filter_key = None
        video_skip = 5
        first = False
        verbose = True
        extend_states = True
        camera_height = DEFAULT_VIEWER_HEIGHT
        camera_width = DEFAULT_VIEWER_WIDTH

        playback_dataset(
            dataset=dataset,
            use_actions=use_actions,
            use_abs_actions=use_abs_actions,
            use_obs=use_obs,
            filter_key=filter_key,
            n=n,
            render=render,
            render_image_names=render_image_names,
            camera_height=camera_height,
            camera_width=camera_width,
            video_path=video_path,
            video_skip=video_skip,
            extend_states=extend_states,
            first=first,
            verbose=verbose,
        )
        if args.task is not None:
            break
        print()
