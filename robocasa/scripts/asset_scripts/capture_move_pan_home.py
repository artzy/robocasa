"""Capture MovePan preview home pose (base + EEF) into a JSON preset."""

from __future__ import annotations

import argparse
from pathlib import Path

import robocasa  # noqa: F401
from robocasa.demos.move_pan_live import (
    DEFAULT_HOME_PRESET,
    PREVIEW_DEFAULT_LAYOUT,
    PREVIEW_DEFAULT_SEED,
    PREVIEW_DEFAULT_STYLE,
    capture_home_from_env,
    make_move_pan_env,
    save_home_preset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture MovePan home pose preset.")
    parser.add_argument("--source-fixture", type=str, default="counter")
    parser.add_argument("--target-fixture", type=str, default="stove")
    parser.add_argument("--layout", type=int, default=PREVIEW_DEFAULT_LAYOUT)
    parser.add_argument("--style", type=int, default=PREVIEW_DEFAULT_STYLE)
    parser.add_argument("--seed", type=int, default=PREVIEW_DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_HOME_PRESET),
        help="Output JSON path for home preset.",
    )
    args = parser.parse_args()

    env, _, _ = make_move_pan_env(
        source_fixture=args.source_fixture,
        target_fixture=args.target_fixture,
        layout=args.layout,
        style=args.style,
        seed=args.seed,
        render_offscreen=True,
    )
    env.reset()

    home = capture_home_from_env(
        env,
        name=f"layout{args.layout}_style{args.style}_seed{args.seed}_{args.source_fixture}_{args.target_fixture}",
        layout=args.layout,
        style=args.style,
        seed=args.seed,
        source_fixture=args.source_fixture,
        target_fixture=args.target_fixture,
    )
    out_path = save_home_preset(home, args.output)
    env.close()
    print(f"Saved MovePan home preset to {out_path}")


if __name__ == "__main__":
    main()
