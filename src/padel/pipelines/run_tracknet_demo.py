from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from src.padel.tracking import TrackVNETHandler


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--video", type=Path, required=True, help="Path to input video.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
    )
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    handler = TrackVNETHandler()
    count = 0

    for det in handler.run_on_video(args.video):
        print(f"frame={det.frame_index} visible={det.visible} x={det.x} y={det.y}")
        count += 1
        if args.max_frames and count >= args.max_frames:
            break

    print(f"Processed {count} frames from {args.video}")


if __name__ == "__main__":
    main()
