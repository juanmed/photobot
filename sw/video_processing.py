import argparse
from pathlib import Path

import cv2


def process_video(video_path: Path) -> None:
    """Open a video file and print frame index and resolution for each frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        print(f"{video_path.name} frame {frame_idx}: {width}x{height}")
        frame_idx += 1

    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read .mp4 videos from a directory and print frame info."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing .mp4 files.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    video_paths = sorted(args.input_dir.glob("*.mp4"))
    if not video_paths:
        print(f"No .mp4 files found in {args.input_dir}")
        return

    for video_path in video_paths:
        process_video(video_path)


if __name__ == "__main__":
    main()
