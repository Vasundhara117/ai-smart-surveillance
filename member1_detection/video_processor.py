"""CLI for converting video into the per-frame detection JSON interface."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import TextIO

from config import DEFAULT_CLASSES, DEFAULT_MAX_AGE, DEFAULT_MIN_IOU, DEFAULT_MODEL
from detector import ObjectDetector
from detection_types import TrackedDetection
from tracker import IoUTracker


def write_frame_record(
    output: TextIO,
    frame_number: int,
    timestamp: float,
    objects: list[TrackedDetection],
) -> None:
    """Write one record matching the fixed Member 2 JSON interface."""
    record = {
        "frame": frame_number,
        "timestamp": round(timestamp, 3),
        "objects": [
            {
                "id": item.object_id,
                "class": item.class_name,
                "confidence": round(item.confidence, 4),
                "bbox": list(item.bbox),
            }
            for item in objects
        ],
    }
    output.write(json.dumps(record, separators=(",", ":")))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and track objects in a video; write per-frame JSON."
    )
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument(
        "--output", default="detection.json", help="Output JSON array (default: detection.json)"
    )
    parser.add_argument(
        "--annotated",
        help="Optional output video path for frames annotated with boxes and IDs",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics YOLO weights")
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated model class names, or 'all' (default: common surveillance objects)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.25, help="Minimum detection confidence"
    )
    parser.add_argument("--device", default="auto", help="Inference device, e.g. auto, cpu, 0")
    parser.add_argument(
        "--max-age", type=int, default=DEFAULT_MAX_AGE,
        help="Frames to retain a temporarily missed track",
    )
    parser.add_argument(
        "--min-iou", type=float, default=DEFAULT_MIN_IOU,
        help="Minimum box IoU to match a detection to a track",
    )
    return parser.parse_args()


def _draw_detections(
    frame: object, objects: list[TrackedDetection], cv2_module: object
) -> object:
    for item in objects:
        x1, y1, x2, y2 = item.bbox
        color = (
            (37 * item.object_id) % 255,
            (97 * item.object_id) % 255,
            (173 * item.object_id) % 255,
        )
        cv2_module.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{item.class_name} ID:{item.object_id} {item.confidence:.2f}"
        cv2_module.putText(
            frame, label, (x1, max(20, y1 - 7)),
            cv2_module.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )
    return frame


def process_video(args: argparse.Namespace) -> int:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for video processing. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from error

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")

    class_names = None
    if args.classes.strip().lower() != "all":
        class_names = [name.strip() for name in args.classes.split(",") if name.strip()]
        if not class_names:
            raise ValueError("--classes must contain at least one class name or 'all'")

    tracker = IoUTracker(max_age=args.max_age, min_iou=args.min_iou)
    detector = ObjectDetector(
        model_path=args.model,
        class_names=class_names,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open the input video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError("Input video does not provide a valid frame rate")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.annotated:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        annotated_path = Path(args.annotated)
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"OpenCV could not create annotated video: {annotated_path}")

    start_time = time.perf_counter()
    frame_number = 0
    try:
        with output_path.open("w", encoding="utf-8") as output:
            output.write("[")
            first_record = True
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame_number += 1
                detections = detector.detect(frame)
                tracked = tracker.update(detections, frame_number)
                if not first_record:
                    output.write(",")
                write_frame_record(output, frame_number, (frame_number - 1) / fps, tracked)
                first_record = False
                if writer is not None:
                    writer.write(_draw_detections(frame, tracked, cv2))
            output.write("]\n")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - start_time
    throughput = frame_number / elapsed if elapsed > 0 else 0.0
    print(
        f"Processed {frame_number} frames in {elapsed:.2f}s "
        f"({throughput:.2f} FPS); detections saved to {output_path}",
        file=sys.stderr,
    )
    if args.annotated:
        print(f"Annotated video saved to {args.annotated}", file=sys.stderr)
    return 0


def main() -> int:
    try:
        return process_video(_parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
