import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from member2_activity.activity_detector import ActivityDetector


SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}


def validate_detection_json(payload: Any) -> bool:
    """Validate the Member 1 detection.json contract."""
    if not isinstance(payload, list):
        return False

    for record in payload:
        if not isinstance(record, dict):
            return False
        if not {"frame", "timestamp", "objects"}.issubset(record):
            return False
        if not isinstance(record["objects"], list):
            return False
        for obj in record["objects"]:
            if not isinstance(obj, dict):
                return False
            required = {"id", "class", "confidence", "bbox"}
            if not required.issubset(obj):
                return False
            if not isinstance(obj["bbox"], list) or len(obj["bbox"]) != 4:
                return False
            if not isinstance(obj["confidence"], (int, float)):
                return False
    return True


def validate_activity_events(payload: Any) -> bool:
    """Validate the Member 2 activity_events.json contract."""
    if not isinstance(payload, list):
        return False

    for event in payload:
        if not isinstance(event, dict):
            return False
        required = {"timestamp", "person_id", "activity", "confidence", "suspicious", "severity"}
        if not required.issubset(event):
            return False
        if not isinstance(event["timestamp"], (int, float)):
            return False
        if not isinstance(event["person_id"], int):
            return False
        if not isinstance(event["confidence"], (int, float)):
            return False
        if not isinstance(event["suspicious"], bool):
            return False
        if not isinstance(event["severity"], str):
            return False
    return True


def run_video_analysis(video_path: str, working_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run the complete real pipeline: video -> Member 1 -> Member 2 -> dashboard events."""
    source = Path(video_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if source.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video type: {source.suffix or 'unknown'}")

    project_root = Path(__file__).resolve().parent.parent
    member1_dir = project_root / "member1_detection"
    output_dir = Path(working_dir) if working_dir else project_root / "member3_dashboard" / "generated_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    detection_path = output_dir / "detection.json"
    annotated_path = output_dir / "annotated_output.mp4"

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(project_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    command = [
        sys.executable,
        "video_processor.py",
        "--input",
        str(source),
        "--output",
        str(detection_path),
        "--annotated",
        str(annotated_path),
    ]

    result = subprocess.run(
        command,
        cwd=str(member1_dir),
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "Member 1 video processing failed."
        raise RuntimeError(message)

    if not detection_path.exists():
        raise FileNotFoundError("Member 1 completed without creating detection.json.")

    with detection_path.open("r", encoding="utf-8") as handle:
        detection_payload = json.load(handle)

    if not validate_detection_json(detection_payload):
        raise ValueError("Member 1 generated invalid detection.json content.")

    activity_path = output_dir / "activity_events.json"
    detector = ActivityDetector()
    activity_events = detector.process_json_file(str(detection_path), str(activity_path))

    if not activity_path.exists():
        raise FileNotFoundError("Member 2 completed without creating activity_events.json.")

    with activity_path.open("r", encoding="utf-8") as handle:
        activity_payload = json.load(handle)

    if not validate_activity_events(activity_payload):
        raise ValueError("Member 2 generated invalid activity_events.json content.")

    return {
        "detection_path": str(detection_path),
        "activity_path": str(activity_path),
        "annotated_video_path": str(annotated_path) if annotated_path.exists() else None,
        "detections": detection_payload,
        "activity_events": activity_events,
    }
