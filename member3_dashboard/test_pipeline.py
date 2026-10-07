import json

import pytest

from member3_dashboard.pipeline import run_video_analysis, validate_activity_events, validate_detection_json


def test_validate_detection_json_accepts_expected_schema():
    payload = [
        {
            "frame": 1,
            "timestamp": 0.0,
            "objects": [{"id": 7, "class": "person", "confidence": 0.91, "bbox": [120, 80, 250, 400]}],
        }
    ]
    assert validate_detection_json(payload) is True


def test_validate_activity_events_accepts_expected_schema():
    payload = [{"timestamp": 12.4, "person_id": 7, "activity": "loitering", "confidence": 0.89, "suspicious": True, "severity": "high"}]
    assert validate_activity_events(payload) is True


def test_run_video_analysis_requires_real_video_file(tmp_path):
    missing = tmp_path / "missing.mp4"
    with pytest.raises(FileNotFoundError):
        run_video_analysis(str(missing), tmp_path)
