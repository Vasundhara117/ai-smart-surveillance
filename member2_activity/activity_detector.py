import json
from typing import List, Dict, Any
from member2_activity.config import (
    USE_HYBRID_MODE,
    ALLOWED_ACTIVITIES,
    ALLOWED_SEVERITIES
)
from member2_activity.feature_extractor import TrajectoryBuffer, FeatureExtractor
from member2_activity.rule_engine import RuleEngine
from member2_activity.classifier import ActivityClassifier


class ActivityDetector:
    """Main Suspicious Activity Detection Pipeline for Member 2."""

    def __init__(self, use_hybrid: bool = USE_HYBRID_MODE):
        self.buffer = TrajectoryBuffer()
        self.feature_extractor = FeatureExtractor(self.buffer)
        self.rule_engine = RuleEngine(self.buffer)
        self.classifier = ActivityClassifier()
        self.use_hybrid = use_hybrid

    def process_frame(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Processes a single frame detection payload from Member 1.
        
        Input payload format:
        {
          "frame": 120,
          "timestamp": 4.0,
          "objects": [
            {
              "id": 7,
              "class": "person",
              "confidence": 0.91,
              "bbox": [120, 80, 250, 400]
            }
          ]
        }
        """
        frame_num = int(frame_data.get("frame", 0))
        timestamp = float(frame_data.get("timestamp", 0.0))
        objects = frame_data.get("objects", [])

        # Update tracking buffer
        self.buffer.update(frame_num, timestamp, objects)

        frame_events = []

        # 1. Evaluate Crowd Rules across all active people in the frame
        crowd_features = self.feature_extractor.extract_crowd_features(objects, timestamp)
        crowd_events = self.rule_engine.evaluate_crowd_rules(crowd_features, timestamp)
        frame_events.extend(crowd_events)

        # 2. Evaluate Individual Person Rules & ML Classifier
        for obj in objects:
            if obj.get("class") != "person":
                continue

            person_id = int(obj["id"])
            det_conf = float(obj.get("confidence", 0.90))

            features = self.feature_extractor.extract_person_features(person_id, timestamp)
            if not features:
                continue

            # Evaluate Rule Engine
            person_rule_events = self.rule_engine.evaluate_person_rules(
                person_id, features, det_conf, timestamp
            )
            frame_events.extend(person_rule_events)

            # Evaluate ML Classifier if in Hybrid mode or if rule engine did not trigger
            if self.use_hybrid and self.classifier.is_trained() and not person_rule_events:
                ml_result = self.classifier.predict(features)
                if ml_result:
                    act_label, ml_conf = ml_result
                    if act_label in ALLOWED_ACTIVITIES.values():
                        # Determine severity based on feature intensity
                        severity = "medium"
                        if features.get("dwell_time", 0) > 10.0 or features.get("max_speed", 0) > 220.0 or features.get("in_restricted_zone", 0) == 1.0:
                            severity = "high"
                        elif features.get("dwell_time", 0) < 6.0 and features.get("max_speed", 0) < 160.0:
                            severity = "low"

                        frame_events.append({
                            "timestamp": float(timestamp),
                            "person_id": int(person_id),
                            "activity": str(act_label),
                            "confidence": float(ml_conf),
                            "suspicious": True,
                            "severity": str(severity)
                        })

        # Validate output schema adherence and convert numpy types to native python types
        validated_events = []
        for ev in frame_events:
            act_str = str(ev.get("activity"))
            sev_str = str(ev.get("severity"))
            if (
                isinstance(ev.get("timestamp"), (int, float)) and
                isinstance(ev.get("person_id"), int) and
                act_str in ALLOWED_ACTIVITIES.values() and
                isinstance(ev.get("confidence"), (int, float)) and
                isinstance(ev.get("suspicious"), bool) and
                sev_str in ALLOWED_SEVERITIES
            ):
                validated_events.append({
                    "timestamp": float(ev["timestamp"]),
                    "person_id": int(ev["person_id"]),
                    "activity": act_str,
                    "confidence": float(ev["confidence"]),
                    "suspicious": bool(ev["suspicious"]),
                    "severity": sev_str
                })

        return validated_events

    def process_stream(self, frame_stream: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes a list of frame detection dictionaries sequentially."""
        all_events = []
        for frame_data in frame_stream:
            events = self.process_frame(frame_data)
            all_events.extend(events)
        return all_events

    def process_json_file(self, input_file_path: str, output_file_path: str) -> List[Dict[str, Any]]:
        """Reads detection JSON input file, processes stream, and writes activity_events.json."""
        with open(input_file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            frame_stream = [data]
        elif isinstance(data, list):
            frame_stream = data
        else:
            raise ValueError("Invalid JSON format. Expected dict or list of frame dicts.")

        events = self.process_stream(frame_stream)

        with open(output_file_path, "w") as f:
            json.dump(events, f, indent=2)

        return events
