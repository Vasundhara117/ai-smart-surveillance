import os
import unittest
import numpy as np
from member2_activity.config import (
    ALLOWED_ACTIVITIES,
    ALLOWED_SEVERITIES,
    RESTRICTED_ZONES
)
from member2_activity.feature_extractor import (
    TrajectoryBuffer,
    FeatureExtractor,
    point_in_polygon
)
from member2_activity.rule_engine import RuleEngine
from member2_activity.classifier import ActivityClassifier
from member2_activity.activity_detector import ActivityDetector
from member2_activity.dataset_generator import generate_synthetic_dataset
from member2_activity.train_model import train_and_evaluate_model


class TestMember2ActivityDetector(unittest.TestCase):

    def setUp(self):
        self.detector = ActivityDetector(use_hybrid=True)

    def test_point_in_polygon(self):
        poly = [[100, 100], [400, 100], [400, 400], [100, 400]]
        self.assertTrue(point_in_polygon((200, 200), poly))
        self.assertFalse(point_in_polygon((50, 50), poly))
        self.assertFalse(point_in_polygon((450, 450), poly))

    def test_loitering_activity(self):
        """Tests that a stationary/low-movement person triggers a loitering event after dwell time."""
        events = []
        person_id = 7
        # Feed 7 seconds of frames at 1 frame per second
        for i in range(8):
            t = float(i)
            frame_data = {
                "frame": i * 30,
                "timestamp": t,
                "objects": [
                    {
                        "id": person_id,
                        "class": "person",
                        "confidence": 0.95,
                        "bbox": [500 + (i % 2), 500 + (i % 2), 550 + (i % 2), 650 + (i % 2)]
                    }
                ]
            }
            evs = self.detector.process_frame(frame_data)
            events.extend(evs)

        loitering_events = [e for e in events if e["activity"] == ALLOWED_ACTIVITIES["LOITERING"]]
        self.assertGreater(len(loitering_events), 0, "Loitering event should be triggered")
        first_event = loitering_events[0]
        self.assertEqual(first_event["person_id"], person_id)
        self.assertTrue(first_event["suspicious"])
        self.assertIn(first_event["severity"], ALLOWED_SEVERITIES)

    def test_restricted_area_intrusion_activity(self):
        """Tests that entering the restricted zone triggers an intrusion event."""
        person_id = 12
        zone_poly = RESTRICTED_ZONES[0]["polygon"]
        inside_x = (zone_poly[0][0] + zone_poly[1][0]) / 2.0
        inside_y = (zone_poly[0][1] + zone_poly[2][1]) / 2.0

        events = []
        for i in range(3):
            t = float(i)
            frame_data = {
                "frame": i * 15,
                "timestamp": t,
                "objects": [
                    {
                        "id": person_id,
                        "class": "person",
                        "confidence": 0.92,
                        "bbox": [inside_x - 20, inside_y - 20, inside_x + 20, inside_y + 40]
                    }
                ]
            }
            evs = self.detector.process_frame(frame_data)
            events.extend(evs)

        intrusion_events = [e for e in events if e["activity"] == ALLOWED_ACTIVITIES["INTRUSION"]]
        self.assertGreater(len(intrusion_events), 0, "Intrusion event should be triggered")
        self.assertEqual(intrusion_events[0]["person_id"], person_id)
        self.assertEqual(intrusion_events[0]["severity"], "high")

    def test_running_activity(self):
        """Tests high-speed movement triggering running event."""
        person_id = 99
        events = []
        # Move person rapidly across screen (e.g. 200px per second)
        for i in range(5):
            t = float(i) * 0.5
            x = 50 + i * 120  # 240px / sec speed
            frame_data = {
                "frame": i * 15,
                "timestamp": t,
                "objects": [
                    {
                        "id": person_id,
                        "class": "person",
                        "confidence": 0.90,
                        "bbox": [x, 50, x + 40, 150]
                    }
                ]
            }
            evs = self.detector.process_frame(frame_data)
            events.extend(evs)

        running_events = [e for e in events if e["activity"] == ALLOWED_ACTIVITIES["RUNNING"]]
        self.assertGreater(len(running_events), 0, "Running event should be triggered")
        self.assertEqual(running_events[0]["person_id"], person_id)

    def test_crowd_formation_activity(self):
        """Tests crowd formation when multiple people gather closely."""
        events = []
        frame_data = {
            "frame": 100,
            "timestamp": 5.0,
            "objects": [
                {"id": 101, "class": "person", "confidence": 0.90, "bbox": [600, 600, 640, 720]},
                {"id": 102, "class": "person", "confidence": 0.91, "bbox": [610, 605, 650, 725]},
                {"id": 103, "class": "person", "confidence": 0.89, "bbox": [605, 615, 645, 735]},
                {"id": 104, "class": "person", "confidence": 0.92, "bbox": [615, 610, 655, 730]}
            ]
        }
        evs = self.detector.process_frame(frame_data)
        events.extend(evs)

        crowd_events = [e for e in events if e["activity"] == ALLOWED_ACTIVITIES["CROWD"]]
        self.assertGreater(len(crowd_events), 0, "Crowd formation event should be triggered")
        self.assertTrue(crowd_events[0]["suspicious"])

    def test_ml_dataset_generation_and_training(self):
        """Tests dataset generation and ML classifier model training."""
        X, y, feature_names = generate_synthetic_dataset(num_samples_per_class=50)
        self.assertEqual(X.shape[0], 250)
        self.assertEqual(X.shape[1], 11)

        clf, scaler = train_and_evaluate_model()
        self.assertIsNotNone(clf)
        self.assertIsNotNone(scaler)

    def test_output_contract_strictness(self):
        """Validates strict adherence to Member 3 interface contract."""
        frame_data = {
            "frame": 200,
            "timestamp": 10.0,
            "objects": [
                {"id": 1, "class": "person", "confidence": 0.95, "bbox": [200, 200, 250, 350]},
                {"id": 2, "class": "person", "confidence": 0.95, "bbox": [205, 205, 255, 355]},
                {"id": 3, "class": "person", "confidence": 0.95, "bbox": [210, 210, 260, 360]}
            ]
        }

        events = self.detector.process_frame(frame_data)
        required_keys = {"timestamp", "person_id", "activity", "confidence", "suspicious", "severity"}

        for event in events:
            self.assertEqual(set(event.keys()), required_keys, f"Event keys mismatch: {event.keys()}")
            self.assertIsInstance(event["timestamp"], float)
            self.assertIsInstance(event["person_id"], int)
            self.assertIn(event["activity"], ALLOWED_ACTIVITIES.values())
            self.assertIsInstance(event["confidence"], float)
            self.assertIsInstance(event["suspicious"], bool)
            self.assertIn(event["severity"], ALLOWED_SEVERITIES)


if __name__ == "__main__":
    unittest.main()
