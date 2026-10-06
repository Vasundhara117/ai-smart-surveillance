"""Dependency-free tests for tracking IDs and the JSON record interface."""

import io
import json
import unittest

from detection_types import Detection
from tracker import IoUTracker
from video_processor import write_frame_record


class IoUTrackerTests(unittest.TestCase):
    def test_keeps_id_for_overlapping_detection(self) -> None:
        tracker = IoUTracker()
        first = tracker.update([Detection("person", 0.9, (10, 10, 50, 80))], 1)
        second = tracker.update([Detection("person", 0.8, (12, 12, 52, 82))], 2)

        self.assertEqual(first[0].object_id, second[0].object_id)
        self.assertEqual(first[0].object_id, 1)

    def test_does_not_match_different_classes(self) -> None:
        tracker = IoUTracker()
        first = tracker.update([Detection("person", 0.9, (10, 10, 50, 80))], 1)
        second = tracker.update([Detection("car", 0.8, (10, 10, 50, 80))], 2)

        self.assertNotEqual(first[0].object_id, second[0].object_id)

    def test_new_detection_does_not_steal_an_existing_track_id(self) -> None:
        tracker = IoUTracker()
        first = tracker.update(
            [
                Detection("person", 0.9, (10, 10, 50, 80)),
                Detection("person", 0.9, (100, 10, 140, 80)),
            ],
            1,
        )
        second = tracker.update(
            [
                Detection("person", 0.9, (200, 10, 240, 80)),
                Detection("person", 0.9, (102, 10, 142, 80)),
            ],
            2,
        )

        self.assertEqual(first[1].object_id, second[1].object_id)
        self.assertNotIn(second[0].object_id, {item.object_id for item in first})

    def test_assigns_new_id_after_track_expires(self) -> None:
        tracker = IoUTracker(max_age=1)
        first = tracker.update([Detection("person", 0.9, (10, 10, 50, 80))], 1)
        tracker.update([], 2)
        second = tracker.update([Detection("person", 0.8, (10, 10, 50, 80))], 4)

        self.assertNotEqual(first[0].object_id, second[0].object_id)

    def test_json_record_has_fixed_interface(self) -> None:
        objects = IoUTracker().update(
            [Detection("person", 0.91234, (1, 2, 30, 40))], 1
        )
        output = io.StringIO()
        write_frame_record(output, 1, 0.0, objects)
        record = json.loads(output.getvalue())

        self.assertEqual(set(record), {"frame", "timestamp", "objects"})
        self.assertEqual(
            set(record["objects"][0]), {"id", "class", "confidence", "bbox"}
        )
        self.assertEqual(record["objects"][0]["bbox"], [1, 2, 30, 40])
        self.assertEqual(record["objects"][0]["confidence"], 0.9123)


if __name__ == "__main__":
    unittest.main()
