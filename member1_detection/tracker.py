"""Class-aware intersection-over-union tracker for frame-by-frame detections."""

from dataclasses import dataclass

from detection_types import Detection, TrackedDetection


@dataclass
class _Track:
    object_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    last_seen_frame: int


class IoUTracker:
    """Assign stable IDs by greedily matching same-class boxes across frames."""

    def __init__(self, max_age: int = 30, min_iou: float = 0.3) -> None:
        if max_age < 0:
            raise ValueError("max_age must be non-negative")
        if not 0.0 <= min_iou <= 1.0:
            raise ValueError("min_iou must be between 0 and 1")

        self.max_age = max_age
        self.min_iou = min_iou
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(
        self, detections: list[Detection], frame_number: int
    ) -> list[TrackedDetection]:
        """Match detections to recent tracks and assign IDs to new detections."""
        if frame_number < 1:
            raise ValueError("frame_number must be a positive, 1-based frame index")

        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            for object_id, track in self._tracks.items():
                if track.class_name != detection.class_name:
                    continue
                if frame_number - track.last_seen_frame > self.max_age:
                    continue
                overlap = self._iou(track.bbox, detection.bbox)
                if overlap >= self.min_iou:
                    candidates.append((overlap, detection_index, object_id))

        candidates.sort(reverse=True)
        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()
        detection_ids: dict[int, int] = {}
        for _, detection_index, object_id in candidates:
            if detection_index in matched_detections or object_id in matched_tracks:
                continue
            matched_detections.add(detection_index)
            matched_tracks.add(object_id)
            detection_ids[detection_index] = object_id

        for detection_index, detection in enumerate(detections):
            object_id = detection_ids.get(detection_index)
            if object_id is None:
                object_id = self._next_id
                self._next_id += 1
                detection_ids[detection_index] = object_id
            self._tracks[object_id] = _Track(
                object_id=object_id,
                class_name=detection.class_name,
                bbox=detection.bbox,
                last_seen_frame=frame_number,
            )

        self._tracks = {
            object_id: track
            for object_id, track in self._tracks.items()
            if frame_number - track.last_seen_frame <= self.max_age
        }

        return [
            TrackedDetection(
                object_id=detection_ids[index],
                class_name=detection.class_name,
                confidence=detection.confidence,
                bbox=detection.bbox,
            )
            for index, detection in enumerate(detections)
        ]

    @staticmethod
    def _iou(
        first: tuple[int, int, int, int], second: tuple[int, int, int, int]
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0
