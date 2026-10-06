"""Shared records used by the detector, tracker, and video processor."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class TrackedDetection:
    object_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
