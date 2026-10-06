"""Configuration defaults for the detection and tracking module."""

DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "backpack",
    "handbag",
    "suitcase",
)
DEFAULT_MAX_AGE = 30
DEFAULT_MIN_IOU = 0.3
