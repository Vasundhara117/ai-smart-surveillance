"""Pretrained YOLO object detector."""

from collections.abc import Sequence

from detection_types import Detection


class ObjectDetector:
    """Run a pretrained Ultralytics YOLO model on individual video frames."""

    def __init__(
        self,
        model_path: str,
        class_names: Sequence[str] | None = None,
        confidence_threshold: float = 0.25,
        device: str = "auto",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is required for object detection. "
                "Install dependencies with: python -m pip install -r requirements.txt"
            ) from error

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = None if device == "auto" else device
        self.class_ids = self._resolve_classes(class_names)

    def _resolve_classes(self, class_names: Sequence[str] | None) -> list[int] | None:
        if class_names is None:
            return None

        model_names = self.model.names
        if isinstance(model_names, list):
            name_to_id = {name: class_id for class_id, name in enumerate(model_names)}
        else:
            name_to_id = {name: class_id for class_id, name in model_names.items()}

        unknown = sorted(set(class_names) - name_to_id.keys())
        if unknown:
            available = ", ".join(sorted(name_to_id))
            raise ValueError(
                f"Unknown model class(es): {', '.join(unknown)}. "
                f"Available classes: {available}"
            )
        return [name_to_id[name] for name in class_names]

    def detect(self, frame: object) -> list[Detection]:
        """Return detections for one BGR frame, without assigning tracking IDs."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                coordinates = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        class_name=str(self.model.names[class_id]),
                        confidence=float(box.conf[0].item()),
                        bbox=tuple(int(round(value)) for value in coordinates),
                    )
                )
        return detections
