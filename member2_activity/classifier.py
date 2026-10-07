import os
import joblib
import numpy as np
from typing import Dict, Any, Optional, Tuple
from member2_activity.config import (
    ML_MODEL_PATH,
    ML_SCALER_PATH,
    ML_CONFIDENCE_THRESHOLD,
    ALLOWED_ACTIVITIES
)

FEATURE_KEYS = [
    "instantaneous_speed",
    "avg_speed",
    "max_speed",
    "acceleration",
    "net_displacement",
    "path_length",
    "linearity",
    "bounding_radius",
    "dwell_time",
    "dist_to_restricted",
    "in_restricted_zone"
]


class ActivityClassifier:
    """Machine Learning Activity Classifier."""

    def __init__(self, model_path: str = ML_MODEL_PATH, scaler_path: str = ML_SCALER_PATH):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.classes_ = None
        self._load_or_initialize()

    def _load_or_initialize(self):
        """Loads model and scaler from disk if available."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.classes_ = list(self.model.classes_)
            except Exception as e:
                print(f"[Warning] Failed to load ML model from {self.model_path}: {e}")
                self.model = None
                self.scaler = None

    def is_trained(self) -> bool:
        return self.model is not None and self.scaler is not None

    def feature_dict_to_vector(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Converts feature dictionary to standard 2D numpy array for ML model."""
        vec = [feature_dict.get(k, 0.0) for k in FEATURE_KEYS]
        return np.array(vec, dtype=np.float32).reshape(1, -1)

    def predict(self, feature_dict: Dict[str, float]) -> Optional[Tuple[str, float]]:
        """Predicts activity class label and probability confidence using ML model."""
        if not self.is_trained():
            return None

        X = self.feature_dict_to_vector(feature_dict)
        X_scaled = self.scaler.transform(X)

        probs = self.model.predict_proba(X_scaled)[0]
        max_idx = np.argmax(probs)
        label = self.classes_[max_idx]
        confidence = float(probs[max_idx])

        if label == "normal" or confidence < ML_CONFIDENCE_THRESHOLD:
            return None

        return label, round(confidence, 2)
