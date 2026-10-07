import os

# Base directory for member2_activity
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Contract allowed values
ALLOWED_ACTIVITIES = {
    "LOITERING": "loitering",
    "INTRUSION": "restricted_area_intrusion",
    "RUNNING": "running",
    "CROWD": "crowd_formation",
}

ALLOWED_SEVERITIES = ["low", "medium", "high"]

# Restricted Area Polygons (List of [x, y] vertices defining 2D polygons)
# Default demo restricted area zone: Bounded box in standard video space
RESTRICTED_ZONES = [
    {
        "name": "Server Room Entrance",
        "polygon": [[100, 100], [400, 100], [400, 400], [100, 400]],
        "min_dwell_sec": 0.5,
        "severity": "high"
    }
]

# Rule Engine Thresholds
THRESHOLDS = {
    # Loitering: Person stays in a small area for > 5.0 seconds
    "loitering_dwell_time_sec": 5.0,
    "loitering_max_radius_px": 60.0,
    
    # Intrusion: Person stays inside restricted polygon
    "intrusion_dwell_time_sec": 0.5,
    
    # Running: Speed exceeds 150 pixels/second over sliding window
    "running_speed_thresh_px_per_sec": 140.0,
    "running_window_frames": 10,
    
    # Crowd Formation: >= 3 people within 150 pixels radius
    "crowd_min_people": 3,
    "crowd_distance_radius_px": 150.0,
    "crowd_formation_window_sec": 3.0,
}

# Feature Extraction Settings
BUFFER_MAX_FRAMES = 150  # Keep up to 5 seconds of track history at 30 FPS
DEFAULT_FPS = 30.0

# Machine Learning Settings
ML_MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
ML_SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
ML_CONFIDENCE_THRESHOLD = 0.65
USE_HYBRID_MODE = True  # Combine rule engine & ML classifier
