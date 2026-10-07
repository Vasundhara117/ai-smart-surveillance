import numpy as np
from typing import Tuple, List
from member2_activity.classifier import FEATURE_KEYS


def generate_synthetic_dataset(num_samples_per_class: int = 250, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generates synthetic dataset of feature vectors for ML model training and evaluation."""
    np.random.seed(seed)

    classes = ["normal", "loitering", "restricted_area_intrusion", "running", "crowd_formation"]
    X_list = []
    y_list = []

    for label in classes:
        for _ in range(num_samples_per_class):
            if label == "normal":
                inst_speed = np.random.uniform(20.0, 75.0)
                avg_speed = inst_speed + np.random.normal(0, 5.0)
                max_speed = inst_speed + np.random.uniform(5.0, 20.0)
                accel = np.random.normal(0, 5.0)
                net_disp = np.random.uniform(50.0, 200.0)
                path_len = net_disp / np.random.uniform(0.75, 0.95)
                linearity = net_disp / path_len
                bounding_rad = np.random.uniform(60.0, 180.0)
                dwell_time = np.random.uniform(0.5, 4.0)
                dist_restr = np.random.uniform(50.0, 300.0)
                in_restr = 0.0

            elif label == "loitering":
                inst_speed = np.random.uniform(2.0, 25.0)
                avg_speed = np.random.uniform(3.0, 20.0)
                max_speed = np.random.uniform(15.0, 45.0)
                accel = np.random.normal(0, 2.0)
                path_len = np.random.uniform(30.0, 120.0)
                linearity = np.random.uniform(0.05, 0.40)
                net_disp = path_len * linearity
                bounding_rad = np.random.uniform(10.0, 55.0)
                dwell_time = np.random.uniform(5.2, 30.0)
                dist_restr = np.random.uniform(40.0, 250.0)
                in_restr = 0.0

            elif label == "restricted_area_intrusion":
                inst_speed = np.random.uniform(10.0, 80.0)
                avg_speed = np.random.uniform(10.0, 75.0)
                max_speed = np.random.uniform(20.0, 100.0)
                accel = np.random.normal(0, 8.0)
                net_disp = np.random.uniform(20.0, 150.0)
                path_len = net_disp / np.random.uniform(0.6, 0.95)
                linearity = net_disp / path_len
                bounding_rad = np.random.uniform(20.0, 120.0)
                dwell_time = np.random.uniform(0.6, 15.0)
                dist_restr = 0.0
                in_restr = 1.0

            elif label == "running":
                inst_speed = np.random.uniform(145.0, 320.0)
                avg_speed = inst_speed + np.random.normal(0, 15.0)
                max_speed = inst_speed + np.random.uniform(10.0, 50.0)
                accel = np.random.uniform(10.0, 60.0)
                net_disp = np.random.uniform(150.0, 450.0)
                path_len = net_disp / np.random.uniform(0.85, 0.99)
                linearity = net_disp / path_len
                bounding_rad = np.random.uniform(120.0, 350.0)
                dwell_time = np.random.uniform(0.5, 4.0)
                dist_restr = np.random.uniform(30.0, 300.0)
                in_restr = 0.0

            elif label == "crowd_formation":
                inst_speed = np.random.uniform(15.0, 60.0)
                avg_speed = np.random.uniform(15.0, 55.0)
                max_speed = np.random.uniform(25.0, 90.0)
                accel = np.random.normal(0, 6.0)
                net_disp = np.random.uniform(20.0, 100.0)
                path_len = net_disp / np.random.uniform(0.40, 0.80)
                linearity = net_disp / path_len
                bounding_rad = np.random.uniform(30.0, 90.0)
                dwell_time = np.random.uniform(2.0, 10.0)
                dist_restr = np.random.uniform(20.0, 200.0)
                in_restr = 0.0

            vec = [
                inst_speed,
                avg_speed,
                max_speed,
                accel,
                net_disp,
                path_len,
                linearity,
                bounding_rad,
                dwell_time,
                dist_restr,
                in_restr
            ]
            X_list.append(vec)
            y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    return X, y, FEATURE_KEYS
