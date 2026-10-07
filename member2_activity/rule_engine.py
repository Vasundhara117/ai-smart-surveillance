from typing import Dict, List, Any, Optional
from member2_activity.config import (
    ALLOWED_ACTIVITIES,
    ALLOWED_SEVERITIES,
    THRESHOLDS,
    RESTRICTED_ZONES
)
from member2_activity.feature_extractor import TrajectoryBuffer, point_in_polygon


class RuleEngine:
    """Rule-Based Logic Engine for Suspicious Activity Detection."""

    def __init__(self, buffer: TrajectoryBuffer):
        self.buffer = buffer

    def evaluate_person_rules(
        self,
        person_id: int,
        features: Dict[str, float],
        det_confidence: float,
        timestamp: float
    ) -> List[Dict[str, Any]]:
        """Evaluates per-person activity rules."""
        events = []
        if not features:
            return events

        # Rule 1: Restricted Area Intrusion
        # High priority check
        history = self.buffer.get_track_history(person_id)
        if history:
            curr_centroid = history[-1]["centroid"]
            for zone in RESTRICTED_ZONES:
                z_name = zone["name"]
                poly = zone["polygon"]
                if point_in_polygon(curr_centroid, poly):
                    entry_time = self.buffer.zone_entry_time[person_id].get(z_name, timestamp)
                    dwell_in_zone = timestamp - entry_time
                    if dwell_in_zone >= zone.get("min_dwell_sec", THRESHOLDS["intrusion_dwell_time_sec"]):
                        conf = min(0.98, round(float(det_confidence * 0.95 + min(dwell_in_zone, 5.0) * 0.01), 2))
                        events.append({
                            "timestamp": float(timestamp),
                            "person_id": int(person_id),
                            "activity": str(ALLOWED_ACTIVITIES["INTRUSION"]),
                            "confidence": float(conf),
                            "suspicious": True,
                            "severity": str(zone.get("severity", "high"))
                        })

        # Rule 2: Running / High Speed Movement
        speed = features["max_speed"]
        speed_thresh = THRESHOLDS["running_speed_thresh_px_per_sec"]
        if speed >= speed_thresh:
            severity = "high" if speed > 220.0 else "medium"
            conf = min(0.99, round(0.75 + (speed - speed_thresh) / 250.0, 2))
            events.append({
                "timestamp": float(timestamp),
                "person_id": int(person_id),
                "activity": str(ALLOWED_ACTIVITIES["RUNNING"]),
                "confidence": float(conf),
                "suspicious": True,
                "severity": str(severity)
            })

        # Rule 3: Loitering (Stationary or low-radius dwell)
        dwell_time = features["dwell_time"]
        bounding_radius = features["bounding_radius"]
        loiter_time_thresh = THRESHOLDS["loitering_dwell_time_sec"]
        loiter_radius_thresh = THRESHOLDS["loitering_max_radius_px"]

        if dwell_time >= loiter_time_thresh and bounding_radius <= loiter_radius_thresh:
            severity = "high" if dwell_time > 10.0 else "medium"
            conf = min(0.99, round(0.70 + min(dwell_time - loiter_time_thresh, 10.0) * 0.025, 2))
            events.append({
                "timestamp": float(timestamp),
                "person_id": int(person_id),
                "activity": str(ALLOWED_ACTIVITIES["LOITERING"]),
                "confidence": float(conf),
                "suspicious": True,
                "severity": str(severity)
            })

        return events

    def evaluate_crowd_rules(
        self,
        crowd_features: Dict[str, Any],
        timestamp: float
    ) -> List[Dict[str, Any]]:
        """Evaluates group/crowd activity rules."""
        events = []
        max_cluster_size = crowd_features.get("max_cluster_size", 0)
        clusters = crowd_features.get("crowd_clusters", [])
        min_crowd = THRESHOLDS["crowd_min_people"]

        if max_cluster_size >= min_crowd:
            for cluster in clusters:
                if len(cluster) >= min_crowd:
                    # Create an event for the cluster head or primary person in the cluster
                    primary_person_id = cluster[0]
                    severity = "high" if len(cluster) >= 5 else "medium"
                    conf = min(0.98, round(0.75 + (len(cluster) - min_crowd) * 0.08, 2))
                    events.append({
                        "timestamp": float(timestamp),
                        "person_id": int(primary_person_id),
                        "activity": str(ALLOWED_ACTIVITIES["CROWD"]),
                        "confidence": float(conf),
                        "suspicious": True,
                        "severity": str(severity)
                    })

        return events
