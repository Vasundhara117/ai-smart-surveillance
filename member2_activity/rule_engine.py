import math
from typing import Dict, List, Any, Tuple

from member2_activity.config import (
    ALLOWED_ACTIVITIES,
    THRESHOLDS,
    RESTRICTED_ZONES,
)
from member2_activity.feature_extractor import (
    TrajectoryBuffer,
    point_in_polygon,
)


class RuleEngine:
    """Rule-based suspicious activity detection with alert cooldowns."""

    def __init__(self, buffer: TrajectoryBuffer):
        self.buffer = buffer

        # Tracks the last emitted alert for each activity identity.
        self.last_alert_time: Dict[Tuple[str, str], float] = {}

        # Default cooldown: 5 seconds.
        self.cooldown_sec = max(
            0.0,
            float(THRESHOLDS.get("alert_cooldown_sec", 5.0)),
        )

    def _should_emit(
        self,
        activity: str,
        identity: str,
        timestamp: float,
    ) -> bool:
        """Return False when the same activity is still in cooldown."""
        timestamp = float(timestamp)

        if not math.isfinite(timestamp):
            return False

        key = (activity, identity)
        previous = self.last_alert_time.get(key)

        if previous is not None:
            elapsed = timestamp - previous

            # Suppress duplicates during the cooldown.
            if 0 <= elapsed < self.cooldown_sec:
                return False

        # Allow a new alert after cooldown expiry.
        self.last_alert_time[key] = timestamp
        return True

    @staticmethod
    def _safe_confidence(value: float) -> float:
        """Keep confidence finite and within [0, 1]."""
        value = float(value)

        if not math.isfinite(value):
            return 0.0

        return round(max(0.0, min(1.0, value)), 2)

    def _create_event(
        self,
        person_id: int,
        activity: str,
        confidence: float,
        severity: str,
        timestamp: float,
    ) -> Dict[str, Any]:
        """Create an event compatible with the existing output schema."""
        return {
            "timestamp": float(timestamp),
            "person_id": int(person_id),
            "activity": str(activity),
            "confidence": self._safe_confidence(confidence),
            "suspicious": True,
            "severity": str(severity),
        }

    def evaluate_person_rules(
        self,
        person_id: int,
        features: Dict[str, float],
        det_confidence: float,
        timestamp: float,
    ) -> List[Dict[str, Any]]:
        """Evaluate restricted intrusion, running and loitering rules."""
        events: List[Dict[str, Any]] = []

        if not features:
            return events

        timestamp = float(timestamp)

        if not math.isfinite(timestamp):
            return events

        # Rule 1: Restricted-area intrusion.
        history = self.buffer.get_track_history(person_id)

        if history:
            curr_centroid = history[-1]["centroid"]

            for zone in RESTRICTED_ZONES:
                zone_name = zone["name"]
                polygon = zone["polygon"]

                if not point_in_polygon(curr_centroid, polygon):
                    continue

                # Safely handle missing zone-entry history.
                zone_entry_times = getattr(
                    self.buffer, "zone_entry_time", {}
                )
                person_entries = zone_entry_times.get(person_id, {})
                entry_time = person_entries.get(zone_name, timestamp)

                dwell_in_zone = max(0.0, timestamp - entry_time)

                required_dwell = zone.get(
                    "min_dwell_sec",
                    THRESHOLDS["intrusion_dwell_time_sec"],
                )

                if dwell_in_zone < required_dwell:
                    continue

                activity = str(ALLOWED_ACTIVITIES["INTRUSION"])
                identity = f"{person_id}:zone:{zone_name}"

                if not self._should_emit(activity, identity, timestamp):
                    continue

                confidence = (
                    float(det_confidence) * 0.95
                    + min(dwell_in_zone, 5.0) * 0.01
                )

                events.append(
                    self._create_event(
                        person_id=person_id,
                        activity=activity,
                        confidence=min(0.98, confidence),
                        severity=zone.get("severity", "high"),
                        timestamp=timestamp,
                    )
                )

        # Rule 2: Running / high-speed movement.
        speed = float(features.get("max_speed", 0.0))
        speed_threshold = THRESHOLDS["running_speed_thresh_px_per_sec"]

        if math.isfinite(speed) and speed >= speed_threshold:
            activity = str(ALLOWED_ACTIVITIES["RUNNING"])
            identity = str(person_id)

            if self._should_emit(activity, identity, timestamp):
                severity = "high" if speed > 220.0 else "medium"

                confidence = (
                    0.75 + (speed - speed_threshold) / 250.0
                )

                events.append(
                    self._create_event(
                        person_id=person_id,
                        activity=activity,
                        confidence=min(0.99, confidence),
                        severity=severity,
                        timestamp=timestamp,
                    )
                )

        # Rule 3: Loitering.
        dwell_time = float(features.get("dwell_time", 0.0))
        bounding_radius = float(
            features.get("bounding_radius", float("inf"))
        )

        dwell_threshold = THRESHOLDS["loitering_dwell_time_sec"]
        radius_threshold = THRESHOLDS["loitering_max_radius_px"]

        if (
            math.isfinite(dwell_time)
            and math.isfinite(bounding_radius)
            and dwell_time >= dwell_threshold
            and bounding_radius <= radius_threshold
        ):
            activity = str(ALLOWED_ACTIVITIES["LOITERING"])
            identity = str(person_id)

            if self._should_emit(activity, identity, timestamp):
                severity = "high" if dwell_time > 10.0 else "medium"

                confidence = (
                    0.70
                    + min(dwell_time - dwell_threshold, 10.0) * 0.025
                )

                events.append(
                    self._create_event(
                        person_id=person_id,
                        activity=activity,
                        confidence=min(0.99, confidence),
                        severity=severity,
                        timestamp=timestamp,
                    )
                )

        return events

    def evaluate_crowd_rules(
        self,
        crowd_features: Dict[str, Any],
        timestamp: float,
    ) -> List[Dict[str, Any]]:
        """Evaluate crowd rules with duplicate-alert suppression."""
        events: List[Dict[str, Any]] = []

        timestamp = float(timestamp)

        if not math.isfinite(timestamp):
            return events

        max_cluster_size = crowd_features.get("max_cluster_size", 0)
        clusters = crowd_features.get("crowd_clusters", [])
        min_crowd = THRESHOLDS["crowd_min_people"]

        if max_cluster_size < min_crowd:
            return events

        for cluster in clusters:
            if len(cluster) < min_crowd:
                continue

            # Treat the same set of tracked people as one cluster identity.
            cluster_identity = ",".join(
                str(pid) for pid in sorted(cluster)
            )

            activity = str(ALLOWED_ACTIVITIES["CROWD"])
            identity = f"cluster:{cluster_identity}"

            if not self._should_emit(activity, identity, timestamp):
                continue

            primary_person_id = cluster[0]
            cluster_size = len(cluster)

            severity = "high" if cluster_size >= 5 else "medium"

            confidence = min(
                0.98,
                0.75 + (cluster_size - min_crowd) * 0.08,
            )

            events.append(
                self._create_event(
                    person_id=primary_person_id,
                    activity=activity,
                    confidence=confidence,
                    severity=severity,
                    timestamp=timestamp,
                )
            )

        return events
