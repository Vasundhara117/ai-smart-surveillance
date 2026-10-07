import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Optional
from member2_activity.config import (
    BUFFER_MAX_FRAMES,
    DEFAULT_FPS,
    RESTRICTED_ZONES,
    THRESHOLDS
)


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """Ray-casting algorithm to test if a 2D point (x, y) is inside a polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def min_dist_to_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> float:
    """Calculates minimum distance from a point to polygon boundary or 0 if inside."""
    if point_in_polygon(point, polygon):
        return 0.0

    px, py = point
    min_dist = float('inf')
    n = len(polygon)

    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]

        # Line segment vector AB and point vector AP
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab_sq = abx**2 + aby**2

        if ab_sq == 0:
            dist = np.hypot(px - ax, py - ay)
        else:
            t = max(0, min(1, (apx * abx + apy * aby) / ab_sq))
            proj_x = ax + t * abx
            proj_y = ay + t * aby
            dist = np.hypot(px - proj_x, py - proj_y)

        if dist < min_dist:
            min_dist = dist

    return min_dist


class TrajectoryBuffer:
    """Stores rolling history of object detection bounding boxes and centroids."""

    def __init__(self, max_frames: int = BUFFER_MAX_FRAMES):
        self.max_frames = max_frames
        # person_id -> deque of dict records: {'frame', 'timestamp', 'bbox', 'centroid'}
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_frames))
        # person_id -> timestamp when first detected in current continuous track
        self.track_start_time: Dict[int, float] = {}
        # person_id -> dict of zone_name -> timestamp entered zone
        self.zone_entry_time: Dict[int, Dict[str, float]] = defaultdict(dict)

    def update(self, frame_num: int, timestamp: float, objects: List[Dict[str, Any]]):
        """Updates internal buffers with detections from current frame."""
        current_ids = set()

        for obj in objects:
            if obj.get("class") != "person":
                continue

            person_id = int(obj["id"])
            current_ids.add(person_id)

            bbox = obj["bbox"]  # [x1, y1, x2, y2]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            record = {
                "frame": frame_num,
                "timestamp": timestamp,
                "bbox": bbox,
                "centroid": (cx, cy),
            }

            if person_id not in self.track_start_time:
                self.track_start_time[person_id] = timestamp

            self.history[person_id].append(record)

            # Check zone entry/dwell times
            for zone in RESTRICTED_ZONES:
                z_name = zone["name"]
                polygon = zone["polygon"]
                is_inside = point_in_polygon((cx, cy), polygon)

                if is_inside:
                    if z_name not in self.zone_entry_time[person_id]:
                        self.zone_entry_time[person_id][z_name] = timestamp
                else:
                    if z_name in self.zone_entry_time[person_id]:
                        del self.zone_entry_time[person_id][z_name]

        # Cleanup stale tracks no longer present (optional purging after timeout)
        stale_ids = [pid for pid in self.history if pid not in current_ids and (timestamp - self.history[pid][-1]["timestamp"]) > 10.0]
        for pid in stale_ids:
            del self.history[pid]
            if pid in self.track_start_time:
                del self.track_start_time[pid]
            if pid in self.zone_entry_time:
                del self.zone_entry_time[pid]

    def get_track_history(self, person_id: int) -> List[Dict[str, Any]]:
        return list(self.history.get(person_id, []))


class FeatureExtractor:
    """Extracts spatial, temporal, dynamic, and social/crowd feature representations."""

    def __init__(self, trajectory_buffer: TrajectoryBuffer):
        self.buffer = trajectory_buffer

    def extract_person_features(self, person_id: int, current_timestamp: float) -> Optional[Dict[str, float]]:
        """Extracts temporal feature vector for a specific person."""
        history = self.buffer.get_track_history(person_id)
        if not history:
            return None

        centroids = [h["centroid"] for h in history]
        timestamps = [h["timestamp"] for h in history]

        total_dwell_time = current_timestamp - self.buffer.track_start_time.get(person_id, current_timestamp)

        if len(centroids) < 2:
            cx, cy = centroids[-1]
            dist_to_restr = min([min_dist_to_polygon((cx, cy), z["polygon"]) for z in RESTRICTED_ZONES]) if RESTRICTED_ZONES else 999.0
            return {
                "instantaneous_speed": 0.0,
                "avg_speed": 0.0,
                "max_speed": 0.0,
                "acceleration": 0.0,
                "net_displacement": 0.0,
                "path_length": 0.0,
                "linearity": 1.0,
                "bounding_radius": 0.0,
                "dwell_time": total_dwell_time,
                "dist_to_restricted": dist_to_restr,
                "in_restricted_zone": float(dist_to_restr == 0.0)
            }

        # Calculate incremental displacements and speeds
        dt = np.diff(timestamps)
        # Prevent division by zero
        dt = np.where(dt <= 0, 1.0 / DEFAULT_FPS, dt)

        coords = np.array(centroids)
        step_diffs = np.diff(coords, axis=0)
        step_dists = np.hypot(step_diffs[:, 0], step_diffs[:, 1])

        inst_speeds = step_dists / dt
        instantaneous_speed = float(inst_speeds[-1])
        max_speed = float(np.max(inst_speeds))

        total_path_length = float(np.sum(step_dists))
        total_time_span = float(timestamps[-1] - timestamps[0])
        avg_speed = total_path_length / total_time_span if total_time_span > 0 else 0.0

        # Acceleration calculation over recent steps
        if len(inst_speeds) >= 2:
            acceleration = float((inst_speeds[-1] - inst_speeds[0]) / (timestamps[-1] - timestamps[1]))
        else:
            acceleration = 0.0

        # Net displacement and Linearity Ratio
        net_disp = float(np.hypot(centroids[-1][0] - centroids[0][0], centroids[-1][1] - centroids[0][1]))
        linearity = net_disp / total_path_length if total_path_length > 0 else 1.0

        # Bounding movement radius from centroid mean
        mean_center = np.mean(coords, axis=0)
        bounding_radius = float(np.max(np.hypot(coords[:, 0] - mean_center[0], coords[:, 1] - mean_center[1])))

        # Distance to nearest restricted zone
        curr_cx, curr_cy = centroids[-1]
        dist_to_restr = min([min_dist_to_polygon((curr_cx, curr_cy), z["polygon"]) for z in RESTRICTED_ZONES]) if RESTRICTED_ZONES else 999.0

        return {
            "instantaneous_speed": instantaneous_speed,
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "acceleration": acceleration,
            "net_displacement": net_disp,
            "path_length": total_path_length,
            "linearity": linearity,
            "bounding_radius": bounding_radius,
            "dwell_time": total_dwell_time,
            "dist_to_restricted": dist_to_restr,
            "in_restricted_zone": float(dist_to_restr == 0.0)
        }

    def extract_crowd_features(self, active_objects: List[Dict[str, Any]], current_timestamp: float) -> Dict[str, Any]:
        """Extracts spatial clustering and density features for crowd analysis."""
        people = [o for o in active_objects if o.get("class") == "person"]
        if not people:
            return {"active_person_count": 0, "max_cluster_size": 0, "crowd_clusters": []}

        coords = []
        pids = []
        for p in people:
            bbox = p["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            coords.append((cx, cy))
            pids.append(p["id"])

        coords_arr = np.array(coords)
        n = len(coords_arr)

        if n == 1:
            return {
                "active_person_count": 1,
                "max_cluster_size": 1,
                "crowd_clusters": [[pids[0]]],
                "pairwise_distances": np.zeros((1, 1))
            }

        # Pairwise distance matrix
        diff = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
        dist_matrix = np.hypot(diff[:, :, 0], diff[:, :, 1])

        # Cluster detection based on distance threshold
        radius = THRESHOLDS["crowd_distance_radius_px"]
        adj_matrix = dist_matrix <= radius

        # Graph connected components for cluster grouping
        visited = [False] * n
        clusters = []

        for i in range(n):
            if not visited[i]:
                cluster = []
                queue = [i]
                visited[i] = True
                while queue:
                    curr = queue.pop(0)
                    cluster.append(pids[curr])
                    for neighbor in range(n):
                        if adj_matrix[curr, neighbor] and not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                clusters.append(cluster)

        max_cluster_size = max([len(c) for c in clusters]) if clusters else 0

        return {
            "active_person_count": n,
            "max_cluster_size": max_cluster_size,
            "crowd_clusters": clusters,
            "pairwise_distances": dist_matrix
        }
