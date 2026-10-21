
import pandas as pd


def group_incidents(
    df: pd.DataFrame,
    gap_seconds: float = 2.0,
) -> pd.DataFrame:
    """Group repeated observations by person and activity."""

    columns = [
        "incident_id",
        "person_id",
        "activity",
        "start_timestamp",
        "end_timestamp",
        "duration_seconds",
        "severity",
        "confidence",
        "suspicious",
        "observation_count",
    ]

    if df.empty:
        return pd.DataFrame(columns=columns)

    required = {"timestamp", "person_id", "activity"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = df.copy()
    data["timestamp"] = pd.to_numeric(
        data["timestamp"], errors="coerce"
    )
    data = data.dropna(subset=["timestamp"])
    data = data.sort_values(
        ["person_id", "activity", "timestamp"]
    )

    incidents = []

    for (person_id, activity), group in data.groupby(
        ["person_id", "activity"], dropna=False, sort=False
    ):
        current = []

        for _, event in group.iterrows():
            if (
                current
                and event["timestamp"] - current[-1]["timestamp"]
                > gap_seconds
            ):
                incidents.append((person_id, activity, current))
                current = []

            current.append(event)

        if current:
            incidents.append((person_id, activity, current))

    results = []

    severity_rank = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }

    for index, (person_id, activity, events) in enumerate(
        incidents, start=1
    ):
        start = float(events[0]["timestamp"])
        end = float(events[-1]["timestamp"])

        severity = max(
            (
                str(e.get("severity", "low")).lower()
                for e in events
            ),
            key=lambda s: severity_rank.get(s, 0),
        )

        confidences = [
            float(e["confidence"])
            for e in events
            if pd.notna(e.get("confidence"))
        ]

        results.append({
            "incident_id": f"INC-{index:04d}",
            "person_id": person_id,
            "activity": activity,
            "start_timestamp": start,
            "end_timestamp": end,
            "duration_seconds": round(end - start, 2),
            "severity": severity,
            "confidence": (
                round(sum(confidences) / len(confidences), 3)
                if confidences else 0.0
            ),
            "suspicious": any(
                bool(e.get("suspicious", False))
                for e in events
            ),
            "observation_count": len(events),
        })

    return pd.DataFrame(results, columns=columns)
