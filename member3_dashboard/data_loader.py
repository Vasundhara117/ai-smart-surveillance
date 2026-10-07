"""
Data Loader Module for Member 3 Surveillance Dashboard.
Handles ingestion, validation, normalization, and streaming simulation of activity_events.json.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from config import DEFAULT_SAMPLE_DATA_PATH, DEFAULT_LIVE_DATA_PATH


def load_activity_events(file_path: str = DEFAULT_SAMPLE_DATA_PATH) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Optional[str]]:
    """
    Loads and normalizes activity events from a JSON file.
    Supports both JSON array format and JSON lines (NDJSON) format.
    
    Returns:
        (events_list, df, error_message)
    """
    if not os.path.exists(file_path):
        return [], pd.DataFrame(), f"File not found: '{file_path}'"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if not content:
            return [], pd.DataFrame(), f"File is empty: '{file_path}'"

        events = []
        # Try parsing standard JSON array first
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                events = parsed
            elif isinstance(parsed, dict):
                events = [parsed]
        except json.JSONDecodeError:
            # Fallback to NDJSON (line by line)
            lines = content.splitlines()
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not events:
            return [], pd.DataFrame(), "No valid JSON activity events found in file."

        # Normalize and validate event fields
        normalized_events = []
        for i, ev in enumerate(events):
            norm_ev = {
                "event_id": ev.get("event_id", f"EVT-{i+1:04d}"),
                "timestamp": float(ev.get("timestamp", 0.0)),
                "person_id": int(ev.get("person_id", -1)),
                "activity": str(ev.get("activity", "unknown")).lower(),
                "confidence": round(float(ev.get("confidence", 0.0)), 3),
                "suspicious": bool(ev.get("suspicious", False)),
                "severity": str(ev.get("severity", "low")).lower()
            }
            # Auto-assign severity if missing but suspicious
            if norm_ev["suspicious"] and norm_ev["severity"] not in ["medium", "high", "critical"]:
                norm_ev["severity"] = "medium"
                
            normalized_events.append(norm_ev)

        # Create DataFrame
        df = pd.DataFrame(normalized_events)
        
        # Add human readable timestamp column (formatted as mm:ss)
        df["time_formatted"] = df["timestamp"].apply(format_timestamp)
        
        return normalized_events, df, None

    except Exception as e:
        return [], pd.DataFrame(), f"Error loading event data: {str(e)}"


def format_timestamp(ts: float) -> str:
    """Formats numeric timestamp into MM:SS or HH:MM:SS format."""
    minutes = int(ts // 60)
    seconds = int(ts % 60)
    tenths = int((ts - int(ts)) * 10)
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{tenths}"
    return f"{minutes:02d}:{seconds:02d}.{tenths}"


def filter_events(
    df: pd.DataFrame,
    suspicious_only: bool = False,
    severities: Optional[List[str]] = None,
    activities: Optional[List[str]] = None,
    min_confidence: float = 0.0,
    search_person_id: Optional[int] = None
) -> pd.DataFrame:
    """Applies multiple filtering options to the event DataFrame."""
    if df.empty:
        return df

    filtered = df.copy()

    if suspicious_only:
        filtered = filtered[filtered["suspicious"] == True]

    if severities:
        filtered = filtered[filtered["severity"].isin(severities)]

    if activities:
        filtered = filtered[filtered["activity"].isin(activities)]

    if min_confidence > 0.0:
        filtered = filtered[filtered["confidence"] >= min_confidence]

    if search_person_id is not None:
        filtered = filtered[filtered["person_id"] == search_person_id]

    return filtered
