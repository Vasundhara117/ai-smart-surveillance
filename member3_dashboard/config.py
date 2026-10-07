"""
Configuration settings for Member 3 Surveillance Dashboard and Alert Framework.
Strict production formatting rules: No emoji icons, no em dashes, no purple gradients.
"""

# App Page Configuration
PAGE_TITLE = "Smart Surveillance Alert and Monitoring Dashboard"
PAGE_ICON = "favicon.ico"
LAYOUT = "wide"

# SDG Goals Mapping
SDG_INFO = {
    "SDG 9": {
        "title": "Industry, Innovation and Infrastructure",
        "description": "Fosters resilient infrastructure through computer vision and automated alert systems."
    },
    "SDG 11": {
        "title": "Sustainable Cities and Communities",
        "description": "Enhances urban safety, smart city monitoring, and public security in community spaces."
    },
    "SDG 16": {
        "title": "Peace, Justice and Strong Institutions",
        "description": "Supports crime prevention, rule of law, and reliable incident documentation with auditable logs."
    }
}

# Severity Color Mappings (Clean high-contrast colors)
SEVERITY_COLORS = {
    "low": "#10b981",        # Emerald Green
    "medium": "#f59e0b",     # Amber
    "high": "#f97316",       # Dark Orange
    "critical": "#ef4444"    # Red
}

# Default Files
DEFAULT_SAMPLE_DATA_PATH = "sample_activity_events.json"
DEFAULT_LIVE_DATA_PATH = "activity_events.json"

# Alert Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.75
REFRESH_INTERVAL_SEC = 2
