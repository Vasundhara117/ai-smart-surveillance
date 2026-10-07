# Member 3: Alerts, Surveillance Dashboard, and Incident Reports

## Project Overview
This repository contains **Member 3's Module** for the AIML course project:  
**"AI-Based Smart Surveillance and Suspicious Activity Detection Framework"**.

### United Nations Sustainable Development Goals (SDGs)
- **SDG 9: Industry, Innovation and Infrastructure** - AI-assisted automated monitoring and smart security infrastructure.
- **SDG 11: Sustainable Cities and Communities** - Enhancing public safety and security in urban and community spaces.
- **SDG 16: Peace, Justice and Strong Institutions** - Supporting crime prevention, transparent surveillance logging, and evidence documentation.

---

## System Architecture and Module Boundaries

```text
Video / CCTV Feed
       │
       ▼
Member 1: Object Detection + Tracking
       │
       ▼ detection.json
Member 2: Suspicious Activity Detection
       │
       ▼ activity_events.json
Member 3: Alerts + Dashboard + Incident Reports  <-- [THIS MODULE]
```

### Member 3 Responsibilities
- Reading activity events from `activity_events.json`
- Real-time alert management and visual/audio notifications
- Command center surveillance dashboard built with Streamlit
- Event history log with multi-parameter search and filtering
- Visual analytics and statistics (incidents by activity, timeline, severity distribution)
- Incident report generation (CSV exports, PDF audit reports, HTML reports)
- Legal compliance documentation (Privacy Policy, Terms and Conditions)

> **Note:** Member 3 does **NOT** implement object detection, tracking, activity classification, or ML model training. It operates completely independently through the agreed JSON data interface.

---

## Input Data Interface (activity_events.json)

Member 3 consumes Member 2's activity events formatted as follows:

```json
[
  {
    "timestamp": 12.4,
    "person_id": 7,
    "activity": "loitering",
    "confidence": 0.89,
    "suspicious": true,
    "severity": "high"
  }
]
```

### Schema Parameters
| Field | Type | Description |
|---|---|---|
| `timestamp` | `float` | Video time in seconds (or epoch timestamp) |
| `person_id` | `int` | Unique tracking ID assigned to person |
| `activity` | `string` | Detected activity type (e.g. `loitering`, `fighting`, `unattended_bag`, `normal_walking`) |
| `confidence` | `float` | Model confidence score (0.0 to 1.0) |
| `suspicious` | `bool` | `true` if suspicious activity, else `false` |
| `severity` | `string` | Risk level: `"low"`, `"medium"`, `"high"`, or `"critical"` |

---

## File Structure

```text
member3_dashboard/
├── app.py                      # Main Streamlit dashboard application
├── alert_manager.py            # Alert processing, severity classification and notification triggers
├── report_generator.py         # CSV, PDF, and HTML report generator
├── data_loader.py              # Ingestion, validation, filtering and simulation helper
├── config.py                   # Configuration parameters, color schemes and SDG mappings
├── favicon.png                 # Verified site favicon icon
├── sample_activity_events.json # Realistic sample activity dataset for testing
├── requirements.txt            # Required Python packages
└── README.md                   # Setup and usage documentation
```

---

## Instructions to Run the Dashboard Independently

### Step 1: Open Terminal and Navigate to `member3_dashboard/`
```bash
cd member3_dashboard
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Dashboard
```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## Integration with Member 2

1. Member 2 outputs activity detection results to `activity_events.json` inside the `member3_dashboard/` folder (or enter custom file path in the sidebar).
2. In the sidebar under **Event Data Source**, select **Live Member 2 Output File**.
3. To test real-time video streaming simulation, click **Play Stream** in the sidebar.
