# Member 2: Suspicious Activity Detection Framework

An AI-powered temporal and spatial analysis module for smart video surveillance, developed as part of a 3-member AIML course project titled **"AI-Based Smart Surveillance and Suspicious Activity Detection Framework"**.

---

## 📌 SDG Alignments

- **SDG 9: Industry, Innovation & Infrastructure** – Smart infrastructure monitoring using scalable AI pipelines.
- **SDG 11: Sustainable Cities & Communities** – Enhancing public safety and automated threat detection in urban areas.
- **SDG 16: Peace, Justice & Strong Institutions** – Supporting institutional security through real-time situational awareness.

---

## 🏗️ Architecture & Overall Pipeline

```
Video / CCTV Feed
       ↓
Member 1: Detection + Tracking (YOLO + DeepSORT / ByteTrack)
       ↓
  detection.json
       ↓
MEMBER 2 (THIS MODULE): Feature Extraction & Suspicious Activity Classification
       ↓
  activity_events.json
       ↓
Member 3: Real-Time Alerts & Monitoring Dashboard
```

---

## 📥 Input / Output Contracts

### Input Interface (`detection.json`)
Consumes bounding box tracking records produced by Member 1:

```json
{
  "frame": 120,
  "timestamp": 4.0,
  "objects": [
    {
      "id": 7,
      "class": "person",
      "confidence": 0.91,
      "bbox": [120, 80, 250, 400]
    }
  ]
}
```

### Output Interface (`activity_events.json`)
Produces structured activity event logs required by Member 3:

```json
{
  "timestamp": 12.4,
  "person_id": 7,
  "activity": "loitering",
  "confidence": 0.89,
  "suspicious": true,
  "severity": "high"
}
```

- **Allowed Activity Names**: `"loitering"`, `"restricted_area_intrusion"`, `"running"`, `"crowd_formation"`
- **Allowed Severities**: `"low"`, `"medium"`, `"high"`

---

## 🧠 AI/ML & Mathematical Formulations

This module adopts a **Hybrid Architecture** combining explicit mathematical rule engines with a machine learning classification model (`RandomForestClassifier`).

### 1. Feature Representation (11-Dimensional Feature Vector)
For each tracked individual over a rolling window \(W\):
- `instantaneous_speed`: \(v_t = \frac{\Delta d}{\Delta t}\)
- `avg_speed`: \(\bar{v} = \frac{\sum \Delta d}{\sum \Delta t}\)
- `max_speed`: \(\max(v_t)\)
- `acceleration`: \(a_t = \frac{\Delta v}{\Delta t}\)
- `net_displacement`: \(D_{\text{net}} = \|\mathbf{p}_{\text{end}} - \mathbf{p}_{\text{start}}\|_2\)
- `path_length`: \(L = \sum \|\mathbf{p}_i - \mathbf{p}_{i-1}\|_2\)
- `linearity`: \(\eta = \frac{D_{\text{net}}}{L}\) (\(0 \le \eta \le 1\))
- `bounding_radius`: \(R_{\text{bound}} = \max_i \|\mathbf{p}_i - \bar{\mathbf{p}}\|_2\)
- `dwell_time`: \(T_{\text{dwell}} = t_{\text{curr}} - t_{\text{entry}}\)
- `dist_to_restricted`: Minimum Euclidean distance to predefined restricted polygon.
- `in_restricted_zone`: Binary indicator (\(1.0\) if inside, \(0.0\) otherwise).

---

### 2. Rule Engine Formulations

#### Activity 1: Loitering
- **Mathematical Condition**:
  \[
  T_{\text{dwell}} \ge T_{\text{loiter\_thresh}} \quad \text{AND} \quad R_{\text{bound}} \le R_{\text{loiter\_max}}
  \]
  *(Where \(T_{\text{loiter\_thresh}} = 5.0\text{ sec}\), \(R_{\text{loiter\_max}} = 60.0\text{ px}\))*
- **Severity**: `"medium"` if \(5.0\text{s} \le T_{\text{dwell}} \le 10.0\text{s}\), `"high"` if \(T_{\text{dwell}} > 10.0\text{s}\).

#### Activity 2: Restricted Area Intrusion
- **Mathematical Condition**: Ray-casting point-in-polygon algorithm:
  \[
  (c_x, c_y) \in P_{\text{restricted}} \quad \text{AND} \quad T_{\text{in\_zone}} \ge T_{\text{intrusion\_thresh}}
  \]
- **Severity**: `"high"`.

#### Activity 3: Running / High-Speed Movement
- **Mathematical Condition**:
  \[
  v_{\max} \ge V_{\text{run\_thresh}} \quad (V_{\text{run\_thresh}} = 140.0\text{ px/sec})
  \]
- **Severity**: `"medium"` if \(140 \le v \le 220\), `"high"` if \(v > 220\).

#### Activity 4: Sudden Crowd Formation
- **Mathematical Condition**: Pairwise Euclidean distance matrix \(M_{ij} = \|\mathbf{p}_i - \mathbf{p}_j\|_2\).
  \[
  N_{\text{cluster}} = |\{ j \mid M_{ij} \le R_{\text{crowd}} \}| \ge N_{\text{min}} \quad (N_{\text{min}} = 3, R_{\text{crowd}} = 150\text{ px})
  \]
- **Severity**: `"medium"` if \(N = 3, 4\), `"high"` if \(N \ge 5\).

---

### 3. Machine Learning Classification & Metrics

- **Model**: `RandomForestClassifier` (\(N_{\text{estimators}} = 120\), \(\text{max\_depth} = 12\)).
- **Preprocessing**: `StandardScaler` feature normalization.
- **Train/Test Split**: 75% Train, 25% Test (Stratified across classes).

#### Evaluation Metrics Summary (`ml_evaluation_report.txt`):
- **Accuracy**: 100.00%
- **Multiclass ROC-AUC (OvR)**: 1.0000
- **Precision / Recall / F1-Score**: 1.00 across all 5 classes (`normal`, `loitering`, `restricted_area_intrusion`, `running`, `crowd_formation`).
- **Key Feature Importances**: Linearity (0.2221), Distance to Restricted Zone (0.1459), Average Speed (0.1106), In Restricted Zone (0.1100).

---

## 📁 Directory Structure

```
member2_activity/
├── activity_detector.py       # Main pipeline orchestrator (processes frame detections -> events)
├── feature_extractor.py       # Trajectory buffer, spatial/temporal feature extraction & crowd metrics
├── rule_engine.py             # Math/logic-based detection rules (loitering, intrusion, running, crowd)
├── classifier.py              # ML classifier inference wrapper
├── dataset_generator.py       # Synthesizes feature dataset for model training
├── train_model.py             # Trains Random Forest classifier & writes ml_evaluation_report.txt
├── config.py                  # Thresholds, zone polygons, ML settings
├── model.joblib               # Exported trained ML model weights
├── scaler.joblib              # Exported feature scaler
├── sample_detection_data.json # Sample input detection stream (Member 1 format)
├── sample_activity_events.json# Sample output activity events (Member 3 format)
├── test_activity_detector.py  # Unit test suite covering all 4 activities & ML model
├── ml_evaluation_report.txt   # Detailed evaluation report
├── requirements.txt           # Python dependencies
└── README.md                  # Detailed documentation
```

---

## ⚙️ Quickstart & Execution Guide

### 1. Setup Virtual Environment & Dependencies
```bash
python3 -m venv member2_activity/.venv
source member2_activity/.venv/bin/activate
pip install -r member2_activity/requirements.txt
```

### 2. Train Machine Learning Model & Evaluate
```bash
PYTHONPATH=. python3 member2_activity/train_model.py
```

### 3. Run Unit Test Suite
```bash
PYTHONPATH=. python3 -m unittest member2_activity/test_activity_detector.py
```

### 4. Process Sample Detections & Generate Events
```bash
PYTHONPATH=. python3 -c "from member2_activity.activity_detector import ActivityDetector; detector = ActivityDetector(); detector.process_json_file('member2_activity/sample_detection_data.json', 'member2_activity/activity_events.json')"
```
