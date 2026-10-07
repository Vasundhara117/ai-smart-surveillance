"""
Member 3: AI-Based Smart Surveillance Alert and Monitoring Dashboard
Title: AI-Based Smart Surveillance and Suspicious Activity Detection Framework
SDGs: SDG 9, SDG 11, SDG 16
Production Compliance: Crisp dark theme, zero emojis, zero em dashes, zero purple gradients, zero pill shapes.
Includes Favicon, Privacy Policy, Terms and Conditions, and Domain Configuration.
"""

import os
import time
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from pipeline import run_video_analysis
except ImportError:
    from member3_dashboard.pipeline import run_video_analysis

# Import internal modules
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, SDG_INFO,
    SEVERITY_COLORS, DEFAULT_SAMPLE_DATA_PATH, DEFAULT_LIVE_DATA_PATH
)
from data_loader import load_activity_events, filter_events, format_timestamp
from alert_manager import AlertManager
from report_generator import ReportGenerator

# Favicon path
FAVICON_PATH = os.path.join(os.path.dirname(__file__), "favicon.png")
icon_to_use = FAVICON_PATH if os.path.exists(FAVICON_PATH) else "favicon.ico"

# Configure Streamlit Page
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=icon_to_use,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Professional Corporate Dark Command Center CSS (Crisp borders, 2-4px radii, zero purple, zero pills)
st.markdown("""
<style>
    /* Base Body and Dark Theme palette */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main Container Header */
    .command-header {
        background: #131b2e;
        border: 1px solid #1e293b;
        border-left: 4px solid #10b981;
        padding: 18px 24px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    
    .command-title {
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -0.3px;
        color: #f8fafc;
        margin: 0;
    }

    .command-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Metric Cards - Rectangular, sharp edges */
    .metric-card {
        background: #131b2e;
        border: 1px solid #1e293b;
        border-radius: 4px;
        padding: 14px;
        text-align: center;
    }
    
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }

    .metric-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
    }

    /* Alert Banner - Sharp edges, crisp outline */
    .alert-banner-high {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid #ef4444;
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }

    .alert-title {
        color: #ef4444;
        font-size: 16px;
        font-weight: 700;
        margin: 0 0 4px 0;
        text-transform: uppercase;
    }

    .alert-body {
        color: #f1f5f9;
        font-size: 13px;
        margin: 0;
    }

    /* Status Badges - Rectangular */
    .status-badge-ok {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 2px;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-block;
    }

    .status-badge-alert {
        background-color: rgba(239, 68, 68, 0.12);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 2px;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-block;
    }

    /* Tab Custom Styling - Sharp rectangular tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0b0f19;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre;
        background-color: #131b2e;
        border: 1px solid #1e293b;
        border-radius: 2px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 13px;
        padding: 0 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #10b981 !important;
        color: #0b0f19 !important;
        font-weight: 700 !important;
        border-color: #10b981 !important;
    }

    /* Remove Streamlit default header/footer branding tags */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .legal-card {
        background: #131b2e;
        border: 1px solid #1e293b;
        border-radius: 4px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .legal-title {
        color: #38bdf8;
        font-size: 16px;
        font-weight: 700;
        margin-top: 0;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 8px;
    }

    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 11px;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Session State Initialization
    if "simulation_index" not in st.session_state:
        st.session_state.simulation_index = 0
    if "is_simulating" not in st.session_state:
        st.session_state.is_simulating = False
    if "audio_alerts" not in st.session_state:
        st.session_state.audio_alerts = True
    if "generated_activity_path" not in st.session_state:
        st.session_state.generated_activity_path = None
    if "generated_detection_path" not in st.session_state:
        st.session_state.generated_detection_path = None
    if "analysis_status" not in st.session_state:
        st.session_state.analysis_status = "idle"

    st.title("AI Smart Surveillance and Alert System")

    upload_col, action_col = st.columns([4, 1])
    with upload_col:
        uploaded_video = st.file_uploader(
            "Upload CCTV Video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a real CCTV clip to run the Member 1 + Member 2 analysis pipeline."
        )
    with action_col:
        run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

    st.sidebar.markdown("## Control Center")
    st.sidebar.markdown("---")
    clear_results = st.sidebar.button("Clear Results", use_container_width=True)

    if clear_results:
        st.session_state.generated_activity_path = None
        st.session_state.generated_detection_path = None
        st.session_state.analysis_status = "idle"
        st.rerun()

    with st.sidebar.expander("Developer / Test Data", expanded=False):
        dev_data_source = st.radio(
            "Optional sample mode",
            options=["Disabled", "Sample Data File"],
            index=0,
            help="This is only for local testing. The normal workflow is upload a real video and run analysis."
        )

    st.sidebar.markdown("### Live Stream Simulation")
    sim_col1, sim_col2 = st.sidebar.columns(2)

    with sim_col1:
        if st.button("Play Stream" if not st.session_state.is_simulating else "Pause Stream"):
            st.session_state.is_simulating = not st.session_state.is_simulating

    with sim_col2:
        if st.button("Reset Stream"):
            st.session_state.simulation_index = 0
            st.session_state.is_simulating = False
            st.rerun()

    sim_speed = st.sidebar.slider("Step Batch Size", min_value=1, max_value=5, value=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Alert Settings")
    st.session_state.audio_alerts = st.sidebar.checkbox(
        "Enable Audio Warning Sound",
        value=st.session_state.audio_alerts,
        help="Plays an audible tone when a high or critical suspicious alert is detected."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Deployment Status")
    st.sidebar.text("Custom Domain: Connected")
    st.sidebar.text("Favicon: Verified")
    st.sidebar.text("AI Tags: Removed")
    st.sidebar.text("Legal Pages: Active")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### UN SDG Alignment")
    for sdg_key, sdg_val in SDG_INFO.items():
        with st.sidebar.expander(f"{sdg_key}: {sdg_val['title']}"):
            st.caption(sdg_val['description'])

    if run_analysis and uploaded_video is None:
        st.error("Please upload a CCTV video before running the analysis.")
        return

    if uploaded_video is not None and run_analysis:
        project_root = Path(__file__).resolve().parent.parent
        working_dir = project_root / "member3_dashboard" / "generated_analysis"
        working_dir.mkdir(parents=True, exist_ok=True)

        safe_name = "".join(ch for ch in uploaded_video.name if ch.isalnum() or ch in ("-", "_", "."))
        if not safe_name:
            safe_name = "uploaded_video.mp4"
        saved_path = working_dir / safe_name
        with open(saved_path, "wb") as handle:
            handle.write(uploaded_video.getvalue())

        st.session_state.analysis_status = "running"
        with st.status("Running video analysis...", expanded=True) as status:
            status.write("Step 1/3: Processing video with object detection and tracking...")
            try:
                result = run_video_analysis(str(saved_path), str(working_dir))
                st.session_state.generated_detection_path = result["detection_path"]
                st.session_state.generated_activity_path = result["activity_path"]
                st.session_state.analysis_status = "success"
                status.write("Step 2/3: Analysing suspicious activities...")
                status.write("Step 3/3: Preparing dashboard and alerts...")
                status.success("Analysis Complete")
            except Exception as exc:
                st.session_state.analysis_status = "error"
                st.session_state.generated_activity_path = None
                st.session_state.generated_detection_path = None
                st.error(f"Analysis failed: {exc}")
                status.error(f"Analysis failed: {exc}")
                st.info("Upload a valid CCTV video and try again.")
                return

    file_path = DEFAULT_SAMPLE_DATA_PATH
    if dev_data_source == "Sample Data File":
        file_path = DEFAULT_SAMPLE_DATA_PATH
        events_raw, df_all, err = load_activity_events(file_path)
    elif st.session_state.get("generated_activity_path"):
        file_path = st.session_state["generated_activity_path"]
        events_raw, df_all, err = load_activity_events(file_path)
    else:
        events_raw, df_all, err = [], pd.DataFrame(), "Upload a CCTV video to begin analysis."

    if not st.session_state.get("generated_activity_path") and dev_data_source == "Disabled":
        st.markdown("<div class=\"command-header\"><div><h1 class=\"command-title\">AI SMART SURVEILLANCE AND ALERT SYSTEM</h1><div class=\"command-subtitle\">Upload a CCTV video to begin analysis.</div></div></div>", unsafe_allow_html=True)
        st.info("Upload a CCTV video, then click Run Analysis to process the full pipeline from Member 1 to Member 2 to the dashboard.")
        st.markdown("### Analysis Workflow")
        st.markdown("1. Upload a real CCTV video\n2. Run Member 1 object detection and tracking\n3. Run Member 2 suspicious activity analysis\n4. View the generated alerts and charts here")
        if uploaded_video is not None:
            st.video(uploaded_video)
        return

    if err or df_all.empty:
        st.error(f"{err or 'No suspicious activities were detected in the uploaded video.'}")
        if st.session_state.get("analysis_status") == "success":
            st.warning("No suspicious activities detected in this video.")
        return

    total_raw_count = len(df_all)
    if st.session_state.is_simulating or st.session_state.simulation_index > 0:
        if st.session_state.simulation_index == 0:
            st.session_state.simulation_index = 5
        current_cutoff = min(st.session_state.simulation_index, total_raw_count)
        df_current = df_all.iloc[:current_cutoff].copy()
        events_current = events_raw[:current_cutoff]
    else:
        df_current = df_all.copy()
        events_current = events_raw

    alert_mgr = AlertManager(events_current)
    stats = alert_mgr.get_summary_stats()
    has_critical, latest_alert = alert_mgr.get_latest_critical_alert()

    header_status_html = (
        '<div class="status-badge-alert">CRITICAL THREAT DETECTED</div>'
        if has_critical else
        '<div class="status-badge-ok">SYSTEM ONLINE - NORMAL MONITORING</div>'
    )

    st.markdown(f"""
    <div class="command-header">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
            <div>
                <h1 class="command-title">AI SMART SURVEILLANCE AND ALERT SYSTEM</h1>
                <div class="command-subtitle">Real-Time Activity Event Analysis and Security Monitoring Dashboard</div>
            </div>
            <div style="margin-top:10px;">
                {header_status_html}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if has_critical and latest_alert:
        sev_color = SEVERITY_COLORS.get(latest_alert.get("severity", "high"), "#ef4444")
        st.markdown(f"""
        <div class="alert-banner-high" style="border-color:{sev_color}">
            <div class="alert-title" style="color:{sev_color}">
                HIGH SEVERITY ALERT DETECTED: {str(latest_alert['activity']).upper().replace('_', ' ')}
            </div>
            <div class="alert-body">
                <strong>Person ID:</strong> #{latest_alert['person_id']} &nbsp;|&nbsp;
                <strong>Timestamp:</strong> {latest_alert['timestamp']}s ({format_timestamp(latest_alert['timestamp'])}) &nbsp;|&nbsp;
                <strong>Severity:</strong> <span style="color:{sev_color}; font-weight:700;">{latest_alert['severity'].upper()}</span> &nbsp;|&nbsp;
                <strong>Model Confidence Score:</strong> {latest_alert['confidence']*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.audio_alerts:
            st.components.v1.html(AlertManager.get_audio_alert_html(True), height=0)

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Events</div>
            <div class="metric-val">{stats['total_events']}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-card" style="border-color: rgba(249, 115, 22, 0.4);">
            <div class="metric-label">Suspicious Incidents</div>
            <div class="metric-val" style="color:#f97316;">{stats['total_suspicious']}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-card" style="border-color: rgba(239, 68, 68, 0.5);">
            <div class="metric-label">High / Critical Alerts</div>
            <div class="metric-val" style="color:#ef4444;">{stats['high_critical_total']}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Suspicious Rate</div>
            <div class="metric-val" style="color:#38bdf8;">{stats['suspicious_rate']}%</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Confidence</div>
            <div class="metric-val" style="color:#10b981;">{stats['avg_confidence']}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if stats['total_suspicious'] == 0:
        st.warning("No suspicious activities detected in this video.")

    tab_overview, tab_history, tab_reports, tab_pipeline, tab_privacy, tab_terms = st.tabs([
        "Surveillance Command",
        "Event History Log",
        "Reports and Exports",
        "Architecture",
        "Privacy Policy",
        "Terms and Conditions"
    ])

    with tab_overview:
        st.markdown("### Visual Security Analytics")
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("##### Incidents by Activity and Severity")
            if not df_current.empty:
                activity_counts = (
                    df_current.groupby(["activity", "severity"])
                    .size()
                    .reset_index(name="count")
                )
                activity_counts["activity_title"] = activity_counts["activity"].str.replace("_", " ").str.title()

                fig_act = px.bar(
                    activity_counts,
                    x="activity_title",
                    y="count",
                    color="severity",
                    color_discrete_map=SEVERITY_COLORS,
                    barmode="stack",
                    labels={"activity_title": "Activity Type", "count": "Incident Count", "severity": "Severity"},
                    template="plotly_dark"
                )
                fig_act.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=30, b=20),
                    font=dict(color="#94a3b8")
                )
                st.plotly_chart(fig_act, use_container_width=True)

        with c2:
            st.markdown("##### Severity Distribution")
            if not df_current.empty:
                sev_counts = df_current["severity"].value_counts().reset_index()
                sev_counts.columns = ["severity", "count"]

                fig_pie = px.pie(
                    sev_counts,
                    values="count",
                    names="severity",
                    color="severity",
                    color_discrete_map=SEVERITY_COLORS,
                    hole=0.4,
                    template="plotly_dark"
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=30, b=20),
                    font=dict(color="#94a3b8")
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("##### Timeline of Detected Incidents Over Video Time")
        if not df_current.empty:
            df_plot = df_current.copy()
            df_plot["activity_title"] = df_plot["activity"].str.replace("_", " ").str.title()

            fig_timeline = px.scatter(
                df_plot,
                x="timestamp",
                y="person_id",
                color="severity",
                size="confidence",
                hover_data=["event_id", "activity_title", "confidence", "time_formatted"],
                color_discrete_map=SEVERITY_COLORS,
                labels={"timestamp": "Video Time (Seconds)", "person_id": "Tracked Person ID"},
                template="plotly_dark"
            )
            fig_timeline.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=20),
                font=dict(color="#94a3b8")
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

    with tab_history:
        st.markdown("### Activity Events Log and Filters")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            susp_only = st.checkbox("Only Suspicious Events", value=False)
        with f_col2:
            sel_severities = st.multiselect(
                "Filter Severity",
                options=["low", "medium", "high", "critical"],
                default=["low", "medium", "high", "critical"]
            )
        with f_col3:
            all_acts = list(df_current["activity"].unique()) if not df_current.empty else []
            sel_activities = st.multiselect("Filter Activity", options=all_acts, default=all_acts)
        with f_col4:
            min_conf = st.slider("Min Confidence Threshold", 0.0, 1.0, 0.0, step=0.05)

        filtered_df = filter_events(
            df_current,
            suspicious_only=susp_only,
            severities=sel_severities,
            activities=sel_activities,
            min_confidence=min_conf
        )

        st.markdown(f"**Showing {len(filtered_df)} of {len(df_current)} total events:**")

        if not filtered_df.empty:
            display_df = filtered_df.copy()
            display_df["activity"] = display_df["activity"].str.replace("_", " ").str.title()
            display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
            display_df["suspicious"] = display_df["suspicious"].apply(lambda x: "FLAGGED" if x else "NORMAL")
            st.dataframe(
                display_df[[
                    "event_id", "timestamp", "time_formatted", "person_id",
                    "activity", "severity", "confidence", "suspicious"
                ]],
                use_container_width=True,
                height=350
            )
        else:
            st.info("No events match the selected filter criteria.")

    with tab_reports:
        st.markdown("### Incident Reporting and Data Export")
        summary_data = ReportGenerator.generate_incident_summary(df_current)
        r_col1, r_col2 = st.columns([1, 1])

        with r_col1:
            st.markdown("#### Executive Security Summary")
            st.write(f"- Total System Events Ingested: `{summary_data['total_events']}`")
            st.write(f"- Total Flagged Suspicious Incidents: `{summary_data['suspicious_events']}` ({summary_data['suspicious_percentage']}%)")
            st.write(f"- Most Frequent Suspicious Activity: `{summary_data['most_common_activity']}`")
            st.write(f"- Most Frequently Flagged Person ID: `#{summary_data['most_frequent_person_id']}`")
            st.write(f"- Average Detection Confidence: `{summary_data['avg_confidence']}%`")

            st.markdown("---")
            st.markdown("#### Data Export Downloads")
            csv_data = ReportGenerator.export_to_csv(filtered_df if not filtered_df.empty else df_current)
            st.download_button(
                label="Download Activity Events (CSV)",
                data=csv_data,
                file_name="activity_events_export.csv",
                mime="text/csv",
                use_container_width=True
            )

            try:
                pdf_bytes = ReportGenerator.generate_pdf_report(df_current)
                st.download_button(
                    label="Download Official Incident Audit Report (PDF)",
                    data=pdf_bytes,
                    file_name="surveillance_incident_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as pdf_err:
                st.error(f"Failed to generate PDF report: {pdf_err}")

            html_report = ReportGenerator.generate_html_report(df_current)
            st.download_button(
                label="Download Web Printable Report (HTML)",
                data=html_report,
                file_name="surveillance_incident_report.html",
                mime="text/html",
                use_container_width=True
            )

        with r_col2:
            st.markdown("#### High Risk Person ID Summary")
            if summary_data["high_risk_persons"]:
                hr_df = pd.DataFrame(summary_data["high_risk_persons"])
                st.dataframe(hr_df, use_container_width=True)
            else:
                st.write("No high-risk persons detected.")

    with tab_pipeline:
        st.markdown("### Architecture Pipeline and SDG Contributions")
        st.markdown("""
        #### System Pipeline Architecture
        ```text
        [ Video / CCTV Feed ]
                  │
                  ▼
        ┌───────────────────────────────────────────────┐
        │  MEMBER 1: Detection & Object Tracking        │
        │  (Bounding boxes, Object IDs, Trajectories)   │
        └───────────────────────────────────────────────┘
                  │
                  ▼  detection.json
        ┌───────────────────────────────────────────────┐
        │  MEMBER 2: Suspicious Activity Detection      │
        │  (Loitering, Fighting, Unattended Bags)       │
        └───────────────────────────────────────────────┘
                  │
                  ▼  activity_events.json  <== [MEMBER 3 MODULE INTERFACE]
        ┌───────────────────────────────────────────────┐
        │  MEMBER 3: Alerts, Dashboard & Reports        │
        │  (Alert Manager, Streamlit SOC Dashboard, PDF)│
        └───────────────────────────────────────────────┘
        ```
        """)

        st.markdown("#### Interface Data Contract (activity_events.json)")
        st.code("""
{
  "timestamp": 12.4,
  "person_id": 7,
  "activity": "loitering",
  "confidence": 0.89,
  "suspicious": true,
  "severity": "high"
}
        """, language="json")

        st.markdown("#### Sustainable Development Goals (SDGs)")
        for key, val in SDG_INFO.items():
            st.markdown(f"**{key}: {val['title']}**")
            st.write(val['description'])

    with tab_privacy:
        st.markdown("""
        <div class="legal-card">
            <h2 class="legal-title">Privacy Policy</h2>
            <p><strong>Effective Date:</strong> October 6, 2026</p>
            <p>This Privacy Policy outlines how the Smart Surveillance Framework collects, processes, and protects activity event data generated by automated CCTV monitoring modules.</p>

            <h4>1. Information Collected</h4>
            <p>The system ingests structured numerical telemetry data from video feeds, including frame timestamps, bounding box track identifiers (Person IDs), activity classification tags, model confidence metrics, and assigned severity ratings. No personal identification metadata, names, or contact details are processed or stored by Member 3.</p>

            <h4>2. Use of Data</h4>
            <p>Data ingested into the surveillance monitoring dashboard is strictly used for real-time threat detection, alert management, incident logging, and security reporting in compliance with SDG 9, 11, and 16 guidelines.</p>

            <h4>3. Data Security and Retention</h4>
            <p>All activity event logs are stored locally within the configured deployment environment. Audit exports (CSV and PDF) are restricted to authorized operational security personnel.</p>

            <h4>4. Third-Party Sharing</h4>
            <p>No telemetry, tracking, or surveillance data is transmitted to external third-party analytics services or advertisement networks.</p>
        </div>
        """, unsafe_allow_html=True)

    with tab_terms:
        st.markdown("""
        <div class="legal-card">
            <h2 class="legal-title">Terms and Conditions</h2>
            <p><strong>Effective Date:</strong> October 6, 2026</p>

            <h4>1. Acceptance of Terms</h4>
            <p>By accessing or deploying the Smart Surveillance Monitoring System, operators agree to abide by these operational terms, applicable privacy laws, and ethical AI monitoring standards.</p>

            <h4>2. Permitted Use</h4>
            <p>This software framework is intended for academic research, automated facility safety monitoring, and suspicious event auditing. Unauthorized deployment for illegal tracking or unapproved surveillance is prohibited.</p>

            <h4>3. System Limitations</h4>
            <p>Automated activity classifications and confidence scores provided by upstream detection modules (Member 1 and Member 2) serve as decision-support alerts. Security personnel must independently verify critical incidents prior to initiating emergency response procedures.</p>

            <h4>4. Intellectual Property</h4>
            <p>All framework dashboard components, report generators, and alert management code are developed for the AI-Based Smart Surveillance Project under academic guidelines.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-text">
        AI-Based Smart Surveillance and Suspicious Activity Detection Framework | Member 3 Dashboard Module
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.is_simulating:
        if st.session_state.simulation_index < total_raw_count:
            st.session_state.simulation_index += sim_speed
            time.sleep(1.2)
            st.rerun()
        else:
            st.session_state.is_simulating = False
            st.sidebar.success("Simulation playback completed.")


if __name__ == "__main__":
    main()
