"""
Report Generator Module for Member 3 Surveillance Dashboard.
Generates CSV exports, structured incident summaries, and PDF/HTML audit reports.
"""

import io
import pandas as pd
from typing import Dict, Any, Optional
from fpdf import FPDF


class ReportGenerator:
    """Generates downloadable security incident reports and data exports."""

    @staticmethod
    def export_to_csv(df: pd.DataFrame) -> str:
        """Exports DataFrame to CSV formatted string."""
        if df.empty:
            return ""
        return df.to_csv(index=False)

    @staticmethod
    def generate_incident_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates detailed incident statistics for security summary."""
        if df.empty:
            return {
                "total_events": 0,
                "suspicious_events": 0,
                "most_common_activity": "N/A",
                "most_frequent_person": "N/A",
                "high_risk_persons": []
            }

        total_events = len(df)
        suspicious_df = df[df["suspicious"] == True]
        suspicious_count = len(suspicious_df)

        most_common_activity = (
            suspicious_df["activity"].mode()[0] if not suspicious_df.empty else "None"
        )
        
        most_frequent_person = (
            int(suspicious_df["person_id"].mode()[0]) if not suspicious_df.empty else "None"
        )

        person_counts = suspicious_df["person_id"].value_counts().to_dict() if not suspicious_df.empty else {}
        high_risk_persons = [
            {"person_id": pid, "suspicious_incidents": count}
            for pid, count in person_counts.items()
        ]

        return {
            "total_events": total_events,
            "suspicious_events": suspicious_count,
            "suspicious_percentage": round((suspicious_count / total_events) * 100.0, 1) if total_events > 0 else 0.0,
            "most_common_activity": most_common_activity.replace("_", " ").title(),
            "most_frequent_person_id": most_frequent_person,
            "high_risk_persons": high_risk_persons,
            "avg_confidence": round(df["confidence"].mean() * 100.0, 1) if total_events > 0 else 0.0
        }

    @staticmethod
    def generate_pdf_report(df: pd.DataFrame, title: str = "Surveillance Incident Audit Report") -> bytes:
        """
        Generates a clean, professional security incident PDF report using fpdf2.
        Returns PDF content as bytes.
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Header banner
        pdf.set_fill_color(15, 23, 42)
        pdf.rect(0, 0, 210, 36, 'F')
        
        pdf.set_text_color(16, 185, 129)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_xy(10, 10)
        pdf.cell(0, 8, "SMART SURVEILLANCE SECURITY FRAMEWORK", ln=True)
        
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_x(10)
        pdf.cell(0, 7, title, ln=True)
        
        pdf.ln(12)
        
        # Summary Section
        summary = ReportGenerator.generate_incident_summary(df)
        
        pdf.set_text_color(15, 23, 42)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "1. Executive Summary Statistics", ln=True)
        pdf.set_draw_color(16, 185, 129)
        pdf.set_line_width(0.6)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(95, 7, f"Total Captured Events: {summary['total_events']}", border=1)
        pdf.cell(95, 7, f"Suspicious Incidents: {summary['suspicious_events']} ({summary['suspicious_percentage']}%)", border=1, ln=True)
        pdf.cell(95, 7, f"Primary Flagged Activity: {summary['most_common_activity']}", border=1)
        pdf.cell(95, 7, f"Most Flagged Person ID: {summary['most_frequent_person_id']}", border=1, ln=True)
        pdf.cell(190, 7, f"Average Model Confidence: {summary['avg_confidence']}%", border=1, ln=True)
        
        pdf.ln(8)
        
        # UN SDGs Alignment
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "2. Sustainable Development Goals (SDG) Alignment", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, "- SDG 9: Industry, Innovation and Infrastructure (Automated computer vision monitoring)\n- SDG 11: Sustainable Cities and Communities (Enhancing urban safety and public security)\n- SDG 16: Peace, Justice and Strong Institutions (Providing auditable security data logs)")
        
        pdf.ln(8)
        
        # Event History Table
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "3. Suspicious Activity Log Details", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        
        # Table Header
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(241, 245, 249)
        pdf.cell(20, 7, "Event ID", border=1, fill=True)
        pdf.cell(25, 7, "Time (s)", border=1, fill=True)
        pdf.cell(22, 7, "Person ID", border=1, fill=True)
        pdf.cell(50, 7, "Activity", border=1, fill=True)
        pdf.cell(30, 7, "Severity", border=1, fill=True)
        pdf.cell(43, 7, "Confidence", border=1, fill=True, ln=True)
        
        # Table Rows
        pdf.set_font("Helvetica", "", 8.5)
        suspicious_df = df[df["suspicious"] == True] if not df.empty else pd.DataFrame()
        
        if suspicious_df.empty:
            pdf.cell(190, 7, "No suspicious incidents recorded.", border=1, ln=True)
        else:
            for _, row in suspicious_df.head(25).iterrows():
                pdf.cell(20, 6, str(row.get("event_id", "N/A")), border=1)
                pdf.cell(25, 6, f"{row['timestamp']:.1f}s", border=1)
                pdf.cell(22, 6, f"ID #{row['person_id']}", border=1)
                pdf.cell(50, 6, str(row["activity"]).replace("_", " ").title(), border=1)
                pdf.cell(30, 6, str(row["severity"]).upper(), border=1)
                pdf.cell(43, 6, f"{row['confidence']*100:.1f}%", border=1, ln=True)

        return bytes(pdf.output())

    @staticmethod
    def generate_html_report(df: pd.DataFrame, title: str = "Surveillance Security Audit Report") -> str:
        """Generates a clean HTML report suitable for printing or browser view."""
        summary = ReportGenerator.generate_incident_summary(df)
        suspicious_df = df[df["suspicious"] == True] if not df.empty else pd.DataFrame()
        
        rows_html = ""
        if suspicious_df.empty:
            rows_html = "<tr><td colspan='6' style='text-align:center;'>No suspicious incidents recorded.</td></tr>"
        else:
            for _, row in suspicious_df.iterrows():
                sev = str(row["severity"]).lower()
                sev_color = "#ef4444" if sev == "critical" else "#f97316" if sev == "high" else "#f59e0b"
                rows_html += f"""
                <tr>
                    <td>{row.get('event_id', 'N/A')}</td>
                    <td>{row['timestamp']:.1f}s ({row.get('time_formatted', '')})</td>
                    <td><strong>ID #{row['person_id']}</strong></td>
                    <td>{str(row['activity']).replace('_', ' ').title()}</td>
                    <td><span style="color:{sev_color}; font-weight:bold; text-transform:uppercase;">{sev}</span></td>
                    <td>{row['confidence']*100:.1f}%</td>
                </tr>
                """

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 30px; }}
                .header {{ background: #1e293b; padding: 24px; border-radius: 4px; border-left: 4px solid #10b981; margin-bottom: 24px; }}
                h1 {{ margin: 0 0 8px 0; color: #10b981; font-size: 22px; font-weight: 700; }}
                h2 {{ color: #38bdf8; font-size: 16px; border-bottom: 1px solid #334155; padding-bottom: 6px; margin-top: 28px; font-weight: 600; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
                .stat-card {{ background: #1e293b; padding: 14px; border-radius: 4px; border: 1px solid #334155; text-align: center; }}
                .stat-val {{ font-size: 20px; font-weight: 700; color: #f8fafc; }}
                .stat-lbl {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; margin-top: 4px; font-weight: 600; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 12px; background: #1e293b; border-radius: 4px; overflow: hidden; }}
                th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; font-size: 13px; }}
                th {{ background: #0f172a; color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 11px; }}
                .sdg-tag {{ display: inline-block; background: #1e293b; border: 1px solid #38bdf8; color: #38bdf8; padding: 3px 8px; border-radius: 2px; font-size: 11px; margin-right: 6px; font-weight: 600; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI SMART SURVEILLANCE REPORT</h1>
                <p style="margin:0; color:#94a3b8; font-size:13px;">{title} | Generated by Member 3 Monitoring Framework</p>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-val">{summary['total_events']}</div>
                    <div class="stat-lbl">Total Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val" style="color:#ef4444;">{summary['suspicious_events']}</div>
                    <div class="stat-lbl">Suspicious Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val" style="color:#10b981;">{summary['suspicious_percentage']}%</div>
                    <div class="stat-lbl">Suspicious Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{summary['avg_confidence']}%</div>
                    <div class="stat-lbl">Avg ML Confidence</div>
                </div>
            </div>

            <h2>Global SDG Alignment</h2>
            <p>
                <span class="sdg-tag">SDG 9: Industry and Innovation</span>
                <span class="sdg-tag">SDG 11: Sustainable Cities</span>
                <span class="sdg-tag">SDG 16: Peace and Justice</span>
            </p>

            <h2>Suspicious Event Log Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Event ID</th>
                        <th>Timestamp</th>
                        <th>Person ID</th>
                        <th>Activity</th>
                        <th>Severity</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </body>
        </html>
        """
