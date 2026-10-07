"""
Alert Manager Module for Member 3 Surveillance Dashboard.
Handles suspicious event identification, severity categorization, active alerts state,
alert audio notifications, and alert history management.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
from config import SEVERITY_COLORS


class AlertManager:
    """Manages active suspicious alerts, severity ranking, and notification rendering."""

    def __init__(self, events: List[Dict[str, Any]]):
        self.events = events
        self.suspicious_events = [e for e in events if e.get("suspicious", False)]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates key metrics for the alert system."""
        total_events = len(self.events)
        total_suspicious = len(self.suspicious_events)
        
        critical_alerts = [e for e in self.suspicious_events if e.get("severity") == "critical"]
        high_alerts = [e for e in self.suspicious_events if e.get("severity") == "high"]
        medium_alerts = [e for e in self.suspicious_events if e.get("severity") == "medium"]
        low_alerts = [e for e in self.suspicious_events if e.get("severity") == "low"]

        avg_confidence = (
            sum(e["confidence"] for e in self.events) / total_events
            if total_events > 0 else 0.0
        )

        suspicious_rate = (total_suspicious / total_events * 100.0) if total_events > 0 else 0.0

        return {
            "total_events": total_events,
            "total_suspicious": total_suspicious,
            "critical_count": len(critical_alerts),
            "high_count": len(high_alerts),
            "medium_count": len(medium_alerts),
            "low_count": len(low_alerts),
            "high_critical_total": len(critical_alerts) + len(high_alerts),
            "suspicious_rate": round(suspicious_rate, 1),
            "avg_confidence": round(avg_confidence * 100.0, 1)
        }

    def get_high_priority_alerts(self) -> List[Dict[str, Any]]:
        """Returns high and critical severity suspicious alerts."""
        return [
            e for e in self.suspicious_events 
            if e.get("severity") in ["high", "critical"]
        ]

    def get_latest_critical_alert(self) -> Tuple[bool, Dict[str, Any]]:
        """Returns the most recent high or critical alert if available."""
        high_priority = self.get_high_priority_alerts()
        if high_priority:
            sorted_alerts = sorted(high_priority, key=lambda x: x["timestamp"], reverse=True)
            return True, sorted_alerts[0]
        return False, {}

    @staticmethod
    def get_severity_badge_html(severity: str) -> str:
        """Returns a crisp rectangular badge tag for a given severity."""
        sev = severity.lower()
        color = SEVERITY_COLORS.get(sev, "#9e9e9e")
        bg_color = color + "1a"  # 10% tint
        
        return f"""<span style="
            background-color: {bg_color};
            color: {color};
            border: 1px solid {color};
            padding: 3px 8px;
            border-radius: 2px;
            font-weight: 700;
            font-size: 0.8rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            display: inline-block;
        ">{sev}</span>"""

    @staticmethod
    def get_audio_alert_html(enabled: bool = True) -> str:
        """
        Generates HTML/JavaScript that uses Web Audio API to produce an audible warning beep.
        Executes client-side in the browser window.
        """
        if not enabled:
            return ""

        return """
        <script>
            try {
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                if (AudioContext) {
                    const ctx = new AudioContext();
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    
                    osc.type = 'sawtooth';
                    osc.frequency.setValueAtTime(880, ctx.currentTime);
                    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.3);
                    
                    gain.gain.setValueAtTime(0.12, ctx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
                    
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    
                    osc.start();
                    osc.stop(ctx.currentTime + 0.3);
                }
            } catch (e) {
                console.log("Audio notification status:", e);
            }
        </script>
        """
