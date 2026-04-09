from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import sys
from pathlib import Path

# Setup path to import config dynamically
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_DIR))
import config

# Action templates keyed by risk-driving feature categories
_ACTION_MAP = {
    'absences': "Review recent attendance logs and address absenteeism patterns.",
    'gpa': "Connect student with academic tutoring and study skills workshops.",
    'failures': "Evaluate course load and consider remedial coursework.",
    'study_time': "Recommend structured study schedule and time-management resources.",
    'default': "Schedule follow-up meeting to assess student progress.",
}


def _generate_actions(risk_tier, top_positive_features):
    """Generate dynamic recommendations based on actual risk drivers."""
    actions = []
    if risk_tier == "High Risk":
        actions.append("Schedule immediate 1-on-1 counseling session.")

    for feature, _ in top_positive_features:
        feat_lower = feature.lower()
        for keyword, action in _ACTION_MAP.items():
            if keyword in feat_lower:
                actions.append(action)
                break

    if not actions:
        actions.append(_ACTION_MAP['default'])

    # Deduplicate while preserving order
    seen = set()
    return [a for a in actions if not (a in seen or seen.add(a))]


def generate_advisor_report(student_id, risk_tier, top_positive_features, top_negative_features):
    """Generates a PDF report for academic advisors based on SHAP explanations."""

    report_path = config.PROJECT_ROOT / "reports" / f"Risk_Report_{student_id}.pdf"

    c = canvas.Canvas(str(report_path), pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "ProGrade Academic Intervention Report")
    c.line(50, height - 60, width - 50, height - 60)

    # Student Info
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 90, f"Student ID: {student_id}")

    # Risk Tier Formatting
    c.drawString(50, height - 110, "Risk Tier: ")
    c.setFont("Helvetica-Bold", 12)
    if risk_tier == "High Risk":
        c.setFillColorRGB(0.8, 0, 0)  # Red
    elif risk_tier == "Medium Risk":
        c.setFillColorRGB(0.8, 0.6, 0)  # Orange
    else:
        c.setFillColorRGB(0, 0.6, 0)  # Green
    c.drawString(110, height - 110, risk_tier)
    c.setFillColorRGB(0, 0, 0)  # Reset to black

    # SHAP Drivers
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 150, "Primary Factors Increasing Risk:")
    c.setFont("Helvetica", 11)
    y_pos = height - 170
    for feature, value in top_positive_features:
        c.drawString(70, y_pos, f"• {feature} (Value: {value:.2f})")
        y_pos -= 20

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos - 20, "Protective Factors Decreasing Risk:")
    c.setFont("Helvetica", 11)
    y_pos -= 40
    for feature, value in top_negative_features:
        c.drawString(70, y_pos, f"• {feature} (Value: {value:.2f})")
        y_pos -= 20

    # Dynamic Recommendations based on actual risk factors
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos - 30, "Recommended Advisor Actions:")
    c.setFont("Helvetica", 11)
    actions = _generate_actions(risk_tier, top_positive_features)
    action_y = y_pos - 50
    for i, action in enumerate(actions, 1):
        c.drawString(70, action_y, f"{i}. {action}")
        action_y -= 20

    c.save()
    print(f"Generated PDF Report: {report_path}")


if __name__ == "__main__":
    # Example usage based on dummy SHAP output
    generate_advisor_report(
        student_id="STU-99482",
        risk_tier="High Risk",
        top_positive_features=[("Absences", 15.0), ("GPA", 1.8), ("Failures_History", 2.0)],
        top_negative_features=[("Study_Time", 4.0)]
    )