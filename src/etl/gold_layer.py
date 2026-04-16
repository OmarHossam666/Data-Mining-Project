"""
Phase 9 — Gold Layer Generation.

Generates the curated ``student_risk_mart.csv`` from the Silver Layer
star-schema warehouse.  Adds human-readable labels, KPI brackets, and
risk status for downstream BI consumption (dashboard, reports).

Usage:
    python src/etl/gold_layer.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

# Ensure src/ is importable
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GOLD_DIR = config.GOLD_DATA_DIR
GOLD_DIR.mkdir(parents=True, exist_ok=True)


def generate_gold_layer() -> pd.DataFrame:
    """Generate the Gold Layer datamart from the Silver star schema.

    Returns:
        Gold-layer DataFrame with human-readable labels and KPI brackets.
    """
    logger.info("Extracting data from Star Schema for Gold Layer ...")
    engine = create_engine(config.DATABASE_URI)

    query = """
    SELECT
        s.student_id,
        f.fact_id,
        sem.year,
        sem.period,
        sem.checkpoint_week,
        s.age,
        s.sex,
        s.address,
        s.family_size,
        s.parent_edu,
        s.internet_access,
        s.paid_classes,
        c.subject,
        c.school,
        f.study_time,
        f.failures_history,
        f.extracurricular,
        d.nationality,
        d.socioeconomic_tier,
        d.first_gen_student,
        d.scholarship_flag,
        f.gpa,
        f.absences,
        f.risk_score,
        f.risk_label
    FROM fact_student_risk f
    JOIN dim_student s     ON f.student_id  = s.student_id
    JOIN dim_course c      ON f.course_id   = c.course_id
    JOIN dim_semester sem   ON f.semester_id = sem.semester_id
    JOIN dim_demographics d ON f.demo_id    = d.demo_id
    """

    df = pd.read_sql(query, engine)
    logger.info("Loaded %d rows from Silver layer.", len(df))

    # --- Transformations & Business Logic ---
    logger.info("Applying Gold Layer transformations & KPIs ...")

    # 1. Human-readable mappings
    df["sex"] = df["sex"].map({"M": "Male", "F": "Female"}).fillna(df["sex"])
    df["address"] = df["address"].map({"U": "Urban", "R": "Rural"}).fillna(df["address"])

    # 2. Risk status label
    df["risk_status"] = df["risk_label"].map({1: "High Risk", 0: "Low Risk"}).fillna("Unknown")

    # 3. Attendance tier
    def _attendance_tier(absences: float | None) -> str:
        if pd.isna(absences):
            return "Unknown"
        if absences <= 2:
            return "High Attendance (0-2)"
        elif absences <= 8:
            return "Medium Attendance (3-8)"
        return "Low Attendance (>8)"

    df["attendance_tier"] = df["absences"].apply(_attendance_tier)

    # 4. GPA bracket
    def _gpa_bracket(gpa: float | None) -> str:
        if pd.isna(gpa):
            return "Unknown"
        if gpa >= 14.0:
            return "Excellent"
        elif gpa >= 12.0:
            return "Good"
        elif gpa >= 10.0:
            return "Average"
        return "Poor"

    df["gpa_bracket"] = df["gpa"].apply(_gpa_bracket)

    # 5. Data quality: drop records without critical keys
    df = df.dropna(subset=["student_id", "fact_id"])

    # 6. Boolean columns
    bool_cols = ["internet_access", "paid_classes", "extracurricular",
                 "first_gen_student", "scholarship_flag"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    # 7. Reorder columns for BI readability
    col_order = [
        "student_id", "fact_id", "year", "period", "checkpoint_week",
        "age", "sex", "address", "family_size", "parent_edu",
        "internet_access", "paid_classes",
        "school", "subject", "study_time", "failures_history", "extracurricular",
        "nationality", "socioeconomic_tier", "first_gen_student", "scholarship_flag",
        "gpa", "gpa_bracket", "absences", "attendance_tier",
        "risk_score", "risk_label", "risk_status",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    # 8. Save
    output_path = GOLD_DIR / "student_risk_mart.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Gold Layer saved → %s (%d rows × %d cols)", output_path, *df.shape)

    return df


if __name__ == "__main__":
    generate_gold_layer()
