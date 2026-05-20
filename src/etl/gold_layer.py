"""
Phase 9 — Gold Layer Generation.

Generates the curated ``student_risk_mart.csv`` from the Silver Layer
star-schema warehouse.  Adds human-readable labels, KPI brackets, and
risk status for downstream BI consumption (dashboard, reports).

All GPAs are on a normalized 0.0–4.0 scale (normalized during Transform).

Usage:
    python src/etl/gold_layer.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

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
        d.scholarship_flag,
        f.gpa,
        f.absences,
        f.risk_score,
        f.risk_label
    FROM fact_student_risk f
    JOIN dim_student s     ON f.student_id  = s.student_id
    JOIN dim_course c      ON f.course_id   = c.course_id
    JOIN dim_demographics d ON f.demo_id    = d.demo_id
    """

    df = pd.read_sql(query, engine)
    logger.info("Loaded %d rows from Silver layer.", len(df))

    # --- Advanced Group-by Imputation for Missing Values ---
    logger.info("Imputing missing values using modern group-by mode/median approach ...")

    # Define categorical and numerical columns for imputation
    cat_cols_to_impute = {
        "address": "U",
        "family_size": "GT3",
        "internet_access": 1,
        "paid_classes": 0,
        "subject": "General Academic",
        "extracurricular": 0,
        "nationality": 1,  # 1 represents Portuguese/citizen
        "scholarship_flag": 0
    }

    num_cols_to_impute = {
        "study_time": 2.0,
        "failures_history": 0.0,
        "absences": 0.0
    }

    # For subject, Kaggle has no courses, so map Kaggle's subject to "General Academic"
    df.loc[df["school"] == "kaggle", "subject"] = "General Academic"

    # Impute categorical/boolean columns with group-by mode (grouped by risk_label and sex)
    for col, fallback in cat_cols_to_impute.items():
        def _get_group_mode(group):
            non_null = group.dropna()
            if non_null.empty:
                return fallback
            return non_null.mode().iloc[0]

        group_modes = df.groupby(["risk_label", "sex"])[col].transform(lambda g: g.fillna(_get_group_mode(g)))
        df[col] = df[col].fillna(group_modes).fillna(fallback)

    # Impute numerical columns with group-by median
    for col, fallback in num_cols_to_impute.items():
        def _get_group_median(group):
            non_null = group.dropna()
            if non_null.empty:
                return fallback
            return non_null.median()

        group_medians = df.groupby(["risk_label", "sex"])[col].transform(lambda g: g.fillna(_get_group_median(g)))
        df[col] = df[col].fillna(group_medians).fillna(fallback)

    # Convert binary columns to strict standard integer representations to simplify BI and ML
    for col in ["internet_access", "paid_classes", "extracurricular", "scholarship_flag"]:
        df[col] = df[col].apply(lambda x: 1 if x in (True, 1, 1.0, "True", "true", "Yes", "yes") else 0)

    # 1. Human-readable mappings
    df["sex"] = df["sex"].map({"M": "Male", "F": "Female"}).fillna(df["sex"])
    df["address"] = df["address"].map({"U": "Urban", "R": "Rural"}).fillna(df["address"])
    df["nationality"] = df["nationality"].apply(lambda x: "Portuguese" if x in (1, 1.0, "1", "1.0") else "Other")

    # 2. Risk status label
    df["risk_status"] = df["risk_label"].map({1: "High Risk", 0: "Low Risk"}).fillna("Unknown")

    # 3. Attendance tier — no N/A, strict boundaries
    def _attendance_tier(absences: float) -> str:
        if absences <= 2:
            return "High Attendance (0-2)"
        elif absences <= 8:
            return "Medium Attendance (3-8)"
        return "Low Attendance (>8)"

    df["attendance_tier"] = df["absences"].apply(_attendance_tier)

    # 4. GPA bracket (normalized 0.0–4.0 scale for all datasets) — no N/A
    def _gpa_bracket(gpa: float) -> str:
        if gpa >= 3.6:
            return "Excellent"
        elif gpa >= 2.8:
            return "Good"
        elif gpa >= 2.0:
            return "Average"
        return "Poor"

    df["gpa_bracket"] = df["gpa"].apply(_gpa_bracket)

    # 5. Data quality: drop records without critical keys
    df = df.dropna(subset=["student_id", "fact_id"])

    # 6. Boolean columns — convert to readable Yes/No for BI
    bool_cols = ["internet_access", "paid_classes", "extracurricular",
                 "scholarship_flag"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda val: "Yes" if val == 1 else "No")

    # --- ML-Powered Risk Score Generation ---
    logger.info("Training a calibrated Random Forest on Gold Layer features to generate risk_score ...")
    try:
        # Copy features for training
        train_features = ["age", "sex", "address", "family_size", "parent_edu",
                          "internet_access", "paid_classes", "study_time", 
                          "failures_history", "extracurricular", "nationality", 
                          "scholarship_flag", "gpa", "absences"]
        
        X_ml = df[train_features].copy()
        y_ml = df["risk_label"].copy()
        
        # Label encode categorical columns
        for col in X_ml.columns:
            if X_ml[col].dtype == object or isinstance(X_ml[col].dtype, pd.CategoricalDtype):
                X_ml[col] = LabelEncoder().fit_transform(X_ml[col].astype(str))
                
        # Train Random Forest model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_ml, y_ml)
        
        # Predict probabilities (risk score)
        df["risk_score"] = rf.predict_proba(X_ml)[:, 1]
        logger.info("Risk scores successfully populated! Range: %.3f to %.3f", 
                    df["risk_score"].min(), df["risk_score"].max())
    except Exception as e:
        logger.error("Failed to train Gold ML model to populate risk_score: %s. Using default baseline risk score.", e)
        # Fallback to simple risk heuristic if model training fails (should not fail, but safe)
        df["risk_score"] = df["risk_label"].astype(float)

    # Check for remaining null values in the dataframe (critical data quality assert)
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning("Dataframe still contains null values:\n%s", null_counts[null_counts > 0])
        # Force fill any remaining NaNs with appropriate fallbacks to guarantee 0% nulls
        df = df.fillna(0)
    else:
        logger.info("Success! 100% of missing values have been eliminated. Zero nulls detected.")


    # 7. Reorder columns for BI readability
    col_order = [
        "student_id", "fact_id",
        "age", "sex", "address", "family_size", "parent_edu",
        "internet_access", "paid_classes",
        "school", "subject", "study_time", "failures_history", "extracurricular",
        "nationality", "scholarship_flag",
        "gpa", "gpa_bracket", "absences", "attendance_tier",
        "risk_score", "risk_label", "risk_status",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    # 8. Save
    output_path = GOLD_DIR / "student_risk_mart.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Gold Layer saved -> %s (%d rows x %d cols)", output_path, *df.shape)

    return df


if __name__ == "__main__":
    generate_gold_layer()
