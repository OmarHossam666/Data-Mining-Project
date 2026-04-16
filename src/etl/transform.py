"""
Phase 3 — Data Transformation.

Cleans, imputes, engineers the binary ``risk_label``, and persists processed
datasets to ``data/processed/`` as full-fidelity CSVs.  Also returns a dict
of DataFrames for downstream warehouse loading.

Key design decisions:
  - **No premature normalization** — scaling is deferred to the Feature
    Engineering phase where it is fit on training data only.
  - **No target-source leakage** — columns used to derive ``risk_label``
    (G3, GradeClass, Target) are explicitly flagged and excluded from the
    ML feature set in downstream phases.
  - **Processed CSVs preserve ALL columns** — the star-schema warehouse
    captures a subset; the ML pipeline reads from processed CSVs for the
    full feature set.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Resolve paths relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Columns that were used to derive risk_label — MUST NOT be used as features
# ---------------------------------------------------------------------------

TARGET_SOURCE_COLUMNS: dict[str, list[str]] = {
    "uci_student": ["G3"],
    "uci_dropout": ["Target"],
    "kaggle": ["GradeClass", "GPA"],   # GradeClass IS a binned GPA
}

# ---------------------------------------------------------------------------
# Per-dataset canonical column renaming
# ---------------------------------------------------------------------------

_RENAME_UCI_STUDENT: dict[str, str] = {
    "famsize": "family_size",
    "internet": "internet_access",
    "paid": "paid_classes",
    "studytime": "study_time",
    "failures": "failures_history",
    "activities": "extracurricular",
}

_RENAME_UCI_DROPOUT: dict[str, str] = {
    "Age at enrollment": "age",
    "Gender": "sex",
    "Curricular units 2nd sem (grade)": "gpa",
    "Course": "subject",
    "Scholarship holder": "scholarship_flag",
    "Nacionality": "nationality",
}

_RENAME_KAGGLE: dict[str, str] = {
    "Age": "age",
    "Gender": "sex",
    "ParentalEducation": "parent_edu",
    "StudyTimeWeekly": "study_time",
    "Absences": "absences",
    "Extracurricular": "extracurricular",
    "Tutoring": "paid_classes",
    "GPA": "gpa",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOOL_MAP = {
    "yes": True, "no": False,
    "Yes": True, "No": False,
    1: True, 0: False,
    True: True, False: False,
}

BOOL_COLUMNS: list[str] = [
    "internet_access", "paid_classes", "extracurricular",
    "first_gen_student", "scholarship_flag",
]


def _safe_bool_convert(series: pd.Series) -> pd.Series:
    """Map a column to boolean, preserving NaN for unmapped values."""
    return series.map(_BOOL_MAP).astype("boolean")


def _impute_missing(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Impute numeric columns with median, categorical with mode.

    Entirely-null columns are dropped with a warning.
    """
    df = df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in num_cols:
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue
        median_val = df[col].median()
        if pd.isna(median_val):
            logger.warning("[%s] Column '%s' is entirely null — dropping.", dataset_name, col)
            df = df.drop(columns=[col])
        else:
            df[col] = df[col].fillna(median_val)
            logger.debug("[%s] Imputed %d nulls in '%s' with median=%.2f", dataset_name, n_missing, col, median_val)

    for col in cat_cols:
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue
        mode_result = df[col].mode()
        if mode_result.empty:
            logger.warning("[%s] Column '%s' is entirely null — dropping.", dataset_name, col)
            df = df.drop(columns=[col])
        else:
            df[col] = df[col].fillna(mode_result.iloc[0])

    return df


# ---------------------------------------------------------------------------
# Per-dataset transform functions
# ---------------------------------------------------------------------------

def _transform_uci_student(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer risk_label for UCI Student Performance dataset."""
    df = df.rename(columns=_RENAME_UCI_STUDENT)

    # Boolean conversion
    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col] = _safe_bool_convert(df[col])

    # Sex mapping (already 'M'/'F' in this dataset — no conversion needed)

    # Target engineering: G3 < 10 → High Risk
    if "G3" not in df.columns:
        raise ValueError("UCI Student dataset missing 'G3' column for target engineering")
    df["risk_label"] = (df["G3"] < 10).astype(int)
    logger.info("[uci_student] risk_label engineered from G3 < 10 — distribution: %s",
                df["risk_label"].value_counts().to_dict())

    return df


def _transform_uci_dropout(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer risk_label for UCI Dropout & Academic Success dataset."""
    df = df.rename(columns=_RENAME_UCI_DROPOUT)

    # Sex mapping: 1 → 'M', 0 → 'F'
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({1: "M", 0: "F"}).fillna(df["sex"])

    # Boolean conversion
    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col] = _safe_bool_convert(df[col])

    # Target engineering: 'Dropout' → 1, else → 0
    if "Target" not in df.columns:
        raise ValueError("UCI Dropout dataset missing 'Target' column")
    df["risk_label"] = (df["Target"] == "Dropout").astype(int)
    logger.info("[uci_dropout] risk_label engineered from Target=='Dropout' — distribution: %s",
                df["risk_label"].value_counts().to_dict())

    return df


def _transform_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer risk_label for Kaggle Student Performance dataset."""
    df = df.rename(columns=_RENAME_KAGGLE)

    # Sex mapping: 0 → 'F', 1 → 'M'
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({0: "F", 1: "M"}).fillna(df["sex"])

    # Boolean conversion
    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col] = _safe_bool_convert(df[col])

    # Target engineering: GradeClass >= 3 (D or F) → High Risk
    if "GradeClass" not in df.columns:
        raise ValueError("Kaggle dataset missing 'GradeClass' column")
    df["risk_label"] = (df["GradeClass"] >= 3).astype(int)
    logger.info("[kaggle] risk_label engineered from GradeClass >= 3 — distribution: %s",
                df["risk_label"].value_counts().to_dict())

    return df


_TRANSFORM_DISPATCH: dict[str, callable] = {
    "uci_student": _transform_uci_student,
    "uci_dropout": _transform_uci_dropout,
    "kaggle": _transform_kaggle,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform_data(
    raw_datasets: dict[str, pd.DataFrame],
    save_processed: bool = True,
) -> dict[str, pd.DataFrame]:
    """Clean, impute, and engineer risk labels for all datasets.

    Args:
        raw_datasets: Dict from :func:`extract_data`.
        save_processed: If True, persist full-fidelity CSVs to
            ``data/processed/`` for the ML pipeline.

    Returns:
        Dict mapping dataset name to cleaned DataFrame.
    """
    logger.info("Starting transformation phase ...")
    transformed: dict[str, pd.DataFrame] = {}

    for name, df in raw_datasets.items():
        logger.info("Transforming dataset: %s (%d rows × %d cols) ...", name, *df.shape)
        df_clean = df.copy()

        # 1. Dataset-specific cleaning + risk_label derivation
        transform_fn = _TRANSFORM_DISPATCH.get(name)
        if transform_fn is None:
            logger.warning("No transform function for dataset '%s' — passing through with imputation only.", name)
        else:
            df_clean = transform_fn(df_clean)

        # 2. Missing-value imputation (median / mode)
        df_clean = _impute_missing(df_clean, name)

        # 3. Validate risk_label was created
        if "risk_label" not in df_clean.columns:
            logger.warning("[%s] risk_label was NOT created — check transform logic!", name)

        # 4. Save full-fidelity processed CSV (for ML pipeline)
        if save_processed:
            out_path = _PROCESSED_DIR / f"{name}_clean.csv"
            df_clean.to_csv(out_path, index=False)
            logger.info("[%s] Saved processed CSV → %s (%d cols preserved)",
                        name, out_path, len(df_clean.columns))

        transformed[name] = df_clean

    logger.info("Transformation phase complete — %d datasets processed.", len(transformed))
    return transformed