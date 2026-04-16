"""
Phase 4 — Warehouse Loading (Silver Layer).

Loads transformed DataFrames into the star-schema SQLite warehouse.
Each dataset is tagged with its source name in the ``school`` column
of :class:`DimCourse` for cross-dataset tracking.

Key design decisions:
  - **No random checkpoint assignment** — temporal checkpoint simulation
    is handled at the Feature Engineering stage via feature masking, not
    by artificially splitting data in the warehouse.
  - **Idempotent** — calling ``load_data()`` drops and recreates all
    tables to prevent duplicate rows on re-runs.
  - ``study_time``, ``failures_history``, ``extracurricular`` stored in
    :class:`StudentRiskFact` (per-enrollment measurements, not course
    attributes).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Resolve project root for config import
_SRC_DIR = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(_SRC_DIR))

from models import Base, DimCourse, DimDemographics, DimSemester, DimStudent, StudentRiskFact
import config

logger = logging.getLogger(__name__)

BATCH_SIZE: int = config.BATCH_SIZE

# Columns we expect in the final concatenated DataFrame
_SCHEMA_COLS: list[str] = [
    "age", "sex", "address", "family_size", "parent_edu",
    "internet_access", "paid_classes",
    "subject", "study_time", "failures_history", "extracurricular",
    "year", "period", "nationality", "socioeconomic_tier",
    "first_gen_student", "scholarship_flag",
    "gpa", "absences", "risk_score", "risk_label",
]


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add missing columns with ``None`` values."""
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df


def _normalize_value(val: object) -> object | None:
    """Convert ``NaN`` / ``NaT`` to ``None`` for clean DB insertion."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def load_data(
    transformed_datasets: dict[str, pd.DataFrame],
    database_uri: str | None = None,
    reset: bool = True,
) -> int:
    """Load transformed data into the star-schema warehouse.

    Args:
        transformed_datasets: Dict from :func:`transform_data`.
        database_uri: SQLAlchemy connection string (defaults to config).
        reset: If True (default), drop and recreate all tables for
            idempotent re-runs.

    Returns:
        Total number of fact rows inserted.
    """
    uri = database_uri or config.DATABASE_URI
    engine = create_engine(uri)
    SessionFactory = sessionmaker(bind=engine)

    # Idempotent: drop + recreate tables on each load
    if reset:
        logger.info("Resetting warehouse tables for idempotent reload ...")
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    logger.info("Starting warehouse load ...")

    # Tag each dataset with its source name
    df_list: list[pd.DataFrame] = []
    for name, df in transformed_datasets.items():
        df_copy = df.copy()
        df_copy["school"] = name
        df_list.append(df_copy)

    df = pd.concat(df_list, ignore_index=True)
    df = _ensure_columns(df, _SCHEMA_COLS)

    # Normalize NaN → None
    df = df.where(pd.notna(df), None)

    # Drop rows where ALL core student fields are null (concat artifacts)
    student_core = ["age", "sex"]
    df = df.dropna(subset=student_core, how="all")

    records = df.to_dict(orient="records")
    total_rows = len(records)
    logger.info("Loading %d rows into star schema ...", total_rows)

    session: Session = SessionFactory()
    inserted = 0

    try:
        for batch_start in range(0, total_rows, BATCH_SIZE):
            batch = records[batch_start : batch_start + BATCH_SIZE]

            for row in batch:
                r = {k: _normalize_value(v) for k, v in row.items()}

                student = DimStudent(
                    age=r.get("age"),
                    sex=r.get("sex"),
                    address=r.get("address"),
                    family_size=r.get("family_size"),
                    parent_edu=r.get("parent_edu"),
                    internet_access=r.get("internet_access"),
                    paid_classes=r.get("paid_classes"),
                )
                course = DimCourse(
                    subject=r.get("subject"),
                    school=r.get("school"),
                )
                semester = DimSemester(
                    year=r.get("year"),
                    period=r.get("period"),
                    checkpoint_week=None,  # Set by feature engineering, not here
                )
                demo = DimDemographics(
                    nationality=r.get("nationality"),
                    socioeconomic_tier=r.get("socioeconomic_tier"),
                    first_gen_student=r.get("first_gen_student"),
                    scholarship_flag=r.get("scholarship_flag"),
                )

                session.add_all([student, course, semester, demo])
                session.flush()

                fact = StudentRiskFact(
                    student_id=student.student_id,
                    course_id=course.course_id,
                    semester_id=semester.semester_id,
                    demo_id=demo.demo_id,
                    gpa=r.get("gpa"),
                    absences=r.get("absences"),
                    risk_score=r.get("risk_score"),
                    risk_label=r.get("risk_label"),
                    study_time=r.get("study_time"),
                    failures_history=r.get("failures_history"),
                    extracurricular=r.get("extracurricular"),
                )
                session.add(fact)
                inserted += 1

            session.commit()
            batch_num = batch_start // BATCH_SIZE + 1
            logger.info(
                "Committed batch %d — %d/%d rows (%.0f%%)",
                batch_num, min(batch_start + BATCH_SIZE, total_rows),
                total_rows, 100 * min(batch_start + BATCH_SIZE, total_rows) / total_rows,
            )

        logger.info("Warehouse load complete — %d fact rows inserted.", inserted)

    except Exception:
        session.rollback()
        logger.exception("Error during warehouse load — rolled back.")
        raise
    finally:
        session.close()

    return inserted