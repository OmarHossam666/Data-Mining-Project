import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, DimStudent, DimCourse, DimSemester, DimDemographics, StudentRiskFact
import config

logger = logging.getLogger(__name__)
engine = create_engine(config.DATABASE_URI)
Session = sessionmaker(bind=engine)

BATCH_SIZE = 500


def ensure_columns(df, expected_columns):
    """Adds missing columns with None value to the DataFrame."""
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
    return df


def _normalize_value(val):
    """Convert NaN/NaT to None for clean DB insertion."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def load_data(transformed_datasets):
    """Loads transformed data into star-schema warehouse using batch inserts."""
    logger.info("Starting load phase with Cross-Dataset Tracking...")

    # Ensure all tables exist before inserting (fixes 'no such table' on fresh clones)
    Base.metadata.create_all(engine)

    session = Session()

    try:
        # Inject dataset name into the 'school' column for cross-dataset tracking
        df_list = []
        for name, d in transformed_datasets.items():
            d_copy = d.copy()
            d_copy['school'] = name
            df_list.append(d_copy)

        df = pd.concat(df_list, ignore_index=True)

        schema_cols = [
            'age', 'sex', 'address', 'family_size', 'parent_edu', 'internet_access', 'paid_classes',
            'subject', 'study_time', 'failures_history', 'extracurricular',
            'year', 'period', 'nationality', 'socioeconomic_tier', 'first_gen_student', 'scholarship_flag',
            'gpa', 'absences', 'risk_score', 'risk_label'
        ]
        df = ensure_columns(df, schema_cols)

        # Normalize NaN → None across the entire DataFrame for clean DB insertion
        df = df.where(pd.notna(df), None)

        # Drop rows where all core student dimension fields are null (artifact of concat)
        student_core_cols = ['age', 'sex', 'address']
        df = df.dropna(subset=student_core_cols, how='all')

        # Simulate checkpoints
        df['checkpoint_week'] = np.random.choice([4, 8, 12], size=len(df))

        records = df.to_dict(orient='records')
        logger.info(f"Loading {len(records)} rows into Star Schema...")

        # Batch insert for performance
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]

            for row in batch:
                r = {k: _normalize_value(v) for k, v in row.items()}

                student = DimStudent(
                    age=r['age'], sex=r['sex'], address=r['address'],
                    family_size=r['family_size'], parent_edu=r['parent_edu'],
                    internet_access=r['internet_access'], paid_classes=r['paid_classes']
                )
                course = DimCourse(
                    subject=r['subject'], school=r['school'],
                    study_time=r['study_time'], failures_history=r['failures_history'],
                    extracurricular=r['extracurricular']
                )
                semester = DimSemester(
                    year=r['year'], period=r['period'],
                    checkpoint_week=str(r['checkpoint_week']) if r['checkpoint_week'] is not None else None
                )
                demo = DimDemographics(
                    nationality=r['nationality'], socioeconomic_tier=r['socioeconomic_tier'],
                    first_gen_student=r['first_gen_student'], scholarship_flag=r['scholarship_flag']
                )

                session.add_all([student, course, semester, demo])
                session.flush()

                fact = StudentRiskFact(
                    student_id=student.student_id, course_id=course.course_id,
                    semester_id=semester.semester_id, demo_id=demo.demo_id,
                    gpa=r['gpa'], absences=r['absences'],
                    risk_score=r['risk_score'], risk_label=r['risk_label']
                )
                session.add(fact)

            # Commit after each batch to reduce memory pressure
            session.commit()
            logger.info(f"Committed batch {i // BATCH_SIZE + 1} ({len(batch)} rows)")

        logger.info("Load complete! Dataset sources successfully tracked.")

    except Exception as e:
        session.rollback()
        logger.error(f"Error during load: {e}")
        raise
    finally:
        session.close()