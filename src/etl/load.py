import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import DimStudent, DimCourse, DimSemester, DimDemographics, StudentRiskFact
import config

logger = logging.getLogger(__name__)

engine = create_engine(config.DATABASE_URI)
Session = sessionmaker(bind=engine)

def ensure_columns(df, expected_columns):
    """Safely adds missing columns as None to prevent KeyErrors."""
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
    return df

def load_data(transformed_datasets):
    logger.info("Starting load phase...")
    session = Session()
    
    try:
        # 1. Combine datasets
        df = pd.concat(transformed_datasets.values(), ignore_index=True)
        
        # 2. Ensure ALL Star Schema columns exist
        schema_cols = [
            'age', 'sex', 'address', 'family_size', 'parent_edu', 'internet_access', 'paid_classes',
            'subject', 'school', 'study_time', 'failures_history', 'extracurricular',
            'year', 'period', 'nationality', 'socioeconomic_tier', 'first_gen_student', 'scholarship_flag',
            'gpa', 'absences', 'risk_score', 'risk_label'
        ]
        df = ensure_columns(df, schema_cols)
        
        # Simulate checkpoints (4, 8, 12) so Phase 4 has temporal data to split!
        df['checkpoint_week'] = np.random.choice([4, 8, 12], size=len(df))
        
        records = df.to_dict(orient='records')
        logger.info(f"Loading {len(records)} unified rows into Dimensions and Fact table...")
        
        # 3. Linearly load dimensions, grab IDs, and load facts
        for row in records:
            # Create Dimensions
            student = DimStudent(age=row['age'], sex=row['sex'], address=row['address'], family_size=row['family_size'], parent_edu=row['parent_edu'], internet_access=row['internet_access'], paid_classes=row['paid_classes'])
            course = DimCourse(subject=row['subject'], school=row['school'], study_time=row['study_time'], failures_history=row['failures_history'], extracurricular=row['extracurricular'])
            semester = DimSemester(year=row['year'], period=row['period'], checkpoint_week=str(row['checkpoint_week']))
            demo = DimDemographics(nationality=row['nationality'], socioeconomic_tier=row['socioeconomic_tier'], first_gen_student=row['first_gen_student'], scholarship_flag=row['scholarship_flag'])
            
            session.add_all([student, course, semester, demo])
            session.flush() # Generates the Primary Keys instantly without hard-committing!
            
            # Create Fact Table record using the newly generated keys
            fact = StudentRiskFact(
                student_id=student.student_id,
                course_id=course.course_id,
                semester_id=semester.semester_id,
                demo_id=demo.demo_id,
                gpa=row['gpa'],
                absences=row['absences'],
                risk_score=row['risk_score'],
                risk_label=row['risk_label'],
                checkpoint_week=row['checkpoint_week']
            )
            session.add(fact)
            
        session.commit()
        logger.info("Load complete! Star Schema is fully populated.")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error during load: {e}")
        raise
    finally:
        session.close()