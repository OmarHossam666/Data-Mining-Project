import logging
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import our models and config
from models import DimStudent, DimCourse, DimSemester, DimDemographics, StudentRiskFact
import config

logger = logging.getLogger(__name__)

engine = create_engine(config.DATABASE_URI)
Session = sessionmaker(bind=engine)

def ensure_columns(df, expected_columns):
    """Helper function to prevent KeyErrors by adding missing columns as None."""
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
    return df

def load_data(transformed_datasets):
    """Loads transformed data into the Star Schema using SQLAlchemy."""
    logger.info("Starting load phase...")
    session = Session()
    
    try:
        # 1. Combine all datasets into one unified DataFrame
        df_unified = pd.concat(transformed_datasets.values(), ignore_index=True)
        
        # --- DIM STUDENT ---
        student_cols = ['age', 'sex', 'address', 'family_size', 'parent_edu', 'internet_access', 'paid_classes']
        df_unified = ensure_columns(df_unified, student_cols)
        
        students_df = df_unified[student_cols].drop_duplicates()
        session.bulk_insert_mappings(DimStudent, students_df.to_dict(orient='records'))
        
        # Commit to generate IDs
        session.commit()
        
        # Validation
        student_count = session.query(DimStudent).count()
        logger.info(f"Load complete. DimStudent row count: {student_count}")
        
        # (Note: You will add similar blocks here for Course, Semester, Demographics, and Fact tables
        # once your feature mapping worksheet is complete!)
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error during load: {e}")
        raise
    finally:
        session.close()