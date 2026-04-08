import logging
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import our models (WITHOUT the engine) and our config
from models import DimStudent, DimCourse, DimSemester, DimDemographics, StudentRiskFact
import config

logger = logging.getLogger(__name__)

# Create the engine right here using our configuration
engine = create_engine(config.DATABASE_URI)
Session = sessionmaker(bind=engine)

def load_data(transformed_datasets):
    logger.info("Starting load phase...")
    session = Session()
    
    try:
        # 1. Combine all 3 datasets into one unified Pandas DataFrame 
        # (Assuming you renamed columns to match during the transform phase)
        df_unified = pd.concat(transformed_datasets.values(), ignore_index=True)
        
        # 2. Extract unique Dimension data (e.g., distinct students)
        students_df = df_unified[['age', 'sex', 'address', 'family_size', 'parent_edu', 'internet_access', 'paid_classes']].drop_duplicates()
        
        # 3. Bulk insert Dimensions
        # to_dict('records') converts the dataframe into a list of dictionaries that SQLAlchemy loves
        session.bulk_insert_mappings(DimStudent, students_df.to_dict(orient='records'))
        session.commit() # Commit so database generates the student_id Primary Keys
        
        # 4. Fetch the generated IDs back (you would merge these back into df_unified)
        # ... logic to merge student_id, course_id, etc., back to the main dataframe ...
        
        # 5. Bulk insert the Fact Table
        fact_df = df_unified[['student_id', 'course_id', 'semester_id', 'demo_id', 'gpa', 'absences', 'risk_score', 'risk_label', 'checkpoint_week']]
        session.bulk_insert_mappings(StudentRiskFact, fact_df.to_dict(orient='records'))
        session.commit()
        
        logger.info("Load complete successfully!")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error during load: {e}")
        raise
    finally:
        session.close()