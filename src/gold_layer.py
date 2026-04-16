import pandas as pd
import logging
from sqlalchemy import create_engine
from pathlib import Path
import sys

# Setup path to import config dynamically
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_DIR))
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GOLD_DIR = config.PROJECT_ROOT / "data" / "gold"
GOLD_DIR.mkdir(parents=True, exist_ok=True)

def generate_gold_layer():
    """Generates the Gold Layer (student_risk_mart.csv) from the Silver layer Star Schema."""
    logger.info("Extracting data from Star Schema for Gold Layer...")
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
        c.study_time, 
        c.failures_history, 
        c.extracurricular,
        d.nationality, 
        d.socioeconomic_tier, 
        d.first_gen_student, 
        d.scholarship_flag,
        f.gpa, 
        f.absences, 
        f.risk_score, 
        f.risk_label
    FROM fact_student_risk f
    JOIN dim_student s ON f.student_id = s.student_id
    JOIN dim_course c ON f.course_id = c.course_id
    JOIN dim_semester sem ON f.semester_id = sem.semester_id
    JOIN dim_demographics d ON f.demo_id = d.demo_id
    """
    
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} rows from Silver layer.")
    
    # Transformations & Business Logic
    logger.info("Applying Gold Layer transformations & KPIs...")
    
    # 1. Human Readable Mappings
    df['sex'] = df['sex'].map({'M': 'Male', 'F': 'Female'}).fillna(df['sex'])
    df['address'] = df['address'].map({'U': 'Urban', 'R': 'Rural'}).fillna(df['address'])
    
    # 2. Add 'risk_status' based on 'risk_label'
    df['risk_status'] = df['risk_label'].apply(lambda x: 'High Risk' if x == 1 else 'Low Risk')
    
    # 3. Create 'attendance_tier'
    def get_attendance_tier(absences):
        if pd.isna(absences):
            return 'Unknown'
        if absences <= 2:
            return 'High Attendance (0-2)'
        elif absences <= 8:
            return 'Medium Attendance (3-8)'
        else:
            return 'Low Attendance (>8)'
            
    df['attendance_tier'] = df['absences'].apply(get_attendance_tier)
    
    # 4. GPA classification
    def get_gpa_bracket(gpa):
        if pd.isna(gpa):
            return 'Unknown'
        if gpa >= 3.8:
             return 'Excellent'
        elif gpa >= 2.5:
             return 'Good'
        elif gpa >= 1.5:
             return 'Average'
        else:
             return 'Poor'
             
    df['gpa_bracket'] = df['gpa'].apply(get_gpa_bracket)
    
    # 5. Data Quality: Handle Nulls and datatypes
    # Drop records without critical keys
    df = df.dropna(subset=['student_id', 'fact_id'])
    
    # Convert booleans where pandas might have cast to object
    bool_cols = ['internet_access', 'paid_classes', 'extracurricular', 'first_gen_student', 'scholarship_flag']
    for col in bool_cols:
         if col in df.columns:
              df[col] = df[col].astype('boolean')
              
    # Reorder columns slightly for better BI readability
    cols = ['student_id', 'fact_id', 'year', 'period', 'checkpoint_week'] + \
           ['age', 'sex', 'address', 'family_size', 'parent_edu', 'internet_access', 'paid_classes'] + \
           ['school', 'subject', 'study_time', 'failures_history', 'extracurricular'] + \
           ['nationality', 'socioeconomic_tier', 'first_gen_student', 'scholarship_flag'] + \
           ['gpa', 'gpa_bracket', 'absences', 'attendance_tier', 'risk_score', 'risk_label', 'risk_status']
    
    df = df[cols]
    
    output_path = GOLD_DIR / "student_risk_mart.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Successfully generated Gold Layer CSV at: {output_path}")
    logger.info(f"Gold Layer dimensions: {df.shape}")
    
    return df

if __name__ == "__main__":
    generate_gold_layer()
