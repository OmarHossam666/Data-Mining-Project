from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship
import sys
from pathlib import Path

# Import our config from Phase 0
import config

Base = declarative_base()

# --- DIMENSION TABLES ---

class DimStudent(Base):
    __tablename__ = 'dim_student'
    student_id = Column(Integer, primary_key=True, autoincrement=True)
    age = Column(Integer)
    sex = Column(String(1)) # 'M' or 'F'
    address = Column(String(10)) # 'U' (Urban) or 'R' (Rural)
    family_size = Column(String(10)) # 'LE3' or 'GT3'
    parent_edu = Column(String(50)) # e.g., 'higher education', 'primary'
    internet_access = Column(Boolean)
    paid_classes = Column(Boolean)

class DimCourse(Base):
    __tablename__ = 'dim_course'
    course_id = Column(Integer, primary_key=True, autoincrement=True)
    subject = Column(String(50)) # e.g., 'Math', 'Portuguese'
    school = Column(String(50))
    study_time = Column(Integer) # e.g., 1 to 4 scale
    failures_history = Column(Integer)
    extracurricular = Column(Boolean)

class DimSemester(Base):
    __tablename__ = 'dim_semester'
    semester_id = Column(Integer, primary_key=True, autoincrement=True)
    year = Column(Integer)
    period = Column(String(2)) # 'G1', 'G2', 'G3'
    checkpoint_week = Column(String(20)) # '4', '8', '12', 'final'

class DimDemographics(Base):
    __tablename__ = 'dim_demographics'
    demo_id = Column(Integer, primary_key=True, autoincrement=True)
    nationality = Column(String(50))
    socioeconomic_tier = Column(String(20))
    first_gen_student = Column(Boolean)
    scholarship_flag = Column(Boolean)

# --- FACT TABLE ---

class StudentRiskFact(Base):
    __tablename__ = 'fact_student_risk'
    # Fact tables often have a surrogate primary key, or use a composite key. 
    # We will add an ID for simple ORM tracking.
    fact_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Keys linking to Dimensions
    student_id = Column(Integer, ForeignKey('dim_student.student_id'), nullable=False)
    course_id = Column(Integer, ForeignKey('dim_course.course_id'), nullable=False)
    semester_id = Column(Integer, ForeignKey('dim_semester.semester_id'), nullable=False)
    demo_id = Column(Integer, ForeignKey('dim_demographics.demo_id'), nullable=False)
    
    # Lean Measurements / Metrics
    gpa = Column(Float)
    absences = Column(Integer)
    risk_score = Column(Float) # Calculated probability of dropping out/failing
    risk_label = Column(Integer) # 0 = Low Risk, 1 = High Risk
    checkpoint_week = Column(Integer) # Numeric week representation

# --- Initialization Function ---
def init_db():
    """Creates the SQLite database and all tables."""
    engine = create_engine(config.DATABASE_URI)
    Base.metadata.create_all(engine)
    print(f"Database successfully initialized at: {config.DB_PATH}")

if __name__ == "__main__":
    init_db()