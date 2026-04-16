"""
ProGrade Star-Schema Data Warehouse Models (Silver Layer).

Defines four dimension tables and one fact table using SQLAlchemy ORM.
The grain is **one row per student-course-semester observation**.

Schema follows Kimball star-schema principles:
  - Dimensions hold descriptive attributes (slowly changing)
  - Fact table holds numeric measurements and foreign keys
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean, Column, Float, ForeignKey, Index, Integer, String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# ---------------------------------------------------------------------------
# Base class (SQLAlchemy 2.0+ style)
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Dimension Tables
# ---------------------------------------------------------------------------

class DimStudent(Base):
    """Personal, family, and social attributes of each student."""

    __tablename__ = "dim_student"

    student_id: int = Column(Integer, primary_key=True, autoincrement=True)
    age: int | None = Column(Integer)
    sex: str | None = Column(String(1))           # 'M' or 'F'
    address: str | None = Column(String(10))      # 'U' (Urban) or 'R' (Rural)
    family_size: str | None = Column(String(10))  # 'LE3' or 'GT3'
    parent_edu: str | None = Column(String(50))
    internet_access: bool | None = Column(Boolean)
    paid_classes: bool | None = Column(Boolean)

    # Back-reference
    facts = relationship("StudentRiskFact", back_populates="student")

    def __repr__(self) -> str:
        return f"<DimStudent(id={self.student_id}, age={self.age}, sex={self.sex})>"


class DimCourse(Base):
    """Course/subject context for the enrollment."""

    __tablename__ = "dim_course"

    course_id: int = Column(Integer, primary_key=True, autoincrement=True)
    subject: str | None = Column(String(50))
    school: str | None = Column(String(50))        # dataset source identifier

    # Back-reference
    facts = relationship("StudentRiskFact", back_populates="course")

    def __repr__(self) -> str:
        return f"<DimCourse(id={self.course_id}, subject={self.subject})>"


class DimSemester(Base):
    """Temporal dimension capturing the academic timeline."""

    __tablename__ = "dim_semester"

    semester_id: int = Column(Integer, primary_key=True, autoincrement=True)
    year: int | None = Column(Integer)
    period: str | None = Column(String(10))
    checkpoint_week: int | None = Column(Integer)  # Fixed: was String(20)

    facts = relationship("StudentRiskFact", back_populates="semester")

    def __repr__(self) -> str:
        return f"<DimSemester(id={self.semester_id}, week={self.checkpoint_week})>"


class DimDemographics(Base):
    """Broader socioeconomic and demographic variables."""

    __tablename__ = "dim_demographics"

    demo_id: int = Column(Integer, primary_key=True, autoincrement=True)
    nationality: str | None = Column(String(50))
    socioeconomic_tier: str | None = Column(String(20))
    first_gen_student: bool | None = Column(Boolean)
    scholarship_flag: bool | None = Column(Boolean)

    facts = relationship("StudentRiskFact", back_populates="demographics")

    def __repr__(self) -> str:
        return f"<DimDemographics(id={self.demo_id})>"


# ---------------------------------------------------------------------------
# Fact Table
# ---------------------------------------------------------------------------

class StudentRiskFact(Base):
    """Central fact table — one row per student-course-semester observation.

    Contains numeric measurements (GPA, absences, risk metrics) and
    student-level academic behaviour attributes that were previously
    mis-placed in DimCourse.
    """

    __tablename__ = "fact_student_risk"

    fact_id: int = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign keys
    student_id: int = Column(Integer, ForeignKey("dim_student.student_id"), nullable=False)
    course_id: int = Column(Integer, ForeignKey("dim_course.course_id"), nullable=False)
    semester_id: int = Column(Integer, ForeignKey("dim_semester.semester_id"), nullable=False)
    demo_id: int = Column(Integer, ForeignKey("dim_demographics.demo_id"), nullable=False)

    # Measurements
    gpa: float | None = Column(Float)
    absences: int | None = Column(Integer)
    risk_score: float | None = Column(Float)
    risk_label: int | None = Column(Integer)      # 0 = Low Risk, 1 = High Risk

    # Student-course behaviour (moved from DimCourse — these are per-enrollment)
    study_time: int | None = Column(Integer)
    failures_history: int | None = Column(Integer)
    extracurricular: bool | None = Column(Boolean)

    # Relationships
    student = relationship("DimStudent", back_populates="facts")
    course = relationship("DimCourse", back_populates="facts")
    semester = relationship("DimSemester", back_populates="facts")
    demographics = relationship("DimDemographics", back_populates="facts")

    # Performance indexes
    __table_args__ = (
        Index("ix_fact_student", "student_id"),
        Index("ix_fact_course", "course_id"),
        Index("ix_fact_semester", "semester_id"),
        Index("ix_fact_demo", "demo_id"),
        Index("ix_fact_risk", "risk_label"),
    )

    def __repr__(self) -> str:
        return (
            f"<StudentRiskFact(id={self.fact_id}, "
            f"gpa={self.gpa}, risk={self.risk_label})>"
        )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_db(database_uri: str | None = None) -> None:
    """Create all tables in the database.

    Args:
        database_uri: SQLAlchemy connection string. Defaults to config value.
    """
    if database_uri is None:
        from src import config
        database_uri = config.DATABASE_URI

    engine = create_engine(database_uri)
    Base.metadata.create_all(engine)
    print(f"Database initialized at: {database_uri}")


if __name__ == "__main__":
    init_db()