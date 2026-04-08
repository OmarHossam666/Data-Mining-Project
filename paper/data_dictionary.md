# ProGrade Data Warehouse: Data Dictionary

## Architecture Overview
This database utilizes a **Star Schema** designed for analytical queries.

**Grain:** One row per student, per semester, per checkpoint. 
*(Example: If a student is evaluated at weeks 4, 8, and 12 during the G1 semester, there will be 3 distinct rows in the fact table for that student).*

## 1. Fact Table: `fact_student_risk`
Contains the core numeric measurements and foreign keys linking to dimensions.
* `fact_id` (PK, Int): Surrogate key for the fact record.
* `student_id` (FK, Int): Link to `dim_student`.
* `course_id` (FK, Int): Link to `dim_course`.
* `semester_id` (FK, Int): Link to `dim_semester`.
* `demo_id` (FK, Int): Link to `dim_demographics`.
* `gpa` (Float): Grade Point Average at the time of checkpoint.
* `absences` (Int): Total absences accumulated up to the checkpoint.
* `risk_score` (Float): Machine Learning probability score (0.0 to 1.0) of academic failure.
* `risk_label` (Int): Binary classification (0 = Low Risk, 1 = High Risk).
* `checkpoint_week` (Int): Numeric week identifier (e.g., 4).

## 2. Dimension Tables
*(Add brief descriptions for your dimension tables here based on the models.py file!)*