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
### `dim_student`
Holds the personal, family, and social attributes of each student.
* `student_id` (PK, Int): Surrogate key uniquely identifying a student.
* `age` (Int): The student's age.
* `sex` (String): Gender of the student ('M' or 'F').
* `address` (String): Home address type ('U' for Urban, 'R' for Rural).
* `family_size` (String): Family size indicator ('LE3' for less/equal to 3, 'GT3' for greater than 3).
* `parent_edu` (String): Highest level of parental education.
* `internet_access` (Boolean): Indicates whether the student has internet access at home.
* `paid_classes` (Boolean): Indicates if the student attends extra paid classes.

### `dim_course`
Stores contextual and historical academic attributes related to the course/subject being evaluated.
* `course_id` (PK, Int): Surrogate key uniquely identifying a course context.
* `subject` (String): The subject being studied (e.g., 'Math', 'Portuguese').
* `school` (String): The school the student is attending.
* `study_time` (Int): Numeric scale representing weekly study time.
* `failures_history` (Int): Number of past class failures.
* `extracurricular` (Boolean): Indicates participation in extracurricular activities.

### `dim_semester`
Captures the temporal dimension of the academic life cycle, allowing analysis across different time periods.
* `semester_id` (PK, Int): Surrogate key uniquely identifying a semester/checkpoint period.
* `year` (Int): The academic year.
* `period` (String): The academic period or term (e.g., 'G1', 'G2', 'G3').
* `checkpoint_week` (String): The specific evaluation milestone (e.g., '4', '8', '12', 'final').

### `dim_demographics`
Contains broader socioeconomic and demographic variables.
* `demo_id` (PK, Int): Surrogate key uniquely identifying a demographic profile.
* `nationality` (String): The student's nationality.
* `socioeconomic_tier` (String): Classification of the student's socioeconomic status.
* `first_gen_student` (Boolean): Indicates if the student is the first in their family to attend higher education.
* `scholarship_flag` (Boolean): Indicates if the student is a scholarship recipient.