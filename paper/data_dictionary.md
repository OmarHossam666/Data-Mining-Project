# ProGrade Data Warehouse: Data Dictionary

## Architecture Overview
This database utilizes a **Star Schema** designed for analytical queries.

**Grain:** One row per student-course observation.

## 1. Fact Table: `fact_student_risk`
Contains the core numeric measurements and foreign keys linking to dimensions.
* `fact_id` (PK, Int): Surrogate key for the fact record.
* `student_id` (FK, Int): Link to `dim_student`.
* `course_id` (FK, Int): Link to `dim_course`.
* `demo_id` (FK, Int): Link to `dim_demographics`.
* `gpa` (Float): Grade Point Average normalized to a 0.0–4.0 scale.
* `absences` (Int): Total absences accumulated. Available for UCI Student and Kaggle datasets; not available for UCI Dropout.
* `risk_score` (Float): Machine Learning probability score (0.0 to 1.0) of academic failure. Populated after ML model inference.
* `risk_label` (Int): Binary classification (0 = Low Risk, 1 = High Risk).
* `study_time` (Float): Weekly study time. UCI Student uses an ordinal 1–4 scale; Kaggle uses hours; UCI Dropout does not have this field.
* `failures_history` (Int): Number of past class failures (UCI Student only).
* `extracurricular` (Boolean): Participation in extracurricular activities (UCI Student and Kaggle only).

## 2. Dimension Tables
### `dim_student`
Holds the personal, family, and social attributes of each student.
* `student_id` (PK, Int): Surrogate key uniquely identifying a student.
* `age` (Int): The student's age.
* `sex` (String): Gender of the student ('M' or 'F').
* `address` (String): Home address type ('U' for Urban, 'R' for Rural). UCI Student only.
* `family_size` (String): Family size indicator ('LE3' or 'GT3'). UCI Student only.
* `parent_edu` (String): Highest level of parental education. Derived as max(mother, father) education.
* `internet_access` (Boolean): Indicates whether the student has internet access at home. UCI Student only.
* `paid_classes` (Boolean): Indicates if the student attends extra paid classes. UCI Student and Kaggle only.

### `dim_course`
Stores contextual attributes related to the course/subject being evaluated.
* `course_id` (PK, Int): Surrogate key uniquely identifying a course context.
* `subject` (String): The subject or course name. UCI Student maps to school names (Gabriel Pereira, Mousinho da Silveira); UCI Dropout uses numeric course codes; Kaggle does not have this field.
* `school` (String): The dataset source identifier ('uci_student', 'uci_dropout', 'kaggle').

### `dim_demographics`
Contains broader demographic variables.
* `demo_id` (PK, Int): Surrogate key uniquely identifying a demographic profile.
* `nationality` (String): The student's nationality (UCI Dropout only).
* `scholarship_flag` (Boolean): Indicates if the student is a scholarship recipient (UCI Dropout only).

## 3. Gold Layer: `student_risk_mart.csv`
A comprehensive, canonical datamart combining facts and dimensions for downstream analytical consumption.

All GPA values are normalized to a **0.0–4.0 scale**:
- UCI Student: G3 final grade (0–20) → normalized to 0.0–4.0
- UCI Dropout: Curricular units 2nd sem grade (0–20) → normalized to 0.0–4.0
- Kaggle: GPA already on 0.0–4.0 scale

Columns:
* `student_id` (Int): Surrogate key.
* `fact_id` (Int): Surrogate key for the original fact record.
* `age` (Int): The student's age.
* `sex` (String): Gender ('Male' or 'Female').
* `address` (String): Home address type ('Urban' or 'Rural'). N/A for datasets without this field.
* `family_size` (String): Family size indicator ('LE3' or 'GT3'). N/A for datasets without this field.
* `parent_edu` (Int/String): Highest level of parental education.
* `internet_access` (String): 'Yes', 'No', or N/A.
* `paid_classes` (String): 'Yes', 'No', or N/A.
* `school` (String): The dataset source ('uci_student', 'uci_dropout', 'kaggle').
* `subject` (String): The subject or course name.
* `study_time` (Float): Weekly study time.
* `failures_history` (Int): Number of past class failures.
* `extracurricular` (String): 'Yes', 'No', or N/A.
* `nationality` (Float/String): The student's nationality code.
* `scholarship_flag` (String): 'Yes', 'No', or N/A.
* `gpa` (Float): Normalized GPA on a 0.0–4.0 scale.
* `gpa_bracket` (String): Qualitative GPA tier ('Excellent' ≥ 3.6, 'Good' ≥ 2.8, 'Average' ≥ 2.0, 'Poor' < 2.0).
* `absences` (Float): Total absences. N/A for datasets without this field.
* `attendance_tier` (String): Grouping by absences ('High Attendance (0-2)', 'Medium Attendance (3-8)', 'Low Attendance (>8)', 'N/A').
* `risk_score` (Float): ML probability score. Populated after model inference.
* `risk_label` (Int): Binary classification (0 = Low Risk, 1 = High Risk).
* `risk_status` (String): Human-readable risk status ('Low Risk' or 'High Risk').

## 4. Cross-Dataset Coverage Matrix

| Column | UCI Student (n=649) | UCI Dropout (n=4424) | Kaggle (n=2392) |
|---|---|---|---|
| age | ✅ | ✅ | ✅ |
| sex | ✅ | ✅ | ✅ |
| address | ✅ | ❌ | ❌ |
| family_size | ✅ | ❌ | ❌ |
| parent_edu | ✅ (Medu/Fedu) | ✅ (Mother's/Father's qual.) | ✅ |
| internet_access | ✅ | ❌ | ❌ |
| paid_classes | ✅ | ❌ | ✅ |
| subject | ✅ (school code) | ✅ (course code) | ❌ |
| study_time | ✅ | ❌ | ✅ |
| failures_history | ✅ | ❌ | ❌ |
| extracurricular | ✅ | ❌ | ✅ |
| nationality | ❌ | ✅ | ❌ |
| scholarship_flag | ❌ | ✅ | ❌ |
| gpa | ✅ (from G3) | ✅ (2nd sem grade) | ✅ |
| absences | ✅ | ❌ | ✅ |
| risk_label | ✅ | ✅ | ✅ |