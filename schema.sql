
CREATE TABLE dim_course (
	course_id INTEGER NOT NULL, 
	subject VARCHAR(50), 
	school VARCHAR(50), 
	study_time INTEGER, 
	failures_history INTEGER, 
	extracurricular BOOLEAN, 
	PRIMARY KEY (course_id)
)

;


CREATE TABLE dim_demographics (
	demo_id INTEGER NOT NULL, 
	nationality VARCHAR(50), 
	socioeconomic_tier VARCHAR(20), 
	first_gen_student BOOLEAN, 
	scholarship_flag BOOLEAN, 
	PRIMARY KEY (demo_id)
)

;


CREATE TABLE dim_semester (
	semester_id INTEGER NOT NULL, 
	year INTEGER, 
	period VARCHAR(2), 
	checkpoint_week VARCHAR(20), 
	PRIMARY KEY (semester_id)
)

;


CREATE TABLE dim_student (
	student_id INTEGER NOT NULL, 
	age INTEGER, 
	sex VARCHAR(1), 
	address VARCHAR(10), 
	family_size VARCHAR(10), 
	parent_edu VARCHAR(50), 
	internet_access BOOLEAN, 
	paid_classes BOOLEAN, 
	PRIMARY KEY (student_id)
)

;


CREATE TABLE fact_student_risk (
	fact_id INTEGER NOT NULL, 
	student_id INTEGER NOT NULL, 
	course_id INTEGER NOT NULL, 
	semester_id INTEGER NOT NULL, 
	demo_id INTEGER NOT NULL, 
	gpa FLOAT, 
	absences INTEGER, 
	risk_score FLOAT, 
	risk_label INTEGER, 
	checkpoint_week INTEGER, 
	PRIMARY KEY (fact_id), 
	FOREIGN KEY(student_id) REFERENCES dim_student (student_id), 
	FOREIGN KEY(course_id) REFERENCES dim_course (course_id), 
	FOREIGN KEY(semester_id) REFERENCES dim_semester (semester_id), 
	FOREIGN KEY(demo_id) REFERENCES dim_demographics (demo_id)
)

;

