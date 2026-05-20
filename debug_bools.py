"""Quick debug of boolean columns in the warehouse."""
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("sqlite:///warehouse.db")

# Check what's actually in the database for boolean columns
print("=== dim_student boolean columns ===")
df = pd.read_sql("SELECT internet_access, paid_classes FROM dim_student LIMIT 20", engine)
print(df)
print(df.dtypes)
print(f"internet_access unique: {df['internet_access'].unique()}")
print(f"paid_classes unique: {df['paid_classes'].unique()}")

print("\n=== dim_demographics boolean columns ===")
df2 = pd.read_sql("SELECT scholarship_flag FROM dim_demographics LIMIT 20", engine)
print(df2)
print(f"scholarship_flag unique: {df2['scholarship_flag'].unique()}")

print("\n=== fact_student_risk boolean columns ===")
df3 = pd.read_sql("SELECT extracurricular FROM fact_student_risk LIMIT 20", engine)
print(df3)
print(f"extracurricular unique: {df3['extracurricular'].unique()}")

# Check full gold query
df4 = pd.read_sql("""
    SELECT s.internet_access, s.paid_classes, f.extracurricular, d.scholarship_flag
    FROM fact_student_risk f
    JOIN dim_student s ON f.student_id = s.student_id
    JOIN dim_demographics d ON f.demo_id = d.demo_id
    LIMIT 20
""", engine)
print("\n=== Gold query boolean columns ===")
print(df4)
print(df4.dtypes)
for col in df4.columns:
    print(f"  {col}: null={df4[col].isnull().sum()}, unique={df4[col].unique()}")
