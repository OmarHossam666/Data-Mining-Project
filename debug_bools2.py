"""Debug gold layer boolean mapping."""
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("sqlite:///warehouse.db")

query = """
SELECT s.internet_access, s.paid_classes, f.extracurricular, d.scholarship_flag
FROM fact_student_risk f
JOIN dim_student s ON f.student_id = s.student_id
JOIN dim_demographics d ON f.demo_id = d.demo_id
"""

df = pd.read_sql(query, engine)
print(f"Total rows: {len(df)}")
for col in df.columns:
    print(f"  {col}: null={df[col].isnull().sum()}, dtype={df[col].dtype}, unique={df[col].unique()[:10]}")

# Test the map operation
print("\n=== Testing map ===")
test = df['internet_access'].copy()
result = test.map({True: "Yes", False: "No", 1: "Yes", 0: "No", 1.0: "Yes", 0.0: "No"}).fillna("N/A")
print(f"After map: {result.value_counts().to_dict()}")

# UCI student rows only (first 649)
uci_rows = df.iloc[:649]
print(f"\nUCI Student rows internet_access null: {uci_rows['internet_access'].isnull().sum()}")
print(f"UCI Student rows internet_access unique: {uci_rows['internet_access'].unique()}")

# UCI dropout rows (next 4424)
dropout_rows = df.iloc[649:649+4424]
print(f"\nUCI Dropout rows internet_access null: {dropout_rows['internet_access'].isnull().sum()}")
print(f"UCI Dropout rows internet_access unique: {dropout_rows['internet_access'].unique()[:5]}")

# Kaggle rows (last 2392)
kaggle_rows = df.iloc[649+4424:]
print(f"\nKaggle rows internet_access null: {kaggle_rows['internet_access'].isnull().sum()}")
print(f"Kaggle rows internet_access unique: {kaggle_rows['internet_access'].unique()[:5]}")
