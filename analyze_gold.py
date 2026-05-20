"""Quick analysis of the gold layer output."""
import pandas as pd

df = pd.read_csv("data/gold/student_risk_mart.csv")
print("Shape:", df.shape)
print()
print("Columns:", df.columns.tolist())
print()
print("Null counts:")
print(df.isnull().sum())
print()
print("Dtypes:")
print(df.dtypes)
print()
print("First 5 rows:")
print(df.head().to_string())
print()
print("Value counts for key columns:")
for col in df.columns:
    null_pct = df[col].isnull().mean() * 100
    unique = df[col].nunique(dropna=False)
    print(f"  {col}: nulls={null_pct:.1f}%, unique={unique}")
    if null_pct > 0 and null_pct < 100:
        # Show sample of non-null values
        sample = df[col].dropna().head(3).tolist()
        print(f"    sample non-null: {sample}")
    if null_pct == 100:
        print(f"    *** COMPLETELY EMPTY ***")

print()
print("=== Checking for 'Unknown' values ===")
for col in df.select_dtypes(include='object').columns:
    unk_count = (df[col] == 'Unknown').sum()
    if unk_count > 0:
        print(f"  {col}: {unk_count} 'Unknown' values ({unk_count/len(df)*100:.1f}%)")

print()
print("=== School/source distribution ===")
print(df['school'].value_counts())
