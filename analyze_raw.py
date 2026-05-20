"""Analyze raw datasets to understand available columns."""
import pandas as pd

print("=== UCI Student Performance ===")
df1 = pd.read_csv("data/raw/uci_student_performance.csv")
print(f"Shape: {df1.shape}")
print(f"Columns: {df1.columns.tolist()}")
print(f"Nulls: {df1.isnull().sum().sum()}")
print(df1.head(2).to_string())
print()

print("=== UCI Dropout & Success ===")
df2 = pd.read_csv("data/raw/uci_dropout_success.csv")
print(f"Shape: {df2.shape}")
print(f"Columns: {df2.columns.tolist()}")
print(f"Nulls: {df2.isnull().sum().sum()}")
print(df2.head(2).to_string())
print()

print("=== Kaggle Student Performance ===")
df3 = pd.read_csv("data/raw/kaggle_student_performance.csv")
print(f"Shape: {df3.shape}")
print(f"Columns: {df3.columns.tolist()}")
print(f"Nulls: {df3.isnull().sum().sum()}")
print(df3.head(2).to_string())
print()

print("=== Post-transform check ===")
for name, path in [("uci_student", "data/processed/uci_student_clean.csv"),
                     ("uci_dropout", "data/processed/uci_dropout_clean.csv"),
                     ("kaggle", "data/processed/kaggle_clean.csv")]:
    df = pd.read_csv(path)
    print(f"\n{name} clean: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    nulls = df.isnull().sum()
    if nulls.sum() > 0:
        print(f"  Remaining nulls: {nulls[nulls > 0].to_dict()}")
    else:
        print(f"  No nulls remaining")
    if 'gpa' in df.columns:
        print(f"  GPA range: {df['gpa'].min():.2f} - {df['gpa'].max():.2f}")
    if 'absences' in df.columns:
        print(f"  Absences range: {df['absences'].min()} - {df['absences'].max()}")
    if 'risk_label' in df.columns:
        print(f"  risk_label dist: {df['risk_label'].value_counts().to_dict()}")
