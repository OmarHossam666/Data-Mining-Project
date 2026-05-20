"""Verify that the processed data is correct for feature engineering."""
import pandas as pd

# Check UCI dropout processed data — this is the primary ML dataset
df = pd.read_csv("data/processed/uci_dropout_clean.csv")
print(f"UCI Dropout clean: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# Verify GPA is normalized
if 'gpa' in df.columns:
    print(f"  GPA range: {df['gpa'].min():.4f} - {df['gpa'].max():.4f}")
    print(f"  GPA mean: {df['gpa'].mean():.4f}")
    print(f"  GPA nulls: {df['gpa'].isnull().sum()}")
    
# Verify risk_label is correct
if 'risk_label' in df.columns:
    print(f"  risk_label dist: {df['risk_label'].value_counts().to_dict()}")
    
# Check that key feature engineering columns still exist
fe_cols = ['Marital Status', 'Application mode', 'Application order', 'subject',
           'Daytime/evening attendance', 'Previous qualification',
           'Previous qualification (grade)', 'nationality',
           "Mother's qualification", "Father's qualification",
           "Mother's occupation", "Father's occupation",
           'Admission grade', 'Displaced', 'Educational special needs',
           'Debtor', 'Tuition fees up to date', 'sex', 'scholarship_flag',
           'age', 'International']

found = [c for c in fe_cols if c in df.columns]
missing = [c for c in fe_cols if c not in df.columns]
print(f"\n  Feature engineering columns found: {len(found)}/{len(fe_cols)}")
if missing:
    print(f"  MISSING: {missing}")
else:
    print(f"  All enrollment features present!")
    
# Check semester features
sem1_cols = ['Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
             'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
             'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)']
sem1_found = [c for c in sem1_cols if c in df.columns]
print(f"  Semester 1 features found: {len(sem1_found)}/{len(sem1_cols)}")

sem2_cols = ['Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
             'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
             'gpa', 'Curricular units 2nd sem (without evaluations)',
             'Unemployment rate', 'Inflation rate', 'GDP']
sem2_found = [c for c in sem2_cols if c in df.columns]
print(f"  Semester 2 features found: {len(sem2_found)}/{len(sem2_cols)}")
