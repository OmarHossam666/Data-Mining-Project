import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

def transform_data(raw_datasets):
    """Cleans, imputes, and engineers features for the warehouse."""
    logger.info("Starting transformation phase with full target mapping...")
    transformed_data = {}

    for name, df in raw_datasets.items():
        logger.info(f"Transforming dataset: {name}...")
        df_clean = df.copy()

        # 0. Canonical Schema Mapping
        if name == 'kaggle':
            df_clean = df_clean.rename(columns={
                'Age': 'age', 'Gender': 'sex', 'ParentalEducation': 'parent_edu',
                'StudyTimeWeekly': 'study_time', 'Absences': 'absences',
                'Extracurricular': 'extracurricular', 'Tutoring': 'paid_classes',
                'GPA': 'gpa'
            })
            if 'sex' in df_clean.columns:
                df_clean['sex'] = df_clean['sex'].map({0: 'F', 1: 'M'}).fillna(df_clean['sex'])
        elif name == 'uci_student':
            df_clean = df_clean.rename(columns={
                'famsize': 'family_size', 'internet': 'internet_access',
                'paid': 'paid_classes', 'studytime': 'study_time',
                'failures': 'failures_history', 'activities': 'extracurricular'
            })
        elif name == 'uci_dropout':
            df_clean = df_clean.rename(columns={
                'Age at enrollment': 'age', 'Gender': 'sex',
                'Curricular units 2nd sem (grade)': 'gpa',
                'Course': 'subject', 'Scholarship holder': 'scholarship_flag',
                'Nacionality': 'nationality'
            })
            if 'sex' in df_clean.columns:
                df_clean['sex'] = df_clean['sex'].map({1: 'M', 0: 'F'}).fillna(df_clean['sex'])

        bool_cols = ['internet_access', 'paid_classes', 'extracurricular', 'first_gen_student', 'scholarship_flag']
        for col in bool_cols:
            if col in df_clean.columns:
                # Convert explicitly to boolean mapping safely
                df_clean[col] = df_clean[col].map({'yes': True, 'no': False, 1: True, 0: False, 'Yes': True, 'No': False, True: True, False: False}).astype(bool)

        # 1. Missing Value Strategy (Median/Mode) — with edge-case protection
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns

        for col in num_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    logger.warning(f"Column '{col}' in {name} is entirely null — dropping.")
                    df_clean = df_clean.drop(columns=[col])
                else:
                    df_clean[col] = df_clean[col].fillna(median_val)

        for col in cat_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_result = df_clean[col].mode()
                if mode_result.empty:
                    logger.warning(f"Column '{col}' in {name} is entirely null — dropping.")
                    df_clean = df_clean.drop(columns=[col])
                else:
                    df_clean[col] = df_clean[col].fillna(mode_result[0])

        # 2. TRUE Target Engineering for ALL Datasets
        if name == 'uci_student':
            # High Risk if G3 grade is less than 10
            if 'G3' in df_clean.columns:
                df_clean['risk_label'] = df_clean['G3'].apply(lambda x: 1 if x < 10 else 0)

        elif name == 'uci_dropout':
            # High Risk if they actually Dropped Out
            if 'Target' in df_clean.columns:
                df_clean['risk_label'] = df_clean['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)

        elif name == 'kaggle':
            # High Risk if GradeClass is 3 (D) or 4 (F)
            if 'GradeClass' in df_clean.columns:
                df_clean['risk_label'] = df_clean['GradeClass'].apply(lambda x: 1 if x >= 3 else 0)

        if 'risk_label' in df_clean.columns:
            logger.info(f"Successfully engineered real risk labels for {name}.")
        else:
            logger.warning(f"Could not find target column to engineer risk label for {name}!")

        # 3. Feature Normalization — per-dataset scaler (intentionally independent per source)
        grade_columns = [col for col in df_clean.columns if 'G3' in col or 'GPA' in col]
        if grade_columns:
            scaler = MinMaxScaler()
            df_clean[grade_columns] = scaler.fit_transform(df_clean[grade_columns])

        transformed_data[name] = df_clean

    logger.info("Transformation phase complete.")
    return transformed_data