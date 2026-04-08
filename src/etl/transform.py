import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

logger = logging.getLogger(__name__)

def engineer_risk_label(g3_grade):
    """Maps G3 grades to risk labels aligned with probation thresholds."""
    if pd.isna(g3_grade):
        return np.nan
    elif g3_grade < 10:
        return 1 # High Risk
    elif g3_grade <= 12:
        return 0 # Medium Risk (Grouping with Low for binary, or keep as multi-class)
    else:
        return 0 # Low Risk

def transform_data(raw_datasets):
    """Cleans, imputes, and engineers features for the warehouse."""
    logger.info("Starting transformation phase...")
    transformed_data = {}
    
    scaler = MinMaxScaler()
    le = LabelEncoder()
    
    for name, df in raw_datasets.items():
        logger.info(f"Transforming dataset: {name}...")
        df_clean = df.copy()
        
        # 1. Missing Value Strategy
        # Median for numerical, Mode for categorical
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns
        
        for col in num_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                
        for col in cat_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                
        # 2. Risk Label Engineering (Example using UCI Student G3 column)
        # Note: You will adjust 'G3' to match your harmonized column names
        if 'G3' in df_clean.columns:
            df_clean['risk_label'] = df_clean['G3'].apply(engineer_risk_label)
            logger.info(f"Engineered risk labels for {name}.")

        # 3. Feature Normalization
        # Example: Scaling a GPA/Grade column
        grade_columns = [col for col in df_clean.columns if 'G3' in col or 'GPA' in col]
        if grade_columns:
            df_clean[grade_columns] = scaler.fit_transform(df_clean[grade_columns])
            
        transformed_data[name] = df_clean
        
    logger.info("Transformation phase complete.")
    return transformed_data